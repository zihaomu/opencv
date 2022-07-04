// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifdef HAVE_OBSENSOR_V4L2
#include "obsensor_stream_channel_v4l2.hpp"

#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <linux/uvcvideo.h>
#include <fstream>
#include <map>
#include <vector>

#include "opencv2/core/utils/filesystem.hpp"

namespace cv{
namespace obsensor{

#define IOCTL_FAILED_RETURN(x)                                 \
    if (x < 0)                                                 \
    {                                                          \
        CV_LOG_WARNING(NULL, "ioctl error return: " << errno); \
        return;                                                \
    }

#define IOCTL_FAILED_LOG(x)                                    \
    if (x < 0)                                                 \
    {                                                          \
        CV_LOG_WARNING(NULL, "ioctl error return: " << errno); \
    }

#define IOCTL_FAILED_CONTINUE(x)                               \
    if (x < 0)                                                 \
    {                                                          \
        CV_LOG_WARNING(NULL, "ioctl error return: " << errno); \
        continue;                                              \
    }

#define IOCTL_FAILED_EXEC(x, statement)                        \
    if (x < 0)                                                 \
    {                                                          \
        CV_LOG_WARNING(NULL, "ioctl error return: " << errno); \
        statement;                                             \
    }

    int xioctl(int fd, int req, void *arg)
    {
        int rst;
        int retry = 5;
        do
        {
            rst = ioctl(fd, req, arg);
            retry--;
        } while (rst == -1 && (errno == EAGAIN || (errno == EBUSY && retry > 0)));

        if (rst < 0)
        {
            CV_LOG_WARNING(NULL, "ioctl: fd=" << fd << ", req=" << req);
        }
        return rst;
    }

    V4L2Context &V4L2Context::getInstance()
    {
        static V4L2Context instance;
        return instance;
    }

    std::vector<UvcDeviceInfo> V4L2Context::queryUvcDeviceInfoList()
    {
        std::vector<UvcDeviceInfo> uvcDevList;
        std::map<std::string, UvcDeviceInfo> uvcDevMap;
        const cv::String videosDir = "/sys/class/video4linux";
        cv::utils::Paths videos;
        if (cv::utils::fs::isDirectory(videosDir))
        {
            cv::utils::fs::glob(videosDir, "*", videos, false, true);
            for (const auto &video : videos)
            {
                UvcDeviceInfo uvcDev;
                cv::String videoName = video.substr(video.find_last_of("/") + 1);
                char buf[PATH_MAX];
                if (realpath(video.c_str(), buf) == nullptr || cv::String(buf).find("virtual") != std::string::npos)
                {
                    continue;
                }
                cv::String videoRealPath = buf;
                cv::String interfaceRealPath = videoRealPath.substr(0, videoRealPath.find_last_of("/"));

                std::string busNum, devNum, devPath;
                while (videoRealPath.find_last_of("/") != std::string::npos)
                {
                    videoRealPath = videoRealPath.substr(0, videoRealPath.find_last_of("/"));
                    if (!(std::ifstream(videoRealPath + "/busnum") >> busNum))
                    {
                        continue;
                    }
                    if (!(std::ifstream(videoRealPath + "/devpath") >> devPath))
                    {
                        continue;
                    }
                    if (!(std::ifstream(videoRealPath + "/devnum") >> devNum))
                    {
                        continue;
                    }
                    uvcDev.uid = busNum + "-" + devPath + "-" + devNum;
                    break;
                    /* code */
                }

                uvcDev.id = cv::String("/dev/") + videoName;
                v4l2_capability caps = {};
                int videoFd = open(uvcDev.id.c_str(), O_RDONLY);
                IOCTL_FAILED_EXEC(xioctl(videoFd, VIDIOC_QUERYCAP, &caps), {
                    close(videoFd);
                    continue;
                });
                close(videoFd);

                if (caps.capabilities & V4L2_CAP_VIDEO_CAPTURE)
                {
                    cv::String modalias;
                    if (!(std::ifstream(video + "/device/modalias") >> modalias) ||
                        modalias.size() < 14 ||
                        modalias.substr(0, 5) != "usb:v" ||
                        modalias[9] != 'p')
                    {
                        continue;
                    }
                    std::istringstream(modalias.substr(5, 4)) >> std::hex >> uvcDev.vid;
                    std::istringstream(modalias.substr(10, 4)) >> std::hex >> uvcDev.pid;
                    std::getline(std::ifstream(video + "/device/interface"), uvcDev.name);
                    std::ifstream(video + "/device/bInterfaceNumber") >> uvcDev.mi;
                    uvcDevMap.insert({interfaceRealPath, uvcDev});
                }
            }
        }
        for (const auto &item : uvcDevMap)
        {
            const auto uvcDev = item.second; // alias
            CV_LOG_INFO(NULL, "UVC device found: name=" << uvcDev.name << ", vid=" << uvcDev.vid << ", pid=" << uvcDev.pid << ", mi=" << uvcDev.mi << ", uid=" << uvcDev.uid << ", id=" << uvcDev.id);
            uvcDevList.push_back(uvcDev);
        }
        return uvcDevList;
    }

    std::shared_ptr<IStreamChannel> V4L2Context::createStreamChannel(const UvcDeviceInfo &devInfo)
    {
        return std::make_shared<V4L2StreamChannel>(devInfo);
    }

    V4L2StreamChannel::V4L2StreamChannel(const UvcDeviceInfo &devInfo) : devInfo_(devInfo),
                                                                         streamType_(parseUvcDeviceNameToStreamType(devInfo_.name)),
                                                                         devFd_(-1),
                                                                         streamState_(STREAM_STOPED),
                                                                         xuRecvBuf_(nullptr),
                                                                         xuSendBuf_(nullptr)
    {

        devFd_ = open(devInfo_.id.c_str(), O_RDWR | O_NONBLOCK, 0);
        if (devFd_ < 0)
        {
            CV_LOG_ERROR(NULL, "Open " << devInfo_.id << " failed ! errno=" << errno)
        }
        else{
            if(streamType_==OBSENSOR_STREAM_DEPTH && setXu(2, DEPTH_TO_COLOR_ALIGN_CMD4, sizeof(DEPTH_TO_COLOR_ALIGN_CMD4))){
                uint8_t *rcvData;
                uint32_t rcvLen;
                if(getXu(1, &rcvData, &rcvLen) && rcvData[8]==0&&rcvData[8]==0){
                    depthFrameProcessor_ = std::make_shared<DepthFrameProcessor>(*(OBExtensionParam*)(rcvData+10));
                }
            }
        }
    }

    V4L2StreamChannel::~V4L2StreamChannel() noexcept
    {
        stop();
        if (devFd_)
        {
            close(devFd_);
            devFd_ = -1;
        }
        if (xuRecvBuf_ != nullptr)
        {
            delete[] xuRecvBuf_;
        }
        if (xuSendBuf_ != nullptr)
        {
            delete[] xuSendBuf_;
        }
    }

    void V4L2StreamChannel::start(const StreamProfile &profile, FrameCallback frameCallback)
    {
        if (streamState_ != STREAM_STOPED)
        {
            CV_LOG_ERROR(NULL, devInfo_.id << ": repetitive operation!")
            return;
        }
        frameCallback_ = frameCallback;
        currentProfile_ = profile;

        struct v4l2_format fmt = {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = profile.width;
        fmt.fmt.pix.height = profile.height;
        fmt.fmt.pix.pixelformat = frameFormatToFourcc(profile.format);
        IOCTL_FAILED_RETURN(xioctl(devFd_, VIDIOC_S_FMT, &fmt));
        IOCTL_FAILED_RETURN(xioctl(devFd_, VIDIOC_G_FMT, &fmt));

        struct v4l2_streamparm streamParm = {};
        streamParm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        streamParm.parm.capture.timeperframe.numerator = 1;
        streamParm.parm.capture.timeperframe.denominator = profile.fps;
        IOCTL_FAILED_RETURN(xioctl(devFd_, VIDIOC_S_PARM, &streamParm));
        IOCTL_FAILED_RETURN(xioctl(devFd_, VIDIOC_G_PARM, &streamParm));

        struct v4l2_requestbuffers req = {};
        req.count = MAX_FRAME_BUFFER_NUM;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        IOCTL_FAILED_RETURN(xioctl(devFd_, VIDIOC_REQBUFS, &req));

        for (uint32_t i = 0; i < req.count && i < MAX_FRAME_BUFFER_NUM; i++)
        {
            struct v4l2_buffer buf = {};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i; // only one buffer
            IOCTL_FAILED_RETURN(xioctl(devFd_, VIDIOC_QUERYBUF, &buf));
            frameBuffList[i].ptr = (uint8_t *)mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, devFd_, buf.m.offset);
            frameBuffList[i].length = buf.length;
        }

        // stream on
        std::unique_lock<std::mutex> lk(streamStateMutex_);
        streamState_ = STREAM_STARTING;
        uint32_t type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        IOCTL_FAILED_EXEC(xioctl(devFd_, VIDIOC_STREAMON, &type), {
            streamState_ = STREAM_STOPED;
            for (uint32_t i = 0; i < MAX_FRAME_BUFFER_NUM; i++)
            {
                if (frameBuffList[i].ptr)
                {
                    munmap(frameBuffList[i].ptr, frameBuffList[i].length);
                    frameBuffList[i].ptr = nullptr;
                    frameBuffList[i].length = 0;
                }
            }
            return;
        });
        grabFrameThread_ = std::thread(&V4L2StreamChannel::grabFrame, this);
    }

    bool V4L2StreamChannel::setProperty(int propId, const uint8_t *data, uint32_t dataSize)
    {
        uint8_t *rcvData;
        uint32_t rcvLen;
        bool rst = true;
        switch (propId)
        {
        case DEPTH_TO_COLOR_ALIGN:
            rst &= setXu(2, DEPTH_TO_COLOR_ALIGN_CMD0, sizeof(DEPTH_TO_COLOR_ALIGN_CMD0));
            rst &= getXu(2, &rcvData, &rcvLen);
            rst &= setXu(2, DEPTH_TO_COLOR_ALIGN_CMD1, sizeof(DEPTH_TO_COLOR_ALIGN_CMD1));
            rst &= getXu(2, &rcvData, &rcvLen);
            rst &= setXu(2, DEPTH_TO_COLOR_ALIGN_CMD2, sizeof(DEPTH_TO_COLOR_ALIGN_CMD2));
            rst &= getXu(2, &rcvData, &rcvLen);
            rst &= setXu(2, DEPTH_TO_COLOR_ALIGN_CMD3, sizeof(DEPTH_TO_COLOR_ALIGN_CMD3));
            rst &= getXu(2, &rcvData, &rcvLen);
            break;
        default:
            break;
        }
        return rst;
    }

    bool V4L2StreamChannel::getProperty(int propId, uint8_t *recvData, uint32_t recvDataSize)
    {
        return false; // todo
    }

    void V4L2StreamChannel::grabFrame()
    {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(devFd_, &fds);
        struct timeval tv = {};
        tv.tv_sec = 0;
        tv.tv_usec = 100000; // 100ms

        struct v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        // buf.index = 0;

        IOCTL_FAILED_EXEC(xioctl(devFd_, VIDIOC_QBUF, &buf), {
            std::unique_lock<std::mutex> lk(streamStateMutex_);
            streamState_ = STREAM_STOPED;
            streamStateCv_.notify_all();
            return;
        });

        while (streamState_ == STREAM_STARTING || streamState_ == STREAM_STARTED)
        {
            IOCTL_FAILED_CONTINUE(select(devFd_ + 1, &fds, NULL, NULL, &tv));
            IOCTL_FAILED_CONTINUE(xioctl(devFd_, VIDIOC_DQBUF, &buf));
            if (streamState_ == STREAM_STARTING)
            {
                std::unique_lock<std::mutex> lk(streamStateMutex_);
                streamState_ = STREAM_STARTED;
                streamStateCv_.notify_all();
            }
            Frame fo = {currentProfile_.format, currentProfile_.width, currentProfile_.height, buf.length, frameBuffList[buf.index].ptr};
            if(depthFrameProcessor_){
                depthFrameProcessor_->process(&fo);
            }
            frameCallback_(&fo);
            IOCTL_FAILED_CONTINUE(xioctl(devFd_, VIDIOC_QBUF, &buf));
        }
        std::unique_lock<std::mutex> lk(streamStateMutex_);
        streamState_ = STREAM_STOPED;
        streamStateCv_.notify_all();
    }

    bool V4L2StreamChannel::setXu(uint8_t ctrl, const uint8_t *data, uint32_t len)
    {

        if (xuSendBuf_ == nullptr)
        {
            xuSendBuf_ = new uint8_t[XU_MAX_DATA_LENGTH];
        }
        memcpy(xuSendBuf_, data, len);
        struct uvc_xu_control_query xu_ctrl_query =
            {
                .unit = XU_UNIT_ID,
                .selector = ctrl,
                .query = UVC_CTRL_FLAG_SET_CUR,
                .size = ctrl == 1 ? 512 : (ctrl == 2 ? 64 : 1024),
                .data = xuSendBuf_};
        if (devFd_ > 0)
        {
            IOCTL_FAILED_EXEC(xioctl(devFd_, UVCIOC_CTRL_QUERY, &xu_ctrl_query), { return false; });
        }
        return true;
    }

    bool V4L2StreamChannel::getXu(uint8_t ctrl, uint8_t **data, uint32_t *len)
    {
        if (xuRecvBuf_ == nullptr)
        {
            xuRecvBuf_ = new uint8_t[XU_MAX_DATA_LENGTH];
        }
        struct uvc_xu_control_query xu_ctrl_query =
            {
                .unit = XU_UNIT_ID,
                .selector = ctrl,
                .query = UVC_CTRL_FLAG_GET_CUR,
                .size = ctrl == 1 ? 512 : (ctrl == 2 ? 64 : 1024),
                .data = xuRecvBuf_};

        IOCTL_FAILED_EXEC(xioctl(devFd_, UVCIOC_CTRL_QUERY, &xu_ctrl_query), {
            *len = 0;
            return false; });

        *len = xu_ctrl_query.size;
        *data = xuRecvBuf_;
        return true;
    }

    void V4L2StreamChannel::stop()
    {
        if (streamState_ == STREAM_STARTING || streamState_ == STREAM_STARTED)
        {
            streamState_ = STREAM_STOPPING;
            std::unique_lock<std::mutex> lk(streamStateMutex_);
            streamStateCv_.wait_for(lk, std::chrono::milliseconds(1000), [&]()
                                    { return streamState_ == STREAM_STOPED; });

            uint32_t type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            IOCTL_FAILED_LOG(xioctl(devFd_, VIDIOC_STREAMOFF, &type));
        }
        if (grabFrameThread_.joinable())
        {
            grabFrameThread_.join();
        }
        for (uint32_t i = 0; i < MAX_FRAME_BUFFER_NUM; i++)
        {
            if (frameBuffList[i].ptr)
            {
                munmap(frameBuffList[i].ptr, frameBuffList[i].length);
                frameBuffList[i].ptr = nullptr;
                frameBuffList[i].length = 0;
            }
        }
    }

    StreamType V4L2StreamChannel::streamType() const
    {
        return streamType_;
    }

} // namespace obsensor
} // namespace cv
#endif // HAVE_OBSENSOR_V4L2