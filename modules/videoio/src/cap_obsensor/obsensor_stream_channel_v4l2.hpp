// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef _CAP_OBSENSOR_STREAM_CHANNEL_V4L2_HPP_
#define _CAP_OBSENSOR_STREAM_CHANNEL_V4L2_HPP_
#ifdef HAVE_OBSENSOR_V4L2

#include "obsensor_uvc_stream_channel.hpp"

#include <mutex>
#include <condition_variable>
#include <thread>

namespace cv{
namespace obsensor{
#define MAX_FRAME_BUFFER_NUM 4
    typedef struct
    {
        uint32_t length = 0;
        uint8_t *ptr = nullptr;
    } V4L2FrameBuffer;

    int xioctl(int fd, int req, void *arg);

    class V4L2Context
    {
    public:
        ~V4L2Context(){};
        static V4L2Context &getInstance();

        std::vector<UvcDeviceInfo> queryUvcDeviceInfoList();
        std::shared_ptr<IStreamChannel> createStreamChannel(const UvcDeviceInfo &devInfo);

    private:
        V4L2Context(){};
    };

    class V4L2StreamChannel : public IStreamChannel
    {
    public:
        V4L2StreamChannel(const UvcDeviceInfo &devInfo);
        virtual ~V4L2StreamChannel();

        virtual void start(const StreamProfile &profile, FrameCallback frameCallback) override;
        virtual void stop() override;
        virtual bool setProperty(int propId, const uint8_t *data, uint32_t dataSize) override
        {
            return false; // todo
        }
        virtual bool getProperty(int propId, uint8_t *recvData, uint32_t recvDataSize) override
        {
            return false; // todo
        }

        virtual StreamType streamType() const override;

    private:
        void grabFrame();

    private:
        const UvcDeviceInfo devInfo_;
        StreamType streamType_;
        int devFd_;

        V4L2FrameBuffer frameBuffList[MAX_FRAME_BUFFER_NUM];

        StreamState streamState_;
        std::mutex streamStateMutex_;
        std::condition_variable streamStateCv_;

        std::thread grabFrameThread_;

        FrameCallback frameCallback_;
        StreamProfile currentProfile_;
    };

} // namespace obsensor
} // namespace cv
#endif // HAVE_OBSENSOR_V4L2
#endif // _CAP_OBSENSOR_STREAM_CHANNEL_V4L2_HPP_