// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "precomp.hpp"

#include "cap_ob3d_capture.hpp"
#include "cap_ob3d/ob3d_stream_channel_interface.hpp"
#ifdef HAVE_OB3D
namespace cv
{
    Ptr<IVideoCapture> create_ob3d_capture(int index)
    {
        return makePtr<VideoCapture_ob3d>(index);
    }

    VideoCapture_ob3d::VideoCapture_ob3d(int index) : isOpened_(false)
    {
        static const ob3d::StreamProfile rgbProfile = {1920, 1080, 30, ob3d::FRAME_FORMAT_MJPG};
        static const ob3d::StreamProfile depthProfile = {640, 480, 30, ob3d::FRAME_FORMAT_Y16};
        static const ob3d::StreamProfile irProfile = {640, 480, 30, ob3d::FRAME_FORMAT_Y16};

        streamChannelGroup_ = ob3d::getStreamChannelGroup(index);
        if (!streamChannelGroup_.empty())
        {
            for (auto &channel : streamChannelGroup_)
            {
                auto streamType = channel->streamType();
                switch (streamType)
                {
                case ob3d::OB3D_STREAM_RGB:
                    channel->start(rgbProfile, [&](ob3d::Frame *frame){
                        std::unique_lock<std::mutex> lk(frameSetMutex_);
                        rgbFrame_ = Mat( 1, frame->dataSize, CV_8UC1, frame->data ).clone(); 
                    });
                    break;
                case ob3d::OB3D_STREAM_DEPTH:
                    channel->start(depthProfile, [&](ob3d::Frame *frame){ 
                        std::unique_lock<std::mutex> lk(frameSetMutex_);
                        depthFrame_ =  Mat(frame->height, frame->width, CV_16UC1, frame->data, frame->width*2).clone(); 
                    });
                    break;
                case ob3d::OB3D_STREAM_IR:
                    channel->start(irProfile, [&](ob3d::Frame *frame){ 
                        std::unique_lock<std::mutex> lk(frameSetMutex_);
                        irFrame_ =  Mat(frame->height, frame->width, CV_16UC1, frame->data, frame->width*2).clone(); 
                    });
                    break;
                default:
                    break;
                }
            }
            isOpened_ = true;
        }
    }

    bool VideoCapture_ob3d::retrieveFrame(int outputType, OutputArray frame)
    {
        std::unique_lock<std::mutex> lk(frameSetMutex_);
        if (outputType == CAP_OB3D_DEPTH_MAP && !depthFrame_.empty())
        {
            depthFrame_.copyTo(frame);
            return true;
        }
        else if (outputType == CAP_OB3D_IR_IMAGE && !irFrame_.empty())
        {
            irFrame_.copyTo(frame);
            return true;
        }
        else if (outputType == CAP_OB3D_BGR_IMAGE && !rgbFrame_.empty())
        {
            imdecode(rgbFrame_, IMREAD_COLOR).copyTo(frame);
            return true;
        }
        return false;
    }

} // namespace cv
#endif // HAVE_OB3D