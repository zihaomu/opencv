// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "precomp.hpp"

#include "cap_obsensor_capture.hpp"
#include "cap_obsensor/obsensor_stream_channel_interface.hpp"
#ifdef HAVE_OBSENSOR
namespace cv
{
    Ptr<IVideoCapture> create_obsensor_capture(int index)
    {
        return makePtr<VideoCapture_obsensor>(index);
    }

    VideoCapture_obsensor::VideoCapture_obsensor(int index) : isOpened_(false)
    {
        static const obsensor::StreamProfile rgbProfile = {640, 480, 30, obsensor::FRAME_FORMAT_MJPG};
        static const obsensor::StreamProfile depthProfile = {640, 480, 30, obsensor::FRAME_FORMAT_Y16};
        static const obsensor::StreamProfile irProfile = {640, 480, 30, obsensor::FRAME_FORMAT_Y16};

        streamChannelGroup_ = obsensor::getStreamChannelGroup(index);
        if (!streamChannelGroup_.empty())
        {
            for (auto &channel : streamChannelGroup_)
            {
                auto streamType = channel->streamType();
                switch (streamType)
                {
                case obsensor::OBSENSOR_STREAM_RGB:
                    channel->start(rgbProfile, [&](obsensor::Frame *frame)
                                   {
                        std::unique_lock<std::mutex> lk(frameMutex_);
                        rgbFrame_ = Mat( 1, frame->dataSize, CV_8UC1, frame->data ).clone(); });
                    break;
                case obsensor::OBSENSOR_STREAM_DEPTH:
                    uint8_t data;
                    channel->setProperty(obsensor::DEPTH_TO_COLOR_ALIGN, &data, 1);
                    channel->start(depthProfile, [&](obsensor::Frame *frame)
                                   { 
                        std::unique_lock<std::mutex> lk(frameMutex_);
                        depthFrame_ =  Mat(frame->height, frame->width, CV_16UC1, frame->data, frame->width*2).clone(); });
                    break;
                case obsensor::OBSENSOR_STREAM_IR:
                    channel->start(irProfile, [&](obsensor::Frame *frame)
                                   { 
                        std::unique_lock<std::mutex> lk(frameMutex_);
                        irFrame_ =  Mat(frame->height, frame->width, CV_16UC1, frame->data, frame->width*2).clone(); });
                    break;
                default:
                    break;
                }
            }
            isOpened_ = true;
        }
    }

    bool VideoCapture_obsensor::grabFrame()
    {
        std::unique_lock<std::mutex> lk(frameMutex_);

        grabbedDepthFrame_ = depthFrame_;
        grabbedIrFrame_ = irFrame_;
        grabbedRgbFrame_ = rgbFrame_;

        depthFrame_.release();
        irFrame_.release();
        rgbFrame_.release();

        return !grabbedDepthFrame_.empty() || !grabbedIrFrame_.empty() || !grabbedRgbFrame_.empty();
    }

    bool VideoCapture_obsensor::retrieveFrame(int outputType, OutputArray frame)
    {
        std::unique_lock<std::mutex> lk(frameMutex_);
        switch (outputType)
        {
        case CAP_OBSENSOR_DEPTH_MAP:
            if (!grabbedDepthFrame_.empty())
            {
                grabbedDepthFrame_.copyTo(frame);
                grabbedDepthFrame_.release();
                return true;
            }
            break;
        case CAP_OBSENSOR_IR_IMAGE:
            if (!grabbedIrFrame_.empty())
            {
                grabbedIrFrame_.copyTo(frame);
                grabbedIrFrame_.release();
                return true;
            }
            break;
        case CAP_OBSENSOR_BGR_IMAGE:
            if (!grabbedRgbFrame_.empty())
            {
                auto mat = imdecode(grabbedRgbFrame_, IMREAD_COLOR);
                grabbedRgbFrame_.release();

                if (!mat.empty())
                {
                    mat.copyTo(frame);
                    return true;
                }
            }
            break;
        default:
            break;
        }

        return false;
    }

} // namespace cv
#endif // HAVE_OBSENSOR