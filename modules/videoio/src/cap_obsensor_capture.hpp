// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _CAP_OBSENSOR_CAPTURE__HPP_
#define _CAP_OBSENSOR_CAPTURE__HPP_

#include <map>
#include <mutex>

#include "cap_obsensor/obsensor_stream_channel_interface.hpp"

#ifdef HAVE_OBSENSOR
namespace cv
{
class VideoCapture_obsensor : public IVideoCapture
{
public:
    VideoCapture_obsensor(int index);
    virtual ~VideoCapture_obsensor(){};

    virtual double getProperty(int propIdx) const CV_OVERRIDE;
    virtual bool setProperty(int propIdx, double propVal) CV_OVERRIDE;
    virtual bool grabFrame() CV_OVERRIDE;
    virtual bool retrieveFrame(int outputType, OutputArray frame) CV_OVERRIDE;
    virtual int getCaptureDomain() CV_OVERRIDE{
        return CAP_OBSENSOR;
    }
    virtual bool isOpened() const CV_OVERRIDE{
        return isOpened_;
    }

private:
    bool isOpened_;
    std::vector<std::shared_ptr<obsensor::IStreamChannel>> streamChannelGroup_;

    std::mutex frameMutex_;

    Mat depthFrame_;
    Mat irFrame_;
    Mat rgbFrame_;

    Mat grabbedDepthFrame_;
    Mat grabbedIrFrame_;
    Mat grabbedRgbFrame_;

    obsensor::CameraParam camParam_;
};
} // namespace cv
#endif // HAVE_OBSENSOR
#endif // _CAP_OBSENSOR_CAPTURE__HPP_
