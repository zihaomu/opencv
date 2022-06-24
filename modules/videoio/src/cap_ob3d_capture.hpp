// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _CAP_OB3D_CAPTURE__HPP_
#define _CAP_OB3D_CAPTURE__HPP_

#include <map>
#include <mutex>

#include "cap_ob3d/ob3d_stream_channel_interface.hpp"

#ifdef HAVE_OB3D
namespace cv
{
class VideoCapture_ob3d : public IVideoCapture
{
public:
    VideoCapture_ob3d(int index);
    virtual ~VideoCapture_ob3d(){};

    virtual double getProperty(int propIdx) const CV_OVERRIDE{
        // todo
        return 0.0;
    };
    virtual bool setProperty(int propIdx, double propVal) CV_OVERRIDE{
        // todo
        return false;
    };

    virtual bool grabFrame() CV_OVERRIDE{
        // todo
        return true;
    };
    virtual bool retrieveFrame(int outputType, OutputArray frame) CV_OVERRIDE;
    virtual int getCaptureDomain() CV_OVERRIDE{
        return CAP_OB3D;
    };
    virtual bool isOpened() const CV_OVERRIDE{
        return isOpened_;
    };

private:
    bool isOpened_;
    std::mutex frameSetMutex_;
    std::map<int, Mat> frameSet_;
    std::vector<Ptr<ob3d::IStreamChannel>> streamChannelGroup_;
    Mat depthFrame_;
    Mat irFrame_;
    Mat rgbFrame_;
};
} // namespace cv
#endif // HAVE_OB3D
#endif // _CAP_OB3D_CAPTURE__HPP_
