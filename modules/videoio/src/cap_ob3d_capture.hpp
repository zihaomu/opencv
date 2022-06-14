// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _CAP_OB3D_CAPTURE__HPP_
#define _CAP_OB3D_CAPTURE__HPP_
#ifdef HAVE_OB3D

namespace cv
{
class VideoCapture_ob3d : public IVideoCapture
{
public:
    VideoCapture_ob3d(int index){};
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
        return false;
    };
    virtual bool retrieveFrame(int outputType, OutputArray frame) CV_OVERRIDE{
        // todo
        return false;
    };
    virtual int getCaptureDomain() CV_OVERRIDE{
        // todo
        return -1;
    };
    virtual bool isOpened() const CV_OVERRIDE{
        // todo
        return false;
    };
};
} // namespace cv
#endif // HAVE_OB3D
#endif // _CAP_OB3D_CAPTURE__HPP_
