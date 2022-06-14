// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "cap_ob3d_capture.hpp"
#ifdef HAVE_OB3D
namespace cv
{
    Ptr<IVideoCapture> create_ob3d_capture(int index){
         return makePtr<VideoCapture_ob3d>(index);
    }

} // namespace cv
#endif // HAVE_OB3D