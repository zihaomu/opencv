// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifdef HAVE_OB3D_V4L2
#include "ob3d_stream_channel_v4l2.hpp"

#include <sys/ioctl.h>

namespace cv{
namespace ob3d{
    std::vector<UvcDeviceInfo> queryUvcDeviceList(){
        // struct v4l2_capability caps = {0};
        // int r;
        // do r = ioctl (fd, VIDIOC_QUERYCAP, &caps);
        // while (-1 == r && EINTR == errno);
        return std::vector<UvcDeviceInfo>();
    }
} // namespace ob3d
} // namespace cv
#endif // HAVE_OB3D_V4L2