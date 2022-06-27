// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifdef HAVE_OB_SENSOR_V4L2
#include "obsensor_stream_channel_v4l2.hpp"

#include <sys/ioctl.h>

namespace cv{
namespace obsensor{
    std::vector<UvcDeviceInfo> queryUvcDeviceList(){
        // struct v4l2_capability caps = {0};
        // int r;
        // do r = ioctl (fd, VIDIOC_QUERYCAP, &caps);
        // while (-1 == r && EINTR == errno);
        return std::vector<UvcDeviceInfo>();
    }
} // namespace obsensor
} // namespace cv
#endif // HAVE_OB_SENSOR_V4L2