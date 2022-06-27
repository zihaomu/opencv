// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef _CAP_OB_SENSOR_STREAM_CHANNEL_V4L2_HPP_
#define _CAP_OB_SENSOR_STREAM_CHANNEL_V4L2_HPP_
#ifdef HAVE_OB_SENSOR_V4L2

#include "obsensor_uvc_stream_channel.hpp"
namespace cv{
namespace obsensor{
    std::vector<UvcDeviceInfo> queryUvcDeviceList();
} // namespace obsensor
} // namespace cv
#endif // HAVE_OB_SENSOR_V4L2
#endif // _CAP_OB_SENSOR_STREAM_CHANNEL_V4L2_HPP_