// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _CAP_OB_SENSOR_UVC_STREAM_CHANNEL_HPP_
#define _CAP_OB_SENSOR_UVC_STREAM_CHANNEL_HPP_
#include "obsensor_stream_channel_interface.hpp"

#ifdef HAVE_OB_SENSOR
namespace cv
{
    namespace obsensor
    {

#define OB3D_CAM_PID 0x2bc5 // usb pid
        struct UvcDeviceInfo
        {
            std::string id = "";
            std::string name = "";
            std::string uid = "";
            uint16_t vid = 0;
            uint16_t pid = 0;
            uint16_t mi = 0;
        };

        typedef enum
        {
            STREAM_STOPED = 0, // stoped or ready
            STREAM_STARTING = 1,
            STREAM_STARTED = 2,
            STREAM_STOPPING = 3,
        } StreamState;

        StreamType parseUvcDeviceNameToStreamType(const std::string &devName);
        FrameFormat frameFourccToFormat(uint32_t fourcc);

    } // namespace obsensor
} // namespace cv
#endif // HAVE_OB_SENSOR
#endif // _CAP_OB_SENSOR_UVC_STREAM_CHANNEL_HPP_