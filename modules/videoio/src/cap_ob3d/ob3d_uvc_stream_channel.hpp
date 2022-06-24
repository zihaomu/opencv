// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _CAP_OB3D_UVC_STREAM_CHANNEL_HPP_
#define _CAP_OB3D_UVC_STREAM_CHANNEL_HPP_
#include "ob3d_stream_channel_interface.hpp"

#ifdef HAVE_OB3D
namespace cv
{
    namespace ob3d
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

    } // namespace ob3d
} // namespace cv
#endif // HAVE_OB3D
#endif // _CAP_OB3D_UVC_STREAM_CHANNEL_HPP_