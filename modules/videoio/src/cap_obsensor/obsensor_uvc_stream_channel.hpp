// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _CAP_OBSENSOR_UVC_STREAM_CHANNEL_HPP_
#define _CAP_OBSENSOR_UVC_STREAM_CHANNEL_HPP_
#include "obsensor_stream_channel_interface.hpp"

#ifdef HAVE_OBSENSOR
namespace cv
{
    namespace obsensor
    {

#define OBSENSOR_CAM_PID 0x2bc5 // usb pid
#define XU_MAX_DATA_LENGTH 1024

        struct UvcDeviceInfo
        {
            std::string id = ""; // uvc sub-device id
            std::string name = "";
            std::string uid = ""; // parent usb device id
            uint16_t vid = 0;
            uint16_t pid = 0;
            uint16_t mi = 0; // uvc interface index
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
        uint32_t frameFormatToFourcc(FrameFormat);

        extern const uint8_t DEPTH_TO_COLOR_ALIGN_CMD0[16];
        extern const uint8_t DEPTH_TO_COLOR_ALIGN_CMD1[16];
        extern const uint8_t DEPTH_TO_COLOR_ALIGN_CMD2[16];
        extern const uint8_t DEPTH_TO_COLOR_ALIGN_CMD3[16];
        // class UvcStreamChannelContext{
        // }

    } // namespace obsensor
} // namespace cv
#endif // HAVE_OBSENSOR
#endif // _CAP_OBSENSOR_UVC_STREAM_CHANNEL_HPP_