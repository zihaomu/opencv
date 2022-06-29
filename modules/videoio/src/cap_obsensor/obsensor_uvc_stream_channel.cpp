// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#if defined(HAVE_OB_SENSOR_V4L2) ||  defined(HAVE_OB_SENSOR_MSMF)

#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include<iterator>

#if defined(HAVE_OB_SENSOR_V4L2)
#include "obsensor_stream_channel_v4l2.hpp"
#elif defined(HAVE_OB_SENSOR_MSMF)
#include "obsensor_stream_channel_msmf.hpp"
#endif // HAVE_OB_SENSOR_V4L2

namespace cv
{
    namespace obsensor
    {

#define fourCc2Int(a, b, c, d)\ 
    ((uint32_t)(a) | ((uint32_t)(b) << 8) | ((uint32_t)(c) << 16) | ((uint32_t)(d) << 24))

        const std::map<uint32_t, FrameFormat> fourccToOBFormat = {
            // {fourCc2Int('U', 'Y', 'V', 'Y'), FRAME_FORMAT_UYVY},
            {fourCc2Int('Y', 'U', 'Y', '2'), FRAME_FORMAT_YUYV},
            // {fourCc2Int('N', 'V', '1', '2'), FRAME_FORMAT_NV12},
            // {fourCc2Int('N', 'V', '2', '1'), FRAME_FORMAT_NV21},
            {fourCc2Int('M', 'J', 'P', 'G'), FRAME_FORMAT_MJPG},
            // {fourCc2Int('H', '2', '6', '4'), FRAME_FORMAT_H264},
            // {fourCc2Int('H', '2', '6', '5'), FRAME_FORMAT_H265},
            // {fourCc2Int('Y', '1', '2', ' '), FRAME_FORMAT_Y12},
            {fourCc2Int('Y', '1', '6', ' '), FRAME_FORMAT_Y16},
            // {fourCc2Int('G', 'R', 'A', 'Y'), FRAME_FORMAT_GRAY},
            // {fourCc2Int('Y', '1', '1', ' '), FRAME_FORMAT_Y11},
            // {fourCc2Int('Y', '8', ' ', ' '), FRAME_FORMAT_Y8},
            // {fourCc2Int('Y', '1', '0', ' '), FRAME_FORMAT_Y10},
            // {fourCc2Int('H', 'E', 'V', 'C'), FRAME_FORMAT_HEVC},
            // {fourCc2Int('Y', '1', '4', ' '), FRAME_FORMAT_Y14},
            // {fourCc2Int('I', '4', '2', '0'), FRAME_FORMAT_I420},
        };

        StreamType parseUvcDeviceNameToStreamType(const std::string &devName)
        {
            std::string uvcDevName = devName;
            std::transform(begin(uvcDevName), end(uvcDevName), begin(uvcDevName), ::tolower);
            if (uvcDevName.find(" depth") != std::string::npos)
            {
                return OB3D_STREAM_DEPTH;
            }
            else if (uvcDevName.find(" ir") != std::string::npos)
            {
                return OB3D_STREAM_IR;
            }

            return OB3D_STREAM_RGB; // else
        }

        FrameFormat frameFourccToFormat(uint32_t fourcc)
        {
            for (const auto &item : fourccToOBFormat)
            {
                if (item.first == fourcc)
                {
                    return item.second;
                }
            }
            return FRAME_FORMAT_UNKNOWN;
        }

        uint32_t frameFormatToForucc(FrameFormat fmt)
        {
            for (const auto &item : fourccToOBFormat)
            {
                if (item.second == fmt)
                {
                    return item.first;
                }
            }
            return 0;
        }

        std::vector<std::shared_ptr<IStreamChannel>> getStreamChannelGroup(uint32_t groupIdx)
        {
            std::vector<std::shared_ptr<IStreamChannel>> streamChannelGroup;

#if defined(HAVE_OB_SENSOR_V4L2)
            auto &ctx = V4L2Context::getInstance();
#elif defined(HAVE_OB_SENSOR_MSMF)
            auto &ctx = MFContext::getInstance();
#endif // HAVE_OB_SENSOR_V4L2

            auto uvcDevInfoList = ctx.queryUvcDeviceInfoList();

            std::map<std::string, std::vector<UvcDeviceInfo>> uvcDevInfoGroupMap;

            auto devInfoIter = uvcDevInfoList.begin();
            while (devInfoIter != uvcDevInfoList.begin())
            {

                if (devInfoIter->vid != OB3D_CAM_PID)
                {
                    devInfoIter = uvcDevInfoList.erase(devInfoIter); // drop it
                    continue;
                }
                devInfoIter++;
            }

            if (!uvcDevInfoList.empty() && uvcDevInfoList.size() <= 3)
            {
                uvcDevInfoGroupMap.insert({"default", uvcDevInfoList});
            }
            else {
                for (auto &devInfo : uvcDevInfoList)
                {
                    // group by uid
                    uvcDevInfoGroupMap[devInfo.uid].push_back(devInfo); // todo: group by sn
                }
            }

            if (uvcDevInfoGroupMap.size() > groupIdx)
            {
                auto uvcDevInfoGroupIter = uvcDevInfoGroupMap.begin();
                std::advance(uvcDevInfoGroupIter, groupIdx);
                for (const auto &devInfo : uvcDevInfoGroupIter->second)
                {
                    streamChannelGroup.emplace_back(ctx.createStreamChannel(devInfo));
                }
            }
            else
            {
                CV_LOG_ERROR(NULL, "Camera index out of range");
            }
            return streamChannelGroup;
        }
    } // namespace obsensor
} // namespace cv
#endif // HAVE_OB_SENSOR_V4L2 || HAVE_OB_SENSOR_MSMF