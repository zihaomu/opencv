// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#if defined(HAVE_OBSENSOR_V4L2) ||  defined(HAVE_OBSENSOR_MSMF)

#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>

#if defined(HAVE_OBSENSOR_V4L2)
#include "obsensor_stream_channel_v4l2.hpp"
#elif defined(HAVE_OBSENSOR_MSMF)
#include "obsensor_stream_channel_msmf.hpp"
#endif // HAVE_OBSENSOR_V4L2

namespace cv
{
    namespace obsensor
    {
    const uint8_t DEPTH_TO_COLOR_ALIGN_CMD0[16] = {0x47, 0x4d, 0x04, 0x00, 0x02, 0x00, 0x52, 0x00, 0x5B, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00};
    const uint8_t DEPTH_TO_COLOR_ALIGN_CMD1[16] = {0x47, 0x4d, 0x04, 0x00, 0x02, 0x00, 0x54, 0x00, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    const uint8_t DEPTH_TO_COLOR_ALIGN_CMD2[16] = {0x47, 0x4d, 0x04, 0x00, 0x02, 0x00, 0x56, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00};
    const uint8_t DEPTH_TO_COLOR_ALIGN_CMD3[16] = {0x47, 0x4d, 0x04, 0x00, 0x02, 0x00, 0x58, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00};

#if defined(HAVE_OBSENSOR_V4L2)
#define fourCc2Int(a, b, c, d) \
    ((uint32_t)(a) | ((uint32_t)(b) << 8) | ((uint32_t)(c) << 16) | ((uint32_t)(d) << 24))
#elif defined(HAVE_OBSENSOR_MSMF)
#define fourCc2Int(a, b, c, d) \
    (((uint32_t)(a) <<24) | ((uint32_t)(b) << 16) | ((uint32_t)(c) << 8) | (uint32_t)(d))
#endif // HAVE_OBSENSOR_V4L2

        const std::map<uint32_t, FrameFormat> fourccToOBFormat = {
            {fourCc2Int('Y', 'U', 'Y', '2'), FRAME_FORMAT_YUYV},
            {fourCc2Int('M', 'J', 'P', 'G'), FRAME_FORMAT_MJPG},
            {fourCc2Int('Y', '1', '6', ' '), FRAME_FORMAT_Y16},
        };

        StreamType parseUvcDeviceNameToStreamType(const std::string &devName)
        {
            std::string uvcDevName = devName;
            std::transform(begin(uvcDevName), end(uvcDevName), begin(uvcDevName), ::tolower);
            if (uvcDevName.find(" depth") != std::string::npos)
            {
                return OBSENSOR_STREAM_DEPTH;
            }
            else if (uvcDevName.find(" ir") != std::string::npos)
            {
                return OBSENSOR_STREAM_IR;
            }

            return OBSENSOR_STREAM_RGB; // else
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

        uint32_t frameFormatToFourcc(FrameFormat fmt)
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

#if defined(HAVE_OBSENSOR_V4L2)
            auto &ctx = V4L2Context::getInstance();
#elif defined(HAVE_OBSENSOR_MSMF)
            auto &ctx = MFContext::getInstance();
#endif // HAVE_OBSENSOR_V4L2

            auto uvcDevInfoList = ctx.queryUvcDeviceInfoList();

            std::map<std::string, std::vector<UvcDeviceInfo>> uvcDevInfoGroupMap;

            auto devInfoIter = uvcDevInfoList.begin();
            while (devInfoIter != uvcDevInfoList.begin())
            {

                if (devInfoIter->vid != OBSENSOR_CAM_PID)
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
#endif // HAVE_OBSENSOR_V4L2 || HAVE_OBSENSOR_MSMF