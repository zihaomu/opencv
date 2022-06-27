// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#if defined(HAVE_OB3D_V4L2) ||  defined(HAVE_OB3D_MSMF)

#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include<iterator>

#if defined(HAVE_OB3D_V4L2)
#include "ob3d_stream_channel_v4l2.hpp"
#elif defined(HAVE_OB3D_MSMF)
#include "ob3d_stream_channel_msmf.hpp"
#endif // HAVE_OB3D_V4L2

namespace cv
{
    namespace ob3d
    {
        template <typename T>
        uint32_t fourCc2Int(const T a, const T b, const T c, const T d)
        {
            static_assert((std::is_integral<T>::value), "fourcc supports integral built-in types only");
            return ((static_cast<uint32_t>(a) << 24) | (static_cast<uint32_t>(b) << 16) | (static_cast<uint32_t>(c) << 8) | (static_cast<uint32_t>(d) << 0));
        }

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

        std::vector<std::shared_ptr<IStreamChannel>> getStreamChannelGroup(uint32_t groupIdx)
        {
            std::vector<std::shared_ptr<IStreamChannel>> streamChannelGroup;
#if defined(HAVE_OB3D_V4L2)
            auto uvcDevList = std::vector<UvcDeviceInfo>();
            // auto uvcDevList = MFContext::getInstance().queryUvcDeviceList();
#elif defined(HAVE_OB3D_MSMF)
            auto uvcDevList = MFContext::getInstance().queryUvcDeviceList();
#endif // HAVE_OB3D_V4L2

            std::map<std::string, std::vector<UvcDeviceInfo>> uvcDevGroupMap;
            for (auto &dev : uvcDevList)
            {
                if (dev.vid != OB3D_CAM_PID)
                {
                    continue;
                }

                // group by uid
                uvcDevGroupMap[dev.uid].push_back(dev); // todo: group by sn
            }

            if (uvcDevGroupMap.size() > groupIdx)
            {
                auto uvcDevGroupIter = uvcDevGroupMap.begin();
                std::advance(uvcDevGroupIter, groupIdx);
                for (const auto &dev : uvcDevGroupIter->second)
                {
#if defined(HAVE_OB3D_V4L2)
                    //  streamChannelGroup.emplace_back(makestd::shared_ptr<MSMFStreamChannel>(dev));
#elif defined(HAVE_OB3D_MSMF)
                    streamChannelGroup.emplace_back(makestd::shared_ptr<MSMFStreamChannel>(dev));
#endif // HAVE_OB3D_V4L2
                }
            }
            else
            {
                // CV_LOG_WARNING(NULL, "Camera index out of range");
            }
            return streamChannelGroup;
        }
    } // namespace ob3d
} // namespace cv
#endif // HAVE_OB3D_V4L2 || HAVE_OB3D_MSMF