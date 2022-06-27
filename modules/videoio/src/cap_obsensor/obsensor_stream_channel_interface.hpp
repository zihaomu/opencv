// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef _CAP_OB_SENSOR_STREAM_CHANNEL_INTERFACE_HPP_
#define _CAP_OB_SENSOR_STREAM_CHANNEL_INTERFACE_HPP_

#ifdef HAVE_OB_SENSOR

#include "../precomp.hpp"

#include <functional>
#include <vector>
#include <memory>

namespace cv{
namespace obsensor{
    typedef enum
    {
        OB3D_STREAM_IR = 1,
        OB3D_STREAM_RGB = 2,
        OB3D_STREAM_DEPTH = 3,
    } StreamType;

    typedef enum
    {
        FRAME_FORMAT_UNKNOWN = -1,
        FRAME_FORMAT_YUYV = 0,
        FRAME_FORMAT_MJPG = 5,
        FRAME_FORMAT_Y16 = 8,
    } FrameFormat;

    struct Frame
    {
        FrameFormat format;
        uint32_t width;
        uint32_t height;
        uint32_t dataSize;
        uint8_t *data;
    };

    struct StreamProfile
    {
        uint32_t width;
        uint32_t height;
        uint32_t fps;
        FrameFormat format;
    };

    typedef std::function<void(Frame *)> FrameCallback;

    class IStreamChannel
    {
    public:
        virtual ~IStreamChannel() noexcept {}
        virtual void start(const StreamProfile &profile, FrameCallback frameCallback) = 0;
        virtual void stop() = 0;
        virtual bool setProperty(int obPropId, const uint8_t *data, uint32_t dataSize) = 0;
        virtual bool getProperty(int obPropId, uint8_t *outData, uint32_t outDataSize) = 0;

        virtual StreamType streamType() const = 0;
    };
    
    // "StreamChannelGroup" mean a group of stream channels from same one physical device
    std::vector<std::shared_ptr<IStreamChannel>> getStreamChannelGroup(uint32_t groupIdx = 0);

} // namespace obsensor
} // namespace cv
#endif // HAVE_OB_SENSOR
#endif // _CAP_OB_SENSOR_STREAM_CHANNEL_INTERFACE_HPP_