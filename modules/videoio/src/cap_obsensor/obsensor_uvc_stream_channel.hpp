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
#define XU_UNIT_ID 4

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

        FrameFormat frameFourccToFormat(uint32_t fourcc);
        uint32_t frameFormatToFourcc(FrameFormat);
        
        struct OBExtensionParam{
            float bl; 
            float bl2;
            float pd; 
            float ps; 
        };

        class DepthFrameProcessor{
        public:
            DepthFrameProcessor(const OBExtensionParam &parma);
            ~DepthFrameProcessor() noexcept;
            void process(Frame *frame);

        private:
            const OBExtensionParam param_;
            uint16_t *lookUpTable_;
        };

        class IUvcStreamChannel: public IStreamChannel{
            public:
                IUvcStreamChannel(const UvcDeviceInfo &devInfo);
                virtual ~IUvcStreamChannel() noexcept {}

                virtual bool setProperty(int propId, const uint8_t *data, uint32_t dataSize) override;
                virtual bool getProperty(int propId, uint8_t *recvData, uint32_t *recvDataSize) override;
                virtual StreamType streamType() const override;

            protected:
                virtual bool setXu(uint8_t ctrl, const uint8_t *data, uint32_t len) = 0;
                virtual bool getXu(uint8_t ctrl, uint8_t **data, uint32_t *len) = 0;

                bool initDepthFrameProcessor();

            protected:
                const UvcDeviceInfo devInfo_;
                StreamType streamType_;
                std::shared_ptr<DepthFrameProcessor> depthFrameProcessor_;
        };
    } // namespace obsensor
} // namespace cv
#endif // HAVE_OBSENSOR
#endif // _CAP_OBSENSOR_UVC_STREAM_CHANNEL_HPP_