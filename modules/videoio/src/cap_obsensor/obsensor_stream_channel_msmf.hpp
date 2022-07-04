// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef _CAP_OBSENSOR_STREAM_CHANNEL_MSMF_HPP_
#define _CAP_OBSENSOR_STREAM_CHANNEL_MSMF_HPP_
#ifdef HAVE_OBSENSOR_MSMF

#include "obsensor_uvc_stream_channel.hpp"

#include <windows.h>
#include <guiddef.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfplay.h>
#include <mfobjects.h>
#include <mfreadwrite.h>
#include <tchar.h>
#include <strsafe.h>
#include <codecvt>
#include <ks.h>
#include <comdef.h>
#include <mutex>
#include <vidcap.h> //IKsTopologyInfo
#include <ksproxy.h> //IKsControl
#include <ksmedia.h> 

namespace cv{
namespace obsensor{
    template <class T>
    class ComPtr
    {
    public:
        ComPtr()
        {
        }
        ComPtr(T *lp)
        {
            p = lp;
        }
        ComPtr(_In_ const ComPtr<T> &lp)
        {
            p = lp.p;
        }
        virtual ~ComPtr()
        {
        }

        void swap(_In_ ComPtr<T> &lp)
        {
            ComPtr<T> tmp(p);
            p = lp.p;
            lp.p = tmp.p;
            tmp = NULL;
        }
        T **operator&()
        {
            CV_Assert(p == NULL);
            return p.operator&();
        }
        T *operator->() const
        {
            CV_Assert(p != NULL);
            return p.operator->();
        }
        operator bool()
        {
            return p.operator!=(NULL);
        }

        T *Get() const
        {
            return p;
        }

        void Release()
        {
            if (p)
                p.Release();
        }

        // query for U interface
        template <typename U>
        HRESULT As(_Out_ ComPtr<U> &lp) const
        {
            lp.Release();
            return p->QueryInterface(__uuidof(U), reinterpret_cast<void **>((T **)&lp));
        }

    private:
        _COM_SMARTPTR_TYPEDEF(T, __uuidof(T));
        TPtr p;
    };

    class MFContext
    {
    public:
        ~MFContext(void);
        static MFContext &getInstance();

        std::vector<UvcDeviceInfo> queryUvcDeviceInfoList();
        std::shared_ptr<IStreamChannel> createStreamChannel(const UvcDeviceInfo &devInfo);

    private:
        MFContext(void);
    };

    typedef struct FrameRate
    {
        unsigned int denominator;
        unsigned int numerator;
    } FrameRate;

    class MSMFStreamChannel : public IStreamChannel, public IMFSourceReaderCallback
    {
    public:
        MSMFStreamChannel(const UvcDeviceInfo &devInfo);
        virtual ~MSMFStreamChannel() noexcept;

        virtual void start(const StreamProfile &profile, FrameCallback frameCallback) override;
        virtual void stop() override;
        virtual bool setProperty(int propId, const uint8_t *data, uint32_t dataSize) override;
        virtual bool getProperty(int propId, uint8_t *recvData, uint32_t recvDataSize) override;

        virtual StreamType streamType() const override;

    private:
        bool setXu(uint8_t ctrl, const uint8_t *data, uint32_t len);
        bool getXu(uint8_t ctrl, uint8_t **data, uint32_t *len);

    private:
        MFContext &mfContext_;
        const UvcDeviceInfo devInfo_;
        StreamType streamType_;

        ComPtr<IMFAttributes> deviceAttrs_ = nullptr;
        ComPtr<IMFMediaSource> deviceSource_ = nullptr;
        ComPtr<IMFAttributes> readerAttrs_ = nullptr;
        ComPtr<IMFSourceReader> streamReader_ = nullptr;
        ComPtr<IAMCameraControl> cameraControl_ = nullptr;
        ComPtr<IAMVideoProcAmp> videoProcAmp_ = nullptr;
        ComPtr<IKsTopologyInfo> xuKsTopologyInfo_ = nullptr;
        ComPtr<IUnknown> xuNodeInstance_ = nullptr;
        ComPtr<IKsControl> xuKsControl_ = nullptr;
        int xuNodeId_;

        FrameCallback frameCallback_;
        StreamProfile currentProfile_;
        int8_t currentStreamIndex_;

        StreamState streamState_;
        std::mutex streamStateMutex_;
        std::condition_variable streamStateCv_;

        uint8_t *xuRecvBuf_;
        uint8_t *xuSendBuf_;

        std::shared_ptr<DepthFrameProcessor> depthFrameProcessor_;

    public:
        STDMETHODIMP QueryInterface(REFIID iid, void **ppv) override;
        STDMETHODIMP_(ULONG)
        AddRef() override;
        STDMETHODIMP_(ULONG)
        Release() override;
        STDMETHODIMP OnReadSample(HRESULT /*hrStatus*/, DWORD dwStreamIndex, DWORD /*dwStreamFlags*/, LONGLONG /*llTimestamp*/, IMFSample *sample) override;
        STDMETHODIMP OnEvent(DWORD /*sidx*/, IMFMediaEvent * /*event*/) override;
        STDMETHODIMP OnFlush(DWORD) override;

    private:
        long refCount_ = 1;
    };
} // namespace obsensor
} // namespace cv::obsensor
#endif // HAVE_OBSENSOR_MSMF
#endif // _CAP_OBSENSOR_STREAM_CHANNEL_MSMF_HPP_