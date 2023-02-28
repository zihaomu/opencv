// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include "op_vkcom.hpp"
#include "net_impl.hpp"

namespace cv
{
namespace dnn
{
#ifdef HAVE_VULKAN

CV__DNN_INLINE_NS_BEGIN


void Net::Impl::initVkComBackend()
{
    CV_TRACE_FUNCTION();
    CV_Assert(preferableBackend == DNN_BACKEND_VKCOM);

    context = vkcom::Context::create();

    for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); it++)
    {
        LayerData &ld = it->second;
        Ptr<Layer> layer = ld.layerInstance;
        if (!layer->supportBackend(preferableBackend))
        {
            continue;
        }

        ld.skip = false;

        try
        {
            ld.backendNodes[DNN_BACKEND_VKCOM] = layer->initVkCom(ld.inputBlobsWrappers, ld.outputBlobsWrappers);
        }
        catch (const cv::Exception& e)
        {
            CV_LOG_ERROR(NULL, "initVkCom failed, fallback to CPU implementation. " << e.what());
            ld.backendNodes[DNN_BACKEND_VKCOM] = Ptr<BackendNode>();
        }
    }
}

CV__DNN_INLINE_NS_END


///////////////////////////////////////////////////////////////////////////////
int transFusedActivType(Ptr<ActivationLayer> &actLayer)
{
    if (actLayer)
    {
        Ptr<ReLULayer> activ_relu = actLayer.dynamicCast<ReLULayer>();
        Ptr<ReLU6Layer> activ_relu6 = actLayer.dynamicCast<ReLU6Layer>();

        if (!activ_relu.empty())
        {
            if (activ_relu->negativeSlope == 0.0f)
            {
                return 1; // kFusedActivRelu
            }
            else // Leaky ReLU
            {
                return -1; // kFusedActivNone
            }
        }
        else if (!activ_relu6.empty())
        {
            return 2; // kFusedActivRelu6
        }
        else
            return -1; // kFusedActivUnsupport
    }
    else
        return 0; // kFusedActivNone
}

void copyToTensor(vkcom::Tensor &dst, const Mat &src)
{
    CV_Assert(src.isContinuous() && src.type() == CV_32F);

    std::vector<int> mat_shape = shape(src);
    dst.reshape((const char*)src.data, mat_shape); // This code will copy the src data from Mat to VkBuffer.

//    std::cout<<"print copyToTensor "<<std::endl;
//    printTensor(dst);
}

void copyToMat(Mat &dst, vkcom::Tensor &src)
{
    CV_Assert(dst.type() == CV_32F);

    if (dst.empty())
    {
//        dst.create()
    }
//    std::cout<<"before copy "<<std::endl;
//    printblob(dst);
    std::vector<int> shape = src.getShape();
    void *data = src.map();
    Mat tmp(shape, CV_32F, data);
    tmp.copyTo(dst);
    src.unMap();
//    std::cout<<"after copy "<<std::endl;
//    printblob(dst);
}

void printTensor(vkcom::Tensor &dst)
{
    std::vector<int> shap = dst.getShape();
    Mat tmp(shap, CV_32F);
    copyToMat(tmp, dst);

    std::cout<<"print data from tensor is "<<std::endl;
    printblob(tmp);
}

vkcom::Tensor VkComTensor(const Ptr<BackendWrapper>& ptr)
{
    CV_Assert(!ptr.empty());
    return ptr.dynamicCast<VkComBackendWrapper>()->getTensor();
}

void setDirty(std::vector<Ptr<BackendWrapper> >& ptrs)
{
    for (const Ptr<BackendWrapper>& ptr : ptrs)
    {
        ptr.dynamicCast<VkComBackendWrapper>()->setDeviceDirty();
    }
}

std::vector<vkcom::Tensor> VkComTensors(const std::vector<Ptr<BackendWrapper> >& ptrs)
{
    std::vector<vkcom::Tensor> vec;
    vec.reserve(ptrs.size());
    for (const Ptr<BackendWrapper>& ptr : ptrs)
    {
        vec.push_back(VkComTensor(ptr));
    }
    return vec;
}

VkComBackendNode::VkComBackendNode(const std::vector<Ptr<BackendWrapper> >& inputsWrapper,
                                   const Ptr<vkcom::OpBase>& op,
                                   const std::vector<Ptr<BackendWrapper> >& outputsWrapper, bool _fuseAdd)
                                   : BackendNode(DNN_BACKEND_VKCOM), fuseAdd(_fuseAdd)
{
    operation = op;

    inputsWrapper_ = inputsWrapper;
    ins = VkComTensors(inputsWrapper_);

    outputsWrapper_ = outputsWrapper;
    outs = VkComTensors(outputsWrapper_);
}

void VkComBackendNode::printTensorInfo()
{
    // TODEL
    Ptr<vkcom::OpConv> convOp = operation.dynamicCast<vkcom::OpConv>();

//    if (convOp)
//    {
//        std::cout<<"print convOp = "<<std::endl;
//        printTensor(*convOp->weightTensorPtr);
//        printTensor(*convOp->biasTensorPtr);
//    }

    Ptr<VkComBackendWrapper> outVk = outputsWrapper_[0].dynamicCast<VkComBackendWrapper>();
    std::cout<<"print output shape"<<std::endl;
    printShape(*outVk->getMat());
}

bool VkComBackendNode::forward()
{
    for (int i = 0, n = inputsWrapper_.size(); i < n; ++i)
    {
        inputsWrapper_[i].dynamicCast<VkComBackendWrapper>()->copyToDevice();
    }

    // NOTE: If fusedAdd is true, we need copy the output tensor to the Device, since it contains the added data.
    if (fuseAdd)
    {
        for (int i = 0, n = outputsWrapper_.size(); i < n; ++i)
        {
            outputsWrapper_[i].dynamicCast<VkComBackendWrapper>()->copyToDevice();
        }
    }

    return operation->forward(ins, outs);
}

VkComBackendWrapper::VkComBackendWrapper(Mat& m) : BackendWrapper(DNN_BACKEND_VKCOM, DNN_TARGET_VULKAN)
{
    // TODO! remove the copyToTensor here, we will do copy later.
    copyToTensor(tensor, m);
    host = &m;
    hostDirty = false;
    deviceDirty = false;
}

// Other constructor, need change the logical. The purpose is to decline the data copy.
VkComBackendWrapper::VkComBackendWrapper(const Ptr<BackendWrapper>& baseBuffer, Mat& m)
    : BackendWrapper(DNN_BACKEND_VKCOM, DNN_TARGET_VULKAN)
{
    Ptr<VkComBackendWrapper> base = baseBuffer.dynamicCast<VkComBackendWrapper>();
    CV_Assert(!base.empty());

    host = &m;
    tensor = base->tensor;
    CV_Assert(tensor.count() >= m.total());
    tensor.reshape(0, shape(m)); // Why this?
    hostDirty = false;
    deviceDirty = false;
}

void VkComBackendWrapper::copyToHost()
{
    Layer::t2.start();
    if (deviceDirty)
        copyToMat(*host, tensor);
    Layer::t2.stop();
}

void VkComBackendWrapper::setHostDirty()
{
    hostDirty = true;
};

void VkComBackendWrapper::setDeviceDirty()
{
    deviceDirty = true;
};

void VkComBackendWrapper::copyToDevice()
{
    Layer::t3.start();
    if (hostDirty)
    {
        copyToTensor(tensor, *host);
        hostDirty = false;
    }
    Layer::t3.stop();
}

vkcom::Tensor VkComBackendWrapper::getTensor()
{
    return tensor;
}

Mat* VkComBackendWrapper::getMat()
{
    return host;
}
#endif
void forwardVkCom(std::vector<Ptr<BackendWrapper> > &outputs,
                  const Ptr<BackendNode>& node)
{
#ifdef HAVE_VULKAN
    CV_Assert(!node.empty());

    Ptr<VkComBackendNode> node_ = node.dynamicCast<VkComBackendNode>();

//    node_->printTensorInfo();
    CV_Assert(node_->forward()); // run layer forward
    setDirty(outputs);
//    node_->printTensorInfo();
//    std::cout<<"setDirty(outputs); "<<std::endl;
#endif
}

bool haveVulkan()
{
#ifdef HAVE_VULKAN
    return vkcom::isAvailable();
#else
    return false;
#endif  // HAVE_VULKAN
}

}  // namespace dnn
}  // namespace cv
