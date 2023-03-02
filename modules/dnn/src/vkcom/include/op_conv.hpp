// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_VKCOM_OP_CONV_HPP
#define OPENCV_DNN_VKCOM_OP_CONV_HPP

#include "vkcom.hpp"
#include "op_base.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

enum ConvShaderType
{
    kConvShaderTypeGeneric = 0,
    kConvShaderType44 = 1, //TODO! support the 4x4 block
    kConvShaderTypeDepthWise = 2,
    kConvShaderTypeWinograd = 3,
    kConvShaderTest = 4,
};

struct ConvShaderConfig
{
    int local_size_x;
    int local_size_y;
    int local_size_z;
    int block_height;
    int block_width;
    int block_depth;
};
class SubOpConvRepackInput;

// Current Vulkan Convolution layer only support Conv2D.
class OpConv : public OpBase
{
public:
    OpConv(std::vector<Mat>& matBlobs, int activType, const int ngroups, const int K, const int C, const int Hk, const int Wk,
           const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
           const int pad_left, const int pad_top, bool fusedAdd);

//    bool forward(Tensor& in, Tensor& filter_weights, Tensor& bias, Tensor& out);
    virtual bool forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs) CV_OVERRIDE;
    Ptr<Tensor> weightTensorPtr;
    Ptr<Tensor> biasTensorPtr;
    Ptr<Tensor> testTensorPtr;
//    Ptr<SubOpConvRepackInput> operation;
private:
    bool computeGroupCount();

    FusedActivationType activ;
    const int ngroups;
    const int K, C, Hk, Wk; // output channel, input channel, height of kernel, width of kernel.
    const int stride_h, stride_w;
    const int dilation_h, dilation_w;
    const int pad_left, pad_top;
    const bool fusedAdd;

    int H0, W0;
    int Hi, Wi;
    int batch;
    int Kg, Cg;
    int CgHkWk, ksize;

    ConvShaderType shaderType;
    ConvShaderConfig config;
};

// This Op will repack the input data from [N, C, H, W] to [ngroups, ceil(H0*W0/VEL_LEN), Cg*Hk*Wk, VEC_LEN]
//class SubOpConvRepackInput : OpBase
//{
//public:
//    SubOpConvRepackInput(const int ngroups,const int batch, const int Hi, const int Wi, const int H0, const int W0,  const int C, const int Hk, const int Wk,
//                         const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
//                         const int pad_left, const int pad_top);
//    virtual bool forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs) CV_OVERRIDE;
//private:
//    bool computeGroupCount();
//    const int ngroups, batch;
//    const int Hi, Wi, H0, W0;
//    const int C, Hk, Wk;
//    const int stride_h, stride_w;
//    const int dilation_h, dilation_w;
//    const int pad_top, pad_left;
//    int Cg;
//    ConvShaderConfig config;
//};

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom

#endif // OPENCV_DNN_VKCOM_OP_CONV_HPP
