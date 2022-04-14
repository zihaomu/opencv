// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2022, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CONVOLUTION_SGEMM_PACK4_HPP
#define OPENCV_CONVOLUTION_SGEMM_PACK4_HPP

#include "../../precomp.hpp"
namespace cv { namespace dnn {
//
////static void im2col_sgemm(Mat& _input, Mat& _output, Mat& _colKernel, const std::vector<float>& bias)
//static void im2col_sgemm_pack4(Mat& input, Mat& output, Mat& colKernel, const std::vector<float>& bias, const std::vector<float>& reluslope)
//{
//    // Check
//    MatShape inputShape = shape(input);
//    MatShape kShape = shape(colKernel);
//    MatShape outputShape = shape(output);
////    CV_Assert(inputShape.size() == 2 && outputShape.size() == 3);
//    int outCh = kShape[0];
//    int outW = outputShape[1];
//    int outH = outputShape[2];
//
//    size_t outSize = outW * outH;
//
//    int inW = input.cols;
//    int inH = input.rows;
//
//    // one imp
////    CV_Assert(inputShape[1] == kShape[1]);
//    int kW = kShape[1];
//    if (kShape.size() == 4)
//    {
//        kW *= kShape[2];
//        kW *= kShape[3];
//    }
//
////    float* outPtr = output.ptr<float>();
////    float* kPtr = colKernel.ptr<float>();
////    std::cout<<"kernel = ";
////    printblob(colKernel);
//
//#ifdef _OPENMP
//#pragma omp parallel for num_threads(std::max(getNumThreads(), 1))
//#endif
//    for (int o_i = 0; o_i < outCh; o_i++)
//    {
////        outPtr += o_i * inH;
//        float* kPtr = colKernel.ptr<float>() + o_i * kW;
//        float* inPtr = input.ptr<float>();
////        float *outPtr = output.row(o_i).ptr<float>();
//        float *outPtr = output.ptr<float>() + o_i * inputShape[0];
//        const float bias0 = bias.empty() ? 0.f : bias[o_i];
//        const float relu = reluslope.empty() ? 1.f : reluslope[o_i];
//
//        for (int i = 0; i < inH; i++)
//        {
//            float* kInPtr = inPtr + i * inW;
//            float* kKPtr = kPtr;
//            float sum0 = bias0;
//            int j = 0;
//#ifdef CV_SIMD
//            v_float32x4 v_kernel, v_input;
//            v_float32x4 v_sum = v_setzero_f32();
//            for (;j < inW - 4; j += 4)
//            {
//                v_kernel = v_load(kKPtr);
//                v_input = v_load(kInPtr);
//
//                v_sum += v_kernel * v_input;
//                kInPtr += 4;
//                kKPtr += 4;
//            }
//            sum0 += v_reduce_sum(v_sum);
//#endif
//            for (; j < inW; j++)
//            {
//                sum0 += kInPtr[0] * kKPtr[0];
//                kInPtr++;
//                kKPtr++;
//            }
//            if (!reluslope.empty())
//                sum0 = sum0 > 0.f ? sum0 : sum0*relu;
//            outPtr[0] = sum0;
//            outPtr++;
//        }
//    }
//}
//
//// Run in the layer initial stage.
//static void im2col_sgemm_transform_kernel_pack4(InputArray _kernel, OutputArray _colkernel)
//{
//    Mat kernel = _kernel.getMat();
//    MatShape kShape = shape(kernel);
////    CV_Assert(kShape.size() == 4);
////    std::vector<int> newShape = {kShape[0], kShape[1] * kShape[2] * kShape[3]};
////
////    Mat colKernel = _colkernel.getMat();
//    kernel.copyTo(_colkernel);
////    colKernel = kernel.clone();
////    colKernel = colKernel.reshape(0, newShape);
//}
//
//// the Input data layout is NC4HW4.
//// the Output data is also NC4HW4.
//static void convolution_im2col_sgemm_pack4(InputArray _input, OutputArray _output, InputArray _colKernel, const std::vector<float>& bias, const std::vector<float>& reluslope, int kernelW, int kernelH, int stride_w, int stride_h, int dilation_w, int dilation_h)
//{
//    Mat input = _input.getMat();
//    Mat colKernel = _colKernel.getMat();
//    Mat output = _output.getMat();
//
//    CV_Assert(!input.empty() && !output.empty() && !_colKernel.empty());
//
//    MatShape inputShape = shape(input);
//    MatShape outputShape = shape(output);
////    MatShape kernelShape = shape(colKernel);
//
//    CV_Assert(inputShape.size() == 5 && outputShape.size() == 4);
//
//    int batchSize = inputShape[0] ;
//    int inputChannel = inputShape[1];
//    int inputH = inputShape[2];
//    int inputW = inputShape[3];
//
//    int outputChannel = outputShape[1];
//    int outputH = outputShape[2];
//    int outputW = outputShape[3];
//
////    int kernelW = kernelShape[2];
////    int kernelH = kernelShape[3];
//
//    // im2col
//    // Note that the im2col used here is different. In oder to reduce the transpose
//    std::vector<int> colInputShape = {outputH*outputW, kernelH*kernelW*inputChannel};
//
//    // TODO! allocalte the memory with universal memory space.
////    Mat colInput(outputH*outputW, kernelH*kernelW*inputChannel, input.type());
//    const int gap = inputW * stride_h - outputW * stride_w;
//
//    size_t chSize = inputW * inputH;
//    size_t bnSize = chSize * inputChannel;
//    size_t kSize = kernelH * kernelW;
//
//    float * imgPtr = input.ptr<float>();
//    Mat colInput = Mat::zeros(outputH*outputW, kernelH*kernelW*inputChannel, input.type());
////    printblob(input);
//    for (int bn_i = 0; bn_i < batchSize; bn_i++)
//    {
//        Mat output_i = output.row(bn_i);
//
//        float * bnImgPtr = imgPtr + bnSize * bn_i;
//        float* ptr = colInput.ptr<float>();
//
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
//        for (int ch_i = 0; ch_i < inputChannel; ch_i++)
//        {
//            // imple one
//            // inputChannel, oH, oW, kH, kW
//            float* chImgPtr = bnImgPtr + ch_i * chSize;
//            float* chPtr = ptr + kSize * ch_i;
//
//            for (int oh_i = 0; oh_i < outputH; oh_i++)
//            {
//                for (int ow_i = 0; ow_i < outputW; ow_i ++)
//                {
//                    float* kPtrImg = chImgPtr + oh_i * stride_h * inputW + ow_i * stride_w;
//                    float* kPtr = chPtr + (oh_i * outputW + ow_i) * inputChannel * kSize;
//                    for (int kh_i = 0; kh_i < kernelH; kh_i++)
//                    {
//                        float * kPtrImgIn = kPtrImg + kh_i * inputW * dilation_h;  // change line
//                        for (int kw_i = 0; kw_i < kernelW; kw_i++)
//                        {
//                            kPtr[0] = kPtrImgIn[0];
//                            kPtr++;
//                            kPtrImgIn += dilation_w;
//                        }
//                    }
//                }
//            }
//        }
//
//
//        // second imp, Ch, kh, kw, oh, hw, with permuate layer in the im2col sgemm.
////        for (int ch_i = 0; ch_i < inputChannel; ch_i++)
////        {
////            float* chImgPtr = bnImgPtr + ch_i * chSize;
////            float* chPtr = ptr + kSize * ch_i;
////
////            for (int kh_i = 0; kh_i < kernelH; kh_i++)
////            {
////                for (int hw_i = 0; hw_i < kernelW; hw_i++)
////                {
////                    const float* sptr = img.row<const float>(dilation_h * u) + dilation_w * v;
////
////                    for (int oh_i = 0; oh_i < outputH; oh_i++)
////                    {
////                        int ow_i = 0;
////                        for (ow_i = 0; ow_i + 3 < outputW; ow_i += 4)
////                        {
////                            ptr[0] = sptr[0];
////                            ptr[1] = sptr[stride_w];
////                            ptr[2] = sptr[stride_w * 2];
////                            ptr[3] = sptr[stride_w * 3];
////
////                            sptr += stride_w * 4;
////                            ptr += 4;
////                        }
////                        for (; j + 1 < outw; j += 2)
////                        {
////                            ptr[0] = sptr[0];
////                            ptr[1] = sptr[stride_w];
////
////                            sptr += stride_w * 2;
////                            ptr += 2;
////                        }
////                        for (; j < outw; j++)
////                        {
////                            ptr[0] = sptr[0];
////
////                            sptr += stride_w;
////                            ptr += 1;
////                        }
////
////                        sptr += gap;
////                    }
////                }
////            }
////        }
//
//        im2col_sgemm(colInput, output_i, colKernel, bias, reluslope);
//    }
//}

}} // namespace cv::dnn

#endif //OPENCV_CONVOLUTION_SGEMM_PACK4_HPP
