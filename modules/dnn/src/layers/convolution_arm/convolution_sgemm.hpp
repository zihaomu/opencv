// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2022, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

// TODEL, change SGEMM
#ifndef OPENCV_CONVOLUTION_SGEMM_HPP
#define OPENCV_CONVOLUTION_SGEMM_HPP

#include "../../precomp.hpp"
namespace cv { namespace dnn {

//static void im2col_sgemm(Mat& input, Mat& output, Mat& colKernel, const std::vector<float>& bias, const std::vector<float>& reluslope)
static void im2col_sgemm(float* inputPtr0, size_t colW, size_t colH, Mat& output, Mat& colKernel, const std::vector<float>& bias, const std::vector<float>& reluslope)
{
    static TickMeter tickMeterTrans;
    static TickMeter tickMeterMult;

    enum { BLK_SIZE = 32, BLK_SIZE_CN = 64 };
    static const int valign = 8;
    static const size_t rowbufsz = alignSize(3687936, valign)*BLK_SIZE;
    static AutoBuffer<float> tempBuffer(rowbufsz);  // only allocate memory once.
    static float * tmpPtr0 = alignPtr(tempBuffer.data(), (int)(valign*sizeof(float)));
//

    // Check
//    MatShape inputShape = shape(input);
    MatShape kShape = shape(colKernel);
    MatShape outputShape = shape(output);
//    CV_Assert(inputShape.size() == 2 && outputShape.size() == 3);
    int outCh = kShape[0];
    int outW = outputShape[1];
    int outH = outputShape[2];

    size_t outSize = outW * outH;

    int inW = colW;
    int inH = colH;

//    if (maxSize < inW *inH)
//        maxSize = inW * inH;
//
//    std::cout<<"max Size = "<<maxSize<<std::endl;

    // one imp
//    CV_Assert(inputShape[1] == kShape[1]);
    int kW = kShape[1];
    if (kShape.size() == 4)
    {
        kW *= kShape[2];
        kW *= kShape[3];
    }

    // imp 1
//#ifdef _OPENMP
//#pragma omp parallel for num_threads(std::max(getNumThreads(), 1))
//#endif
//    for (int o_i = 0; o_i < outCh; o_i++)
//    {
//        float* kPtr = colKernel.ptr<float>() + o_i * kW;
//        float* inPtr = inputPtr0;
//        float *outPtr = output.ptr<float>() + o_i * colH;
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

    // Imp 2, first transpose, than, gemm.
    tickMeterTrans.start();
//    Mat tmp;
//    tmp.create(inW, inH, input.type());
    float * tmpPtr = tmpPtr0;
    // permute
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i =0; i < inW; i ++ )
    {
        float * tmpptr = tmpPtr + inH * i;
        float * inptr = inputPtr0 + i;
        for (int j = 0; j < colH; j++)
        {
            tmpptr[0] = inptr[0];
            inptr += inW;
            tmpptr++;
        }
    }
//    tmp = input.t();

    tickMeterTrans.stop();
    std::cout<<"transpose time = "<< tickMeterTrans.getTimeMilli()<<std::endl;

    int temp = inW;
    inW = inH;
    inH = temp;


    tickMeterMult.start();
//    CV_Assert(tmp.isContinuous());
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int o_i = 0; o_i < outCh; o_i++)
    {
        float* kPtr = colKernel.ptr<float>() + o_i * kW;
        float* inPtr = tmpPtr0;
        float *outPtr = output.ptr<float>() + o_i * inH;
        const float bias0 = bias.empty() ? 0.f : bias[o_i];
        const float relu = reluslope.empty() ? 1.f : reluslope[o_i];

        for (int i = 0; i < inH; i++)
        {
            float* kInPtr = inPtr + i * inW;
            float* kKPtr = kPtr;
            float sum0 = bias0;
            int j = 0;
#ifdef CV_SIMD
            v_float32x4 v_kernel, v_input;
            v_float32x4 v_sum = v_setzero_f32();
            for (;j < inW - 4; j += 4)
            {
                v_kernel = v_load(kKPtr);
                v_input = v_load(kInPtr);

                v_sum += v_kernel * v_input;
                kInPtr += 4;
                kKPtr += 4;
            }
            sum0 += v_reduce_sum(v_sum);
#endif
            for (; j < inW; j++)
            {
                sum0 += kInPtr[0] * kKPtr[0];
                kInPtr++;
                kKPtr++;
            }
            sum0 = sum0 > 0.f ? sum0 : sum0*relu;
            outPtr[0] = sum0;
            outPtr++;
        }
    }


    tickMeterMult.stop();
    std::cout<<"Mult time = "<< tickMeterMult.getTimeMilli()<<std::endl;
}

// Run in the layer initial stage.
static void im2col_sgemm_transform_kernel(InputArray _kernel, OutputArray _colkernel)
{
    Mat kernel = _kernel.getMat();
    MatShape kShape = shape(kernel);
//    CV_Assert(kShape.size() == 4);
//    std::vector<int> newShape = {kShape[0], kShape[1] * kShape[2] * kShape[3]};
//
//    Mat colKernel = _colkernel.getMat();
    kernel.copyTo(_colkernel);
//    colKernel = kernel.clone();
//    colKernel = colKernel.reshape(0, newShape);
}

// the Input data layout is NCHW.
static void convolution_im2col_sgemm(InputArray _input, OutputArray _output, InputArray _colKernel, const std::vector<float>& bias, const std::vector<float>& reluslope, int kernelW, int kernelH, int stride_w, int stride_h, int dilation_w, int dilation_h)
{
    Mat input = _input.getMat();
    Mat colKernel = _colKernel.getMat();
    Mat output = _output.getMat();
    TickMeter tickMeterIm2Col;

    CV_Assert(!input.empty() && !output.empty() && !_colKernel.empty());

    MatShape inputShape = shape(input);
    MatShape outputShape = shape(output);
//    MatShape kernelShape = shape(colKernel);

    CV_Assert(inputShape.size() == 4 && outputShape.size() == 4);

    // NCHW
    int batchSize = inputShape[0];
    int inputChannel = inputShape[1];
    int inputH = inputShape[2];
    int inputW = inputShape[3];

    int outputChannel = outputShape[1];
    int outputH = outputShape[2];
    int outputW = outputShape[3];

//    int kernelW = kernelShape[2];
//    int kernelH = kernelShape[3];

    // im2col
    // Note that the im2col used here is different. In oder to reduce the transpose
    std::vector<int> colInputShape = {outputH*outputW, kernelH*kernelW*inputChannel};

    // TODO! allocalte the memory with universal memory space.
//    Mat colInput(outputH*outputW, kernelH*kernelW*inputChannel, input.type());
    const int gap = inputW * stride_h - outputW * stride_w;

    size_t chSize = inputW * inputH;
    size_t bnSize = chSize * inputChannel;
    size_t kSize = kernelH * kernelW;

    float * imgPtr = input.ptr<float>();

    enum { BLK_SIZE = 32, BLK_SIZE_CN = 64 };
    static const int valign = 8;
    static const size_t rowbufsz = alignSize(3687936, valign)*BLK_SIZE;
    static AutoBuffer<float> tempBuffer(rowbufsz);  // only allocate memory once.
    static float * tmpPtr0 = alignPtr(tempBuffer.data(), (int)(valign*sizeof(float)));

    size_t colH = kernelH*kernelW*inputChannel;
    size_t colW = outputH*outputW;

    for (int bn_i = 0; bn_i < batchSize; bn_i++)
    {
        Mat output_i = output.row(bn_i);

        float * bnImgPtr = imgPtr + bnSize * bn_i;
        float* ptr = tmpPtr0;

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

        tickMeterIm2Col.start();
        // second imp, Ch, kh, kw, oh, ow, with permuate layer in the im2col sgemm.
        const int gap = inputW * stride_h - outputW * stride_w;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int ch_i = 0; ch_i < inputChannel; ch_i++)
        {
            // imple one
            // inputChannel, kh, kw, oh, ow
            float* chImgPtr = bnImgPtr + ch_i * chSize;
            float* kPtr = ptr + outputW * outputH * kSize * ch_i;

            for (int kh_i = 0; kh_i < kernelH; kh_i++)
            {
                for (int kw_i = 0; kw_i < kernelW; kw_i++)
                {
                    float* kPtrImg = chImgPtr + kh_i * dilation_h * inputW + kw_i * dilation_w;
//                    float* kPtr = chPtr + (oh_i * outputW + ow_i) * inputChannel * kSize

                    for (int oh_i = 0; oh_i < outputH; oh_i++)
                    {
                        for (int ow_i = 0; ow_i < outputW; ow_i++)
                        {
                            kPtr[0] = kPtrImg[0];
                            kPtrImg += stride_w;
                            kPtr++;
                        }
                        kPtrImg += gap;
                    }
                }
            }
        }

        tickMeterIm2Col.stop();
        std::cout<<"Im2Col time = "<< tickMeterIm2Col.getTimeMilli()<<std::endl;
//        colInput = colInput.t();
        im2col_sgemm(tmpPtr0, colW, colH, output_i, colKernel, bias, reluslope);
    }


}

}} // namespace cv::dnn

#endif //OPENCV_CONVOLUTION_SGEMM_HPP
