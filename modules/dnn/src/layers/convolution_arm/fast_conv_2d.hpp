//
// Created by Zihao Mu on 2022/4/21.
//

#ifndef OPENCV_FAST_CONV_2D_HPP
#define OPENCV_FAST_CONV_2D_HPP

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __ARM_NEON
enum { FX_CONV_MR=8, FX_CONV_NR=12 };
#elif defined __AVX__
enum { FX_CONV_MR=6, FX_CONV_NR=8 };
#else
enum { FX_CONV_MR=6, FX_CONV_NR=4 };
#endif

#ifdef __ARM_NEON
enum { FX_VEC_NLANES=4 };
#elif defined __AVX__
enum { FX_VEC_NLANES=8 };
#else
enum { FX_VEC_NLANES=1 };
#endif

#include "../../precomp.hpp"
namespace cv { namespace dnn {

typedef struct fast_conv2d_t
{
    int layout, ngroups;
    int Ock, Ick, Hk, Wk;
    int stride_y, stride_x;
    int dilation_y, dilation_x;

    float* weightsPtr;
    float* biasPtr;
    ActivationLayer* activ;
    bool ifMinMaxAct = false;
    float minval = -FLT_MAX, maxval = FLT_MAX;
} fast_conv2d_t;

struct fast_conv2d_t* init_fast_conv2d(
        int ngroups,
        int Ock, int Ick, int Hk, int Wk,
        int stride_y, int stride_x,
        int dilation_y, int dilation_x,
        float* weightsPtr,
        float* biasPtr,
        Ptr<ActivationLayer>& actLayer)
{
    fast_conv2d_t* conv = (fast_conv2d_t*)malloc(sizeof(*conv));  // create a new class named conv
    memset(conv, 0, sizeof(*conv));

    if (!(ngroups > 0 && Ock > 0 && Ick > 0 && Ock % ngroups == 0))
        CV_Assert(ngroups > 0 && Ock > 0 && Ick > 0 && Ock % ngroups == 0);
    CV_Assert(Hk > 0 && Wk > 0);
    CV_Assert(stride_y > 0 && stride_x > 0);
    CV_Assert(dilation_y > 0 && dilation_x > 0);

    conv->Ock = Ock; conv->Ick = Ick; conv->Hk = Hk; conv->Wk = Wk;  // [oC, iC, kH, kW]
    conv->stride_y = stride_y;
    conv->stride_x = stride_x;
    conv->dilation_y = dilation_y;
    conv->dilation_x = dilation_x;

    conv->ngroups = ngroups;
    conv->biasPtr = biasPtr;
    if (!actLayer.empty())
    {
        Ptr<ReLULayer> activ_relu = actLayer.dynamicCast<ReLULayer>();
        Ptr<ReLU6Layer> activ_relu6 = actLayer.dynamicCast<ReLU6Layer>();

        if( !activ_relu.empty() )
        {
            conv->minval = 0.0f;
            conv->ifMinMaxAct = true;
            conv->activ = 0;
        }
        else if( !activ_relu6.empty() )
        {
            conv->minval = 0.0f;
            conv->maxval = 6.0f;
            conv->ifMinMaxAct = true;
            conv->activ = 0;
        }
        else
            conv->activ = actLayer.get();
    }
    else
        conv->activ = 0;

    // store bias; append some zero's to make sure that
    // we can always read FX_CONV_MR elements starting from any valid index
    {
        int k = 0, nbias = Ock + FX_CONV_MR-1;
        conv->biasPtr = (float*)malloc(nbias*sizeof(conv->biasPtr[0]));
        for(; k < Ock; k++)
            conv->biasPtr[k] = biasPtr ? biasPtr[k] : 0.f;
        for(; k < nbias; k++)
            conv->biasPtr[k] = 0.f;
    }

    if (ngroups == Ock && ngroups == Ick) {
        // for depth-wise convolutions on NCHW data we just preserve the weights in OckIckHW layout,
        // but add some padding to make the weights array layout more SIMD-friendly
        int ksize = Hk*Wk;
        int padded_ksize = ((ksize + FX_VEC_NLANES-1)/FX_VEC_NLANES)*FX_VEC_NLANES;  // this code aims to let memory fit with vector size.
        int nweights = Ick*padded_ksize;
        conv->weightsPtr = (float*)malloc(nweights*sizeof(conv->weightsPtr[0]));
        memset(conv->weightsPtr, 0, nweights*sizeof(conv->weightsPtr[0]));
        for(int c = 0; c < Ick; c++) {
            for (int k = 0; k < ksize; k++)
                conv->weightsPtr[c*padded_ksize + k] = weightsPtr[c*ksize + k];
        }
    } else {
        // the weights are packed as
        // ngroups x (ceil((K/ngroups)/FX_CONV_MR)*FX_CONV_MR) x (Cg*Hk*Wk) x FX_CONV_MR tensor
        int Kg = Ock/ngroups, Cg = Ick/ngroups;
        int Kg_aligned = ((Kg + FX_CONV_MR - 1)/FX_CONV_MR)*FX_CONV_MR;
        size_t nweights = ngroups*Kg_aligned*Cg*Hk*Wk;
        conv->weightsPtr = (float*)malloc(nweights*sizeof(conv->weightsPtr[0]));
        memset(conv->weightsPtr, 0, nweights*sizeof(conv->weightsPtr[0]));
        float* packed_wptr = conv->weightsPtr;

        // pack the weight.
        for(int g = 0; g < ngroups; g++) {
            for(int k0 = 0; k0 < Kg_aligned; k0 += FX_CONV_MR) {
                int dk = Kg - k0 < FX_CONV_MR ? Kg - k0 : FX_CONV_MR;
                for(int c = 0; c < Cg; c++) {
                    for(int yx = 0; yx < Hk*Wk; yx++, packed_wptr += FX_CONV_MR) {
                        const float* wptr = weightsPtr + ((g*Kg + k0)*Cg + c)*Hk*Wk + yx;
                        int k = 0;
                        for(; k < dk; k++, wptr += Cg*Hk*Wk)  // wptr change the line every time.
                            packed_wptr[k] = *wptr;
                        for(; k < FX_CONV_MR; k++)
                            packed_wptr[k] = 0.f;
                    }
                }
            }
        }
    }
    return conv;
}


static void conv_block( int k, const float *a, const float *b,
                        float *c, int ldc, const float* bias,
                        float minval, float maxval, bool activ)
{
#ifdef __ARM_NEON
    float32x4_t c0 = vdupq_n_f32(bias[0]), c1 = c0, c2 = c0;
    float32x4_t c3 = vdupq_n_f32(bias[1]), c4 = c3, c5 = c3;
    float32x4_t c6 = vdupq_n_f32(bias[2]), c7 = c6, c8 = c6;
    float32x4_t c9 = vdupq_n_f32(bias[3]), c10 = c9, c11 = c9;
    float32x4_t c12 = vdupq_n_f32(bias[4]), c13 = c12, c14 = c12;
    float32x4_t c15 = vdupq_n_f32(bias[5]), c16 = c15, c17 = c15;
    float32x4_t c18 = vdupq_n_f32(bias[6]), c19 = c18, c20 = c18;
    float32x4_t c21 = vdupq_n_f32(bias[7]), c22 = c21, c23 = c21;

    for( int p = 0; p < k; p++, a += FX_CONV_MR, b += FX_CONV_NR )

    {
        float32x4_t a0 = vld1q_f32(a);
        float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4), b2 = vld1q_f32(b + 8);

        c0 = vfmaq_laneq_f32(c0, b0, a0, 0);
        c1 = vfmaq_laneq_f32(c1, b1, a0, 0);
        c2 = vfmaq_laneq_f32(c2, b2, a0, 0);
        c3 = vfmaq_laneq_f32(c3, b0, a0, 1);
        c4 = vfmaq_laneq_f32(c4, b1, a0, 1);
        c5 = vfmaq_laneq_f32(c5, b2, a0, 1);

        c6 = vfmaq_laneq_f32(c6, b0, a0, 2);
        c7 = vfmaq_laneq_f32(c7, b1, a0, 2);
        c8 = vfmaq_laneq_f32(c8, b2, a0, 2);
        c9 = vfmaq_laneq_f32(c9, b0, a0, 3);
        c10 = vfmaq_laneq_f32(c10, b1, a0, 3);
        c11 = vfmaq_laneq_f32(c11, b2, a0, 3);

        a0 = vld1q_f32(a + 4);

        c12 = vfmaq_laneq_f32(c12, b0, a0, 0);
        c13 = vfmaq_laneq_f32(c13, b1, a0, 0);
        c14 = vfmaq_laneq_f32(c14, b2, a0, 0);
        c15 = vfmaq_laneq_f32(c15, b0, a0, 1);
        c16 = vfmaq_laneq_f32(c16, b1, a0, 1);
        c17 = vfmaq_laneq_f32(c17, b2, a0, 1);

        c18 = vfmaq_laneq_f32(c18, b0, a0, 2);
        c19 = vfmaq_laneq_f32(c19, b1, a0, 2);
        c20 = vfmaq_laneq_f32(c20, b2, a0, 2);
        c21 = vfmaq_laneq_f32(c21, b0, a0, 3);
        c22 = vfmaq_laneq_f32(c22, b1, a0, 3);
        c23 = vfmaq_laneq_f32(c23, b2, a0, 3);
    }

    if (activ) {
        float32x4_t vmin = vdupq_n_f32(minval), vmax = vdupq_n_f32(maxval);
        c0 = vminq_f32(vmaxq_f32(c0, vmin), vmax);
        c1 = vminq_f32(vmaxq_f32(c1, vmin), vmax);
        c2 = vminq_f32(vmaxq_f32(c2, vmin), vmax);
        c3 = vminq_f32(vmaxq_f32(c3, vmin), vmax);
        c4 = vminq_f32(vmaxq_f32(c4, vmin), vmax);
        c5 = vminq_f32(vmaxq_f32(c5, vmin), vmax);
        c6 = vminq_f32(vmaxq_f32(c6, vmin), vmax);
        c7 = vminq_f32(vmaxq_f32(c7, vmin), vmax);
        c8 = vminq_f32(vmaxq_f32(c8, vmin), vmax);
        c9 = vminq_f32(vmaxq_f32(c9, vmin), vmax);
        c10 = vminq_f32(vmaxq_f32(c10, vmin), vmax);
        c11 = vminq_f32(vmaxq_f32(c11, vmin), vmax);
        c12 = vminq_f32(vmaxq_f32(c12, vmin), vmax);
        c13 = vminq_f32(vmaxq_f32(c13, vmin), vmax);
        c14 = vminq_f32(vmaxq_f32(c14, vmin), vmax);
        c15 = vminq_f32(vmaxq_f32(c15, vmin), vmax);
        c16 = vminq_f32(vmaxq_f32(c16, vmin), vmax);
        c17 = vminq_f32(vmaxq_f32(c17, vmin), vmax);
        c18 = vminq_f32(vmaxq_f32(c18, vmin), vmax);
        c19 = vminq_f32(vmaxq_f32(c19, vmin), vmax);
        c20 = vminq_f32(vmaxq_f32(c20, vmin), vmax);
        c21 = vminq_f32(vmaxq_f32(c21, vmin), vmax);
        c22 = vminq_f32(vmaxq_f32(c22, vmin), vmax);
        c23 = vminq_f32(vmaxq_f32(c23, vmin), vmax);
    }
    vst1q_f32(c, c0); vst1q_f32(c+4, c1); vst1q_f32(c+8, c2);
    vst1q_f32(c + ldc, c3); vst1q_f32(c + ldc + 4, c4); vst1q_f32(c + ldc + 8, c5);
    vst1q_f32(c + ldc*2, c6); vst1q_f32(c + ldc*2 + 4, c7); vst1q_f32(c + ldc*2 + 8, c8);
    vst1q_f32(c + ldc*3, c9); vst1q_f32(c + ldc*3 + 4, c10); vst1q_f32(c + ldc*3 + 8, c11);
    vst1q_f32(c + ldc*4, c12); vst1q_f32(c + ldc*4 + 4, c13); vst1q_f32(c + ldc*4 + 8, c14);
    vst1q_f32(c + ldc*5, c15); vst1q_f32(c + ldc*5 + 4, c16); vst1q_f32(c + ldc*5 + 8, c17);
    vst1q_f32(c + ldc*6, c18); vst1q_f32(c + ldc*6 + 4, c19); vst1q_f32(c + ldc*6 + 8, c20);
    vst1q_f32(c + ldc*7, c21); vst1q_f32(c + ldc*7 + 4, c22); vst1q_f32(c + ldc*7 + 8, c23);
#else
    for( int i = 0; i < FX_CONV_MR; i++ )
    {
        float beta = bias[i];
        for( int j = 0; j < FX_CONV_NR; j++ )
            c[i*ldc + j] = beta;
    }
    for( int p = 0; p < k; p++ )
    {
        int ch = p/9, yx = p - ch*9;
        int ky = yx/3, kx = yx % 3;

        for( int i = 0; i < FX_CONV_MR; i++ )
        {
            float alpha = a[FX_CONV_MR*p + i];
            for( int j = 0; j < FX_CONV_NR; j++ )
            {
                //if(call == 1 && kx == 1 && ky == 1 && i == 0 && j == 0)
                //    printf("c == %d: inpval = %.2f, w = %.2f\n", ch, b[FX_CONV_NR*p+j], alpha);
                c[i*ldc+j] += b[FX_CONV_NR*p + j]*alpha;
            }
        }
    }
    if (activ) {
        for( int i = 0; i < FX_CONV_MR; i++ )
        {
            for( int j = 0; j < FX_CONV_NR; j++ ) {
                float v = c[i*ldc + j];
                v = std::min(std::max(v, minval), maxval);
                c[i*ldc + j] = v;
            }
        }
    }
#endif
}

void fast_conv2d(InputArray _input, OutputArray _output,
                 const struct fast_conv2d_t* conv, int ntasks)
{
    Mat input = _input.getMat();
    Mat output = _output.getMat();

    MatShape inputShape = shape(input);
    MatShape outputShape = shape(output);

    CV_Assert(inputShape.size() == 4 && outputShape.size() == 4);

    if (conv->ngroups == conv->Ock && conv->ngroups == conv->Ick) {
        fast_depthwise_conv2d( input, output, conv, );
        return;
    }

    int N = inputShape[0], C = inputShape[1], Hi = inputShape[2], Wi = inputShape[3];  // [N, C, H, W]
    int K = conv->Ock, Hk = conv->Hk, Wk = conv->Wk;
    int H0 = outputShape[2], W0 = outputShape[3], ngroups = conv->ngroups;         // ngroups
    int Cg = C/ngroups, Kg = K/ngroups, Kg_aligned = ((Kg + FX_CONV_MR-1)/FX_CONV_MR)*FX_CONV_MR;  // align to MR

    //  MR works on channel dimension.
    int HkWkCg = Hk*Wk*Cg, HkWkC = HkWkCg*ngroups;
    const size_t inp_planesize = (size_t)Hi*Wi;
    const size_t out_planesize = (size_t)H0*W0;
    float minval = conv->minval, maxval = conv->maxval; // What's this for? maybe activation.

//    bool fast_activ = conv->activ == ACTIV_RELU ||
//                      conv->activ == ACTIV_RELU6 ||
//                      conv->activ == ACTIV_CLIP;
    int stripes_per_sample = (H0*W0+FX_CONV_NR-1)/FX_CONV_NR; // align to NR
    int stride_y = conv->stride_y, stride_x = conv->stride_x;
    int dilation_y = conv->dilation_y, dilation_x = conv->dilation_x;
//    int pad_top = conv->pad_top, pad_bottom = conv->pad_bottom;
//    int pad_left = conv->pad_left, pad_right = conv->pad_right;
    size_t taskbufsize = FX_CONV_NR*Hk*Wk*Cg;

    float* inpbuf_all = (float*)malloc(ntasks*taskbufsize*sizeof(inpbuf_all[0]) + Hk*Wk*3*sizeof(int));  // what's the final tail.
    int* ofstab = (int*)(inpbuf_all + ntasks*taskbufsize);
    int* xytab = ofstab + Hk*Wk;

    // remove duplicated computation of index.
    for (int y = 0; y < Hk; y++)
        for( int x = 0; x < Wk; x++)
        {
            int k = y*Wk + x;
            int dy = y*dilation_y, dx = x*dilation_x;
            xytab[k*2] = dy;
            xytab[k*2+1] = dx;
            ofstab[k] = dy*Wi + dx;
        }

    float* inp = input.ptr<float>();
    float* out = output.ptr<float>();


#ifdef _OPENMP
#pragma omp parallel for
#endif
    // (K x Cg*Hk*Wk) * (Cg*Hk*Wk x H0*W0)
    for (int task_id = 0; task_id < ntasks; task_id++) {
        float* inpbuf_task = &inpbuf_all[taskbufsize*task_id];
        int ngs0 = (N*ngroups*stripes_per_sample)*task_id/ntasks, ngs1 = (N*ngroups*stripes_per_sample)*(task_id+1)/ntasks;
        //printf("task id=%d: ngs0=%d, ngs1 = %d\n", task_id, ngs0, ngs1);
        for(int ngs = ngs0; ngs < ngs1; ngs++) {
            int n = ngs/(ngroups*stripes_per_sample), gs = ngs - n*(ngroups*stripes_per_sample);
            // yx0 and yx1 is for = matrix multiple.

            int g = gs/stripes_per_sample, yx0 = (gs - g*stripes_per_sample)*FX_CONV_NR;
            int yx1 = std::min(yx0 + FX_CONV_NR, H0*W0);
            // From output index to input index

            int y0 = yx0/W0, x0 = yx0 - y0*W0;
            size_t inp_plane_ofs = (n*ngroups + g)*Cg*Hi*Wi;
            int yi_ = y0*stride_y;//- pad_top;
            int xi_ = x0*stride_x;//- pad_left;
            if (yx1 < yx0 + FX_CONV_NR)
                memset(inpbuf_task, 0, taskbufsize*sizeof(inpbuf_task[0]));
            // 1. pack input data
            if (stride_x == 1 && yx1 == yx0 + FX_CONV_NR &&
                0 <= yi_ && yi_ + (Hk-1)*dilation_y < Hi &&
                0 <= xi_ && xi_ + FX_CONV_NR-1 + (Wk-1)*dilation_x < Wi) {
                // A. almost general case (stride_x == 1) when the whole slice
                //    (yx0 <= yx < yx0 + FX_CONV_NR) is inside one row of the input tensor, i.e.
                //    (n=n, c=c, y=y0, x0 <= x < x0+FX_CONV_NR).
                //    in this case we pack data for FX_CONV_NR output elements at once
                const float* inptr = inp + inp_plane_ofs + yi_*Wi + xi_;
                float* inpbuf = inpbuf_task;
                if ((Hk|Wk) == 1) {
                    // A1. special optimization for 1x1 kernel
                    for (int c = 0; c < Cg; c++, inptr += Hi*Wi, inpbuf += FX_CONV_NR) {
                        memcpy(inpbuf, inptr, FX_CONV_NR*sizeof(inpbuf[0]));
                    }
                } else {
                    for (int c = 0; c < Cg; c++, inptr += Hi*Wi) {
                        for (int k = 0; k < Hk*Wk; k++, inpbuf += FX_CONV_NR) {
                            const float* inptr_k = inptr + ofstab[k];
                            memcpy(inpbuf, inptr_k, FX_CONV_NR*sizeof(inpbuf[0]));
                        }
                    }
                }
            }
            else if ((Hk|Wk) == 1)
            {
                // B. 1x1 case, if it's not classified as A.
                //    in this case the input slice is always inside input tensor,
                //    but it may cross an input tensor row.
                for (int yx = yx0; yx < yx1; yx++) {
                    float* inpbuf = inpbuf_task + (yx - yx0);
                    yi_ = y0*stride_y;
                    xi_ = x0*stride_x;
                    const float* inptr = inp + inp_plane_ofs + yi_*Wi + xi_;
                    for (int c = 0; c < Cg; c++, inptr += Hi*Wi, inpbuf += FX_CONV_NR)
                        *inpbuf = *inptr;
                    if (++x0 >= W0) {
                        x0 = 0;
                        ++y0;
                    }
                }
            }
            else
            {
                for (int yx = yx0; yx < yx1; yx++)
                {
                    float* inpbuf = inpbuf_task + (yx - yx0);
                    yi_ = y0*stride_y; // - pad_top;
                    xi_ = x0*stride_x; // - pad_left;
                    if (0 <= yi_ && yi_ + (Hk-1)*dilation_y < Hi &&
                        0 <= xi_ && xi_ + (Wk-1)*dilation_x < Wi) {
                        // C. the (Hk x Wk) patch is inside input plane, do offset-based packing
                        const float* inptr = inp + inp_plane_ofs + yi_*Wi + xi_;
                        for (int c = 0; c < Cg; c++, inptr += Hi*Wi) {
                            for (int k = 0; k < Hk*Wk; k++, inpbuf += FX_CONV_NR)
                                *inpbuf = inptr[ofstab[k]];
                        }
                    }
                    else
                    {
                        // D. the slowest path where we need to check each element in the (Hk x Wk) patch
                        for (int k = 0; k < Hk*Wk; k++, inpbuf += FX_CONV_NR) {
                            int yi = yi_ + xytab[k*2];
                            int xi = xi_ + xytab[k*2+1];
                            if ((unsigned)yi < (unsigned)Hi &&
                                (unsigned)xi < (unsigned)Wi)
                            {
                                const float* inptr = inp + inp_plane_ofs + yi*Wi + xi;
                                for (int c = 0; c < Cg; c++, inptr += Hi*Wi)
                                {
                                    //if (y0 == 0 && x0 == 0 && yi == 0 && xi == 0)
                                    //    printf("c == %d: inpval = %.2f\n", c, *inptr);
                                    inpbuf[c*(Hk*Wk*FX_CONV_NR)] = *inptr;
                                }
                            }
                            else
                            {
                                for (int c = 0; c < Cg; c++)
                                    inpbuf[c*(Hk*Wk*FX_CONV_NR)] = 0.f;
                            }
                        }
                    }
                    if (++x0 >= W0)
                    {
                        x0 = 0;
                        ++y0;
                    }
                }
            }

            // 2. do convolution, compute Kg x (yx1 - yx0) part of the output tensor
            {
                int outstep0 = H0*W0;
                float* outptr0 = out + (n*ngroups + g)*Kg*outstep0 + yx0;
                // n * ngroups -> batch size, g is in this batch, kg is number of channel in this group, yx0 is the begin of this matrix.
                float cbuf[FX_CONV_MR*FX_CONV_NR];
                bool partial0 = yx1 - yx0 < FX_CONV_NR;
                for(int k = 0; k < Kg; k += FX_CONV_MR, outptr0 += outstep0*FX_CONV_MR)
                {
                    int dk = Kg - k < FX_CONV_MR ? Kg - k : FX_CONV_MR;
                    bool partial = partial0 || dk < FX_CONV_MR;
                    float* outptr = outptr0;
                    int outstep = outstep0;
                    if (partial)
                    {
                        outptr = cbuf;
                        outstep = FX_CONV_NR;
                    }
                    conv_block(HkWkCg, conv->weightsPtr+(g*Kg_aligned + k)*HkWkCg,
                               inpbuf_task, outptr, outstep, conv->biasPtr + Kg*g + k,
                               0.0f, FLT_MAX, reluActiv);

                    if (partial)
                    {
                        for (int k1 = 0; k1 < dk; k1++)
                            memcpy(outptr0 + k1*outstep0, &cbuf[k1*FX_CONV_NR],
                                   (yx1 - yx0)*sizeof(cbuf[0]));
                    }
                }
            }
        }
    }

    free(inpbuf_all);
}

}} // namespace cv::dnn



#endif //OPENCV_FAST_CONV_2D_HPP
