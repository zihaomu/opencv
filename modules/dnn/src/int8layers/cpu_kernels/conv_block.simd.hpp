// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void convBlock_INT8(int np, const float* a, const float* b, float* c, int ldc, bool init_c, const int convMR, const int convNR);

#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_AVX

#if !CV_FMA3 // AVX workaround
#undef _mm256_fmadd_ps
#define _mm256_fmadd_ps(a, b, c) _mm256_add_ps(c, _mm256_mul_ps(a, b))
#endif

void convBlock_int8(int np, const float* a, const float* b, float* c, int ldc, bool init_c, const int convMR, const int convNR)
{
    CV_Assert(convMR == 4 && convNR == 24);
    __m256 c00 = _mm256_set1_ps(0.f), c01 = c00, c02 = c00;
    __m256 c10 = c00, c11 = c00, c12 = c00;
    __m256 c20 = c00, c21 = c00, c22 = c00;
    __m256 c30 = c00, c31 = c00, c32 = c00;

    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps(), b2 = _mm256_setzero_ps();

    for (int p = 0; p < np; p++, a += convMR, b += convNR)
    {
        a0 = _mm256_set1_ps(a[0]), a1 = _mm256_set1_ps(a[1]);
        b0 = _mm256_load_ps(b), b1 = _mm256_load_ps(b + 8), b2 = _mm256_load_ps(b + 16);

        c00 = _mm256_fmadd_ps(b0, a0, c00);
        c01 = _mm256_fmadd_ps(b1, a0, c01);
        c02 = _mm256_fmadd_ps(b2, a0, c02);

        c10 = _mm256_fmadd_ps(b0, a1, c10);
        c11 = _mm256_fmadd_ps(b1, a1, c11);
        c12 = _mm256_fmadd_ps(b2, a1, c12);

        a0 = _mm256_set1_ps(a[2]), a1 = _mm256_set1_ps(a[3]);

        c20 = _mm256_fmadd_ps(b0, a0, c20);
        c21 = _mm256_fmadd_ps(b1, a0, c21);
        c22 = _mm256_fmadd_ps(b2, a0, c22);

        c30 = _mm256_fmadd_ps(b0, a1, c30);
        c31 = _mm256_fmadd_ps(b1, a1, c31);
        c32 = _mm256_fmadd_ps(b2, a1, c32);
    }

    if (!init_c)
    {
        c00 = _mm256_add_ps(c00, _mm256_load_ps(c));
        c01 = _mm256_add_ps(c01, _mm256_load_ps(c + 8));
        c02 = _mm256_add_ps(c02, _mm256_load_ps(c + 16));

        c10 = _mm256_add_ps(c10, _mm256_load_ps(c + ldc));
        c11 = _mm256_add_ps(c11, _mm256_load_ps(c + ldc + 8));
        c12 = _mm256_add_ps(c12, _mm256_load_ps(c + ldc + 16));

        c20 = _mm256_add_ps(c20, _mm256_load_ps(c + ldc*2));
        c21 = _mm256_add_ps(c21, _mm256_load_ps(c + ldc*2 + 8));
        c22 = _mm256_add_ps(c22, _mm256_load_ps(c + ldc*2 + 16));

        c30 = _mm256_add_ps(c30, _mm256_load_ps(c + ldc*3));
        c31 = _mm256_add_ps(c31, _mm256_load_ps(c + ldc*3 + 8));
        c32 = _mm256_add_ps(c32, _mm256_load_ps(c + ldc*3 + 16));
    }

    _mm256_storeu_ps(c, c00), _mm256_storeu_ps(c+8, c01), _mm256_storeu_ps(c+16, c02);
    _mm256_storeu_ps(c + ldc, c10), _mm256_storeu_ps(c + ldc + 8, c11), _mm256_storeu_ps(c + ldc + 16, c12);
    _mm256_storeu_ps(c + ldc*2, c20), _mm256_storeu_ps(c + ldc*2 + 8, c21), _mm256_storeu_ps(c + ldc*2 + 16, c22);
    _mm256_storeu_ps(c + ldc*3, c30), _mm256_storeu_ps(c + ldc*3 + 8, c31), _mm256_storeu_ps(c + ldc*3 + 16, c32);
    _mm256_zeroupper();
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END

// NEON code work around.
namespace opt_NEON
{
#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_NEON

void convBlock_INT8(int np, const char* _a, const char* _b, int* c, int ldc, bool init_c, const int width, const int convMR, const int convNR)
{
    CV_Assert(convMR == 4 && CONV_PACKN == 4);
    const int8_t* a = (const int8_t*)_a;
    const int8_t* b = (const int8_t*)_b;


#if CV_NEON_AARCH64
    if (width > 12) // AARCH64
    {
        int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c02 = c00, c03 = c00, c04 = c00, c05 = c00;
        int32x4_t c10 = vdupq_n_s32(0), c11 = c10, c12 = c10, c13 = c10, c14 = c10, c15 = c10;
        int32x4_t c20 = vdupq_n_s32(0), c21 = c20, c22 = c20, c23 = c20, c24 = c20, c25 = c20;
        int32x4_t c30 = vdupq_n_s32(0), c31 = c30, c32 = c30, c33 = c30, c34 = c30, c35 = c30;

        for (int p = 0; p < np; p += 4, a += 4 * convMR, b += 4 * convNR)
        {
            int8x16_t a0 = vld1q_s8(a), b0, b1, b2;

            b0 = vld1q_s8(b); b1 = vld1q_s8(b + 16); b2 = vld1q_s8(b + 32);
            c00 = vdotq_laneq_s32(c00, b0, a0, 0);
            c01 = vdotq_laneq_s32(c01, b1, a0, 0);
            c02 = vdotq_laneq_s32(c02, b2, a0, 0);
            c10 = vdotq_laneq_s32(c10, b0, a0, 1);
            c11 = vdotq_laneq_s32(c11, b1, a0, 1);
            c12 = vdotq_laneq_s32(c12, b2, a0, 1);
            c20 = vdotq_laneq_s32(c20, b0, a0, 2);
            c21 = vdotq_laneq_s32(c21, b1, a0, 2);
            c22 = vdotq_laneq_s32(c22, b2, a0, 2);
            c30 = vdotq_laneq_s32(c30, b0, a0, 3);
            c31 = vdotq_laneq_s32(c31, b1, a0, 3);
            c32 = vdotq_laneq_s32(c32, b2, a0, 3);

            b0 = vld1q_s8(b + 48); b1 = vld1q_s8(b + 64); b2 = vld1q_s8(b + 80);
            c03 = vdotq_laneq_s32(c03, b0, a0, 0);
            c04 = vdotq_laneq_s32(c04, b1, a0, 0);
            c05 = vdotq_laneq_s32(c05, b2, a0, 0);
            c13 = vdotq_laneq_s32(c13, b0, a0, 1);
            c14 = vdotq_laneq_s32(c14, b1, a0, 1);
            c15 = vdotq_laneq_s32(c15, b2, a0, 1);
            c23 = vdotq_laneq_s32(c23, b0, a0, 2);
            c24 = vdotq_laneq_s32(c24, b1, a0, 2);
            c25 = vdotq_laneq_s32(c25, b2, a0, 2);
            c33 = vdotq_laneq_s32(c33, b0, a0, 3);
            c34 = vdotq_laneq_s32(c34, b1, a0, 3);
            c35 = vdotq_laneq_s32(c35, b2, a0, 3);
        }

        if (!init_c)
        {
#undef NEON_UPDATE_QCONV_BLOCK
#define NEON_UPDATE_QCONV_BLOCK(i) \
        c##i##0 = vaddq_s32(c##i##0, vld1q_s32(c+i*ldc)); \
        c##i##1 = vaddq_s32(c##i##1, vld1q_s32(c+i*ldc+4)); \
        c##i##2 = vaddq_s32(c##i##2, vld1q_s32(c+i*ldc+8)); \
        c##i##3 = vaddq_s32(c##i##3, vld1q_s32(c+i*ldc+12)); \
        c##i##4 = vaddq_s32(c##i##4, vld1q_s32(c+i*ldc+16)); \
        c##i##5 = vaddq_s32(c##i##5, vld1q_s32(c+i*ldc+20))

        NEON_UPDATE_QCONV_BLOCK(0);
        NEON_UPDATE_QCONV_BLOCK(1);
        NEON_UPDATE_QCONV_BLOCK(2);
        NEON_UPDATE_QCONV_BLOCK(3);
        }

#undef NEON_STORE_QCONV_BLOCK
#define NEON_STORE_QCONV_BLOCK(i) \
        vst1q_s32(c+i*ldc, c##i##0); \
        vst1q_s32(c+i*ldc+4, c##i##1); \
        vst1q_s32(c+i*ldc+8, c##i##2); \
        vst1q_s32(c+i*ldc+12, c##i##3); \
        vst1q_s32(c+i*ldc+16, c##i##4); \
        vst1q_s32(c+i*ldc+20, c##i##5)

        NEON_STORE_QCONV_BLOCK(0);
        NEON_STORE_QCONV_BLOCK(1);
        NEON_STORE_QCONV_BLOCK(2);
        NEON_STORE_QCONV_BLOCK(3);
    }
    else
#endif
    {
        int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c02 = c00;
        int32x4_t c10 = vdupq_n_s32(0), c11 = c10, c12 = c10;
        int32x4_t c20 = vdupq_n_s32(0), c21 = c20, c22 = c20;
        int32x4_t c30 = vdupq_n_s32(0), c31 = c30, c32 = c30;

        for (int p = 0; p < np; p += 4, a += 4 * convMR, b += 4 * convNR)
        {
            int8x16_t a0 = vld1q_s8(a), b0, b1, b2;

            b0 = vld1q_s8(b); b1 = vld1q_s8(b + 16); b2 = vld1q_s8(b + 32);
            c00 = vdotq_laneq_s32(c00, b0, a0, 0);
            c01 = vdotq_laneq_s32(c01, b1, a0, 0);
            c02 = vdotq_laneq_s32(c02, b2, a0, 0);
            c10 = vdotq_laneq_s32(c10, b0, a0, 1);
            c11 = vdotq_laneq_s32(c11, b1, a0, 1);
            c12 = vdotq_laneq_s32(c12, b2, a0, 1);
            c20 = vdotq_laneq_s32(c20, b0, a0, 2);
            c21 = vdotq_laneq_s32(c21, b1, a0, 2);
            c22 = vdotq_laneq_s32(c22, b2, a0, 2);
            c30 = vdotq_laneq_s32(c30, b0, a0, 3);
            c31 = vdotq_laneq_s32(c31, b1, a0, 3);
            c32 = vdotq_laneq_s32(c32, b2, a0, 3);
        }

        if (!init_c)
        {
#undef NEON_UPDATE_QCONV_BLOCK
#define NEON_UPDATE_QCONV_BLOCK(i) \
            c##i##0 = vaddq_s32(c##i##0, vld1q_s32(c+i*ldc)); \
            c##i##1 = vaddq_s32(c##i##1, vld1q_s32(c+i*ldc+4)); \
            c##i##2 = vaddq_s32(c##i##2, vld1q_s32(c+i*ldc+8))

            NEON_UPDATE_QCONV_BLOCK(0);
            NEON_UPDATE_QCONV_BLOCK(1);
            NEON_UPDATE_QCONV_BLOCK(2);
            NEON_UPDATE_QCONV_BLOCK(3);
        }

#undef NEON_STORE_QCONV_BLOCK
#define NEON_STORE_QCONV_BLOCK(i) \
        vst1q_s32(c+i*ldc, c##i##0); \
        vst1q_s32(c+i*ldc+4, c##i##1); \
        vst1q_s32(c+i*ldc+8, c##i##2)

        NEON_STORE_QCONV_BLOCK(0);
        NEON_STORE_QCONV_BLOCK(1);
        NEON_STORE_QCONV_BLOCK(2);
        NEON_STORE_QCONV_BLOCK(3);
    }
}

#endif
}
}} // namespace cv::dnn
