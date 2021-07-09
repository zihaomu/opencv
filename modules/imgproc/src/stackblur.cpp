// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2019-2021, Shenzhen Institute of Artificial Intelligence and
// Robotics for Society, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

static unsigned short const stackblur_mul[255] =
        {
                512,512,456,512,328,456,335,512,405,328,271,456,388,335,292,512,
                454,405,364,328,298,271,496,456,420,388,360,335,312,292,273,512,
                482,454,428,405,383,364,345,328,312,298,284,271,259,496,475,456,
                437,420,404,388,374,360,347,335,323,312,302,292,282,273,265,512,
                497,482,468,454,441,428,417,405,394,383,373,364,354,345,337,328,
                320,312,305,298,291,284,278,271,265,259,507,496,485,475,465,456,
                446,437,428,420,412,404,396,388,381,374,367,360,354,347,341,335,
                329,323,318,312,307,302,297,292,287,282,278,273,269,265,261,512,
                505,497,489,482,475,468,461,454,447,441,435,428,422,417,411,405,
                399,394,389,383,378,373,368,364,359,354,350,345,341,337,332,328,
                324,320,316,312,309,305,301,298,294,291,287,284,281,278,274,271,
                268,265,262,259,257,507,501,496,491,485,480,475,470,465,460,456,
                451,446,442,437,433,428,424,420,416,412,408,404,400,396,392,388,
                385,381,377,374,370,367,363,360,357,354,350,347,344,341,338,335,
                332,329,326,323,320,318,315,312,310,307,304,302,299,297,294,292,
                289,287,285,282,280,278,275,273,271,269,267,265,263,261,259
        };

static unsigned char const stackblur_shr[255] =
        {
                9, 11, 12, 13, 13, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17,
                17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19,
                19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20,
                20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21,
                21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22,
                22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
                22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23,
                23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
                23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
                23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
                23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
                24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
                24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
                24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
                24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24
        };


namespace cv{

// Stack Blur on Horizontal Direction
class ParallelStackBlurRow : public ParallelLoopBody
{
public:
    ParallelStackBlurRow (const Mat &_src, Mat &_dst, int radius): src(_src), dst(_dst) ,radius(radius)
    {

#define STACKBLUR_MAX_RADIUS 254

        const int cn = src.channels();

        CV_Assert(cn == 1 || cn == 3 || cn == 4);
        CV_Assert(radius <= STACKBLUR_MAX_RADIUS);

        width= dst.size().width;
        wm = width - 1;

        mul_sum = stackblur_mul[radius];
        shr_sum = stackblur_shr[radius];

    }

    ~ParallelStackBlurRow() {}

    ParallelStackBlurRow& operator=(const ParallelStackBlurRow &) { return *this; }

    virtual void operator ()(const Range& range) const CV_OVERRIDE
    {

        const int div = 2 * radius + 1;
        const int cn = src.channels();

        int row = range.start;
        int i, temp, xp, sp;
        uchar *src_ptr, *src_new;
        ushort *dst_ptr;

        // Const Stack Space
        uchar stack[div * 4];

        unsigned int sum[4];
        unsigned int sumIn[4];
        unsigned int sumOut[4];

        memset(sum, 0, sizeof(unsigned int) * 4);
        memset(sumIn, 0, sizeof(unsigned int) * 4);
        memset(sumOut, 0, sizeof(unsigned int) * 4);
        memset(stack, 0, sizeof(uchar) * 4 * div);

        if( cn == 1)
        {
            for (; row < range.end; row++)
            {
                memset(sum, 0, sizeof(unsigned int) * 4);
                memset(sumIn, 0, sizeof(unsigned int) * 4);
                memset(sumOut, 0, sizeof(unsigned int) * 4);
                memset(stack, 0, sizeof(uchar) * 4 * div);

                src_ptr = src.data + src.step.p[0] * row;

                for (i = 0; i <= radius; i++)
                {
                    temp = static_cast<int>(*src_ptr);
                    stack[i] = *src_ptr;
                    sum[0] += temp * (i + 1);
                    sumOut[0] += temp;
                }

                for (i = 1; i <= radius; i++)
                {
                    if (i <= wm) src_ptr += 1;
                    temp = static_cast<int>(*src_ptr);
                    stack[ i + radius] = *(src_ptr);
                    sum[0] += temp * (radius + 1 - i);
                    sumIn[0] += temp;  // Sum in is less than sum Out
                }

                sp = radius;
                xp = radius ;
                if (xp > wm) xp = wm;

                dst_ptr = (ushort *)(dst.data + dst.step.p[0] * row);
                src_ptr = src.data + src.step.p[0] * row + xp * cn;

                int stack_start= 0;

                for(i = 0; i < width; i++)
                {
                    stack_start = sp + div - radius;

                    if (stack_start >= div) stack_start -= div;

                    *(dst_ptr) = static_cast<ushort>((sum[0] * mul_sum) >> shr_sum);

                    sum[0] -= sumOut[0];
                    sumOut[0] -= static_cast<int>(stack[stack_start]);
                    src_new = src_ptr;

                    if(xp < wm)
                    {
                        src_new += 1;
                    }

                    stack[stack_start] = *(src_new);

                    sumIn[0] += static_cast<int>(*(src_new));
                    sum[0] += sumIn[0];

                    int sp1 = sp + 1;
                    sp1 &= -(sp1 < div);

                    temp = static_cast<int>(stack[sp1]);
                    sumOut[0] += temp;
                    sumIn[0] -= temp;

                    dst_ptr += cn;

                    if (xp < wm)
                    {
                        xp++;
                        src_ptr += cn;
                    }

                    ++sp;
                    if (sp >= div) sp = 0;
                }

            }
        }
        else if (cn == 3)
        {
            for (; row < range.end; row++)
            {
                memset(sum, 0, sizeof(unsigned int) * 4);
                memset(sumIn, 0, sizeof(unsigned int) * 4);
                memset(sumOut, 0, sizeof(unsigned int) * 4);
                memset(stack, 0, sizeof(uchar) * 4 * div);

                src_ptr = src.data + src.step.p[0] * row;
                for (i = 0; i <= radius; i++)
                {
                    temp = static_cast<int>(*src_ptr);
                    stack[i*cn] = *src_ptr;
                    sum[0] += temp * (i + 1);
                    sumOut[0] += temp;

                    temp = static_cast<int>(*(src_ptr + 1));
                    stack[i*cn + 1] = *(src_ptr + 1);
                    sum[1] += temp * (i + 1);
                    sumOut[1] += temp;

                    temp = static_cast<int>(*(src_ptr + 2));
                    stack[i*cn + 2] = *(src_ptr + 2);
                    sum[2] += temp * (i + 1);
                    sumOut[2] += temp;
                }

                for (i = 1; i <= radius; i++)
                {
                    if (i <= wm) src_ptr += cn;
                    temp = static_cast<int>(*src_ptr);
                    stack[ (i + radius) * cn] = *(src_ptr);
                    sum[0] += temp * (radius + 1 - i);
                    sumIn[0] += temp;

                    temp = static_cast<int>(*(src_ptr + 1));
                    stack[ (i + radius) * cn + 1] = *(src_ptr + 1);
                    sum[1] += temp * (radius + 1 - i);
                    sumIn[1] += temp;

                    temp = static_cast<int>(*(src_ptr + 2));
                    stack[ (i + radius) * cn + 2] = *(src_ptr + 2);
                    sum[2] += temp * (radius + 1 - i);
                    sumIn[2] += temp;
                }

                sp = radius;
                xp = radius ;
                if (xp > wm) xp = wm;

                dst_ptr = (ushort *)(dst.data + dst.step.p[0] * row);
                src_ptr = src.data + src.step.p[0] * row + xp * cn;

                int stack_start= 0;

                for(i = 0; i < width; i++)
                {
                    stack_start = sp + div - radius;

                    if (stack_start >= div) stack_start -= div;

                    *(dst_ptr) = static_cast<ushort>((sum[0] * mul_sum) >> shr_sum);
                    *(dst_ptr + 1) = static_cast<ushort>((sum[1] * mul_sum) >> shr_sum);
                    *(dst_ptr + 2) = static_cast<ushort>((sum[2] * mul_sum) >> shr_sum);

                    sum[0] -= sumOut[0];
                    sum[1] -= sumOut[1];
                    sum[2] -= sumOut[2];

                    sumOut[0] -= static_cast<int>(stack[stack_start*cn]);
                    sumOut[1] -= static_cast<int>(stack[stack_start*cn + 1]);
                    sumOut[2] -= static_cast<int>(stack[stack_start*cn + 2]);

                    src_new = src_ptr;

                    if(xp < wm)
                    {
                        src_new += cn;
                    }

                    stack[stack_start*cn] = *(src_new);
                    stack[stack_start*cn + 1] = *(src_new + 1);
                    stack[stack_start*cn + 2] = *(src_new + 2);

                    sumIn[0] += static_cast<int>(*(src_new));
                    sumIn[1] += static_cast<int>(*(src_new + 1));
                    sumIn[2] += static_cast<int>(*(src_new + 2));

                    sum[0] += sumIn[0];
                    sum[1] += sumIn[1];
                    sum[2] += sumIn[2];

                    int sp1 = sp + 1;
                    sp1 &= -(sp1 < div);

                    temp = static_cast<int>(stack[sp1*cn]);
                    sumOut[0] += temp;
                    sumIn[0] -= temp;

                    temp = static_cast<int>(stack[sp1*cn + 1]);
                    sumOut[1] += temp;
                    sumIn[1] -= temp;

                    temp = static_cast<int>(stack[sp1*cn + 2]);
                    sumOut[2] += temp;
                    sumIn[2] -= temp;

                    dst_ptr += cn;

                    if (xp < wm)
                    {
                        xp++;
                        src_ptr += cn;
                    }

                    ++sp;
                    if (sp >= div) sp = 0;
                }

            }
        }
        else if (cn == 4)
        {

#if CV_SIMD
            v_uint32x4 v_sum = v_setzero_u32();
            v_uint32x4 v_sumIn = v_setzero_u32();
            v_uint32x4 v_sumOut = v_setzero_u32();
            v_uint32x4 v_temp = v_setzero_u32();
            v_uint32x4 v_mul_sum = v_setall_u32((int) mul_sum);
            v_uint32x4 v_stack[div];

            for(int k = 0; k < div; k++)
            {
                v_stack[k] = v_setzero_u32();
            }

#endif
            for (; row < range.end; row++)
            {

#ifdef CV_SIMD
                v_sum = v_setzero_u32();
                v_sumIn = v_setzero_u32();
                v_sumOut = v_setzero_u32();
#elif
                memset(sum, 0, sizeof(unsigned int) * 4);
                memset(sumIn, 0, sizeof(unsigned int) * 4);
                memset(sumOut, 0, sizeof(unsigned int) * 4);
                memset(stack, 0, sizeof(uchar) * 4 * div);
#endif

                src_ptr = src.data + src.step.p[0] * row;
                for (i = 0; i <= radius; i++)
                {
#ifdef CV_SIMD
                    v_stack[i] = v_load_expand_q(src_ptr);
                    v_temp = v_stack[i];
                    v_sum += v_temp * v_setall_u32(i + 1);
                    v_sumOut += v_temp;
#elif
                    temp = static_cast<int>(*src_ptr);
                    stack[i*cn] = *src_ptr;
                    sum[0] += temp * (i + 1);
                    sumOut[0] += temp;

                    temp = static_cast<int>(*(src_ptr + 1));
                    stack[i*cn + 1] = *(src_ptr + 1);
                    sum[1] += temp * (i + 1);
                    sumOut[1] += temp;

                    temp = static_cast<int>(*(src_ptr + 2));
                    stack[i*cn + 2] = *(src_ptr + 2);
                    sum[2] += temp * (i + 1);
                    sumOut[2] += temp;

                    temp = static_cast<int>(*(src_ptr + 3));
                    stack[i*cn + 3] = *(src_ptr + 3);
                    sum[3] += temp * (i + 1);
                    sumOut[3] += temp;
#endif
                }

                for (i = 1; i <= radius; i++)
                {
                    if (i <= wm) src_ptr += cn;
#ifdef CV_SIMD
                    v_stack[i + radius] = v_load_expand_q(src_ptr);
                    v_temp = v_stack[i + radius];
                    v_sum +=  v_temp * v_setall_u32(radius + 1 - i);
                    v_sumIn += v_temp;
#elif
                    temp = static_cast<int>(*src_ptr);
                    stack[ (i + radius) * cn] = *(src_ptr);
                    sum[0] += temp * (radius + 1 - i);
                    sumIn[0] += temp;

                    temp = static_cast<int>(*(src_ptr + 1));
                    stack[ (i + radius) * cn + 1] = *(src_ptr + 1);
                    sum[1] += temp * (radius + 1 - i);
                    sumIn[1] += temp;

                    temp = static_cast<int>(*(src_ptr + 2));
                    stack[ (i + radius) * cn + 2] = *(src_ptr + 2);
                    sum[2] += temp * (radius + 1 - i);
                    sumIn[2] += temp;

                    temp = static_cast<int>(*(src_ptr + 3));
                    stack[ (i + radius) * cn + 3] = *(src_ptr + 3);
                    sum[3] += temp * (radius + 1 - i);
                    sumIn[3] += temp;
#endif
                }

                sp = radius;
                xp = radius ;

                if (xp > wm) xp = wm;

                dst_ptr = (ushort *)(dst.data + dst.step.p[0] * row);
                src_ptr = src.data + src.step.p[0] * row + xp * cn;

                int stack_start= 0;
                for(i = 0; i < width; i++)
                {
                    stack_start = sp + div - radius;
                    if (stack_start >= div) stack_start -= div;
#ifdef CV_SIMD
//                        v_store_low(dst_ptr, v_pack(v_pack((v_sum * v_mul_sum)>> shr_sum, v_setzero_u32()), v_setzero_u16()));
                    v_store_low(dst_ptr, v_pack((v_sum * v_mul_sum)>> shr_sum, v_setzero_u32()));

                    v_sum -= v_sumOut;
                    v_sumOut -= v_stack[stack_start];
#elif
                    *(dst_ptr) = static_cast<ushort>((sum[0] * mul_sum) >> shr_sum);
                    *(dst_ptr + 1) = static_cast<ushort>((sum[1] * mul_sum) >> shr_sum);
                    *(dst_ptr + 2) = static_cast<ushort>((sum[2] * mul_sum) >> shr_sum);
                    *(dst_ptr + 3) = static_cast<ushort>((sum[3] * mul_sum) >> shr_sum);

                    sum[0] -= sumOut[0];
                    sum[1] -= sumOut[1];
                    sum[2] -= sumOut[2];
                    sum[3] -= sumOut[3];

                    sumOut[0] -= static_cast<int>(stack[stack_start*cn]);
                    sumOut[1] -= static_cast<int>(stack[stack_start*cn + 1]);
                    sumOut[2] -= static_cast<int>(stack[stack_start*cn + 2]);
                    sumOut[3] -= static_cast<int>(stack[stack_start*cn + 3]);
#endif

                    src_new = src_ptr;
                    if(xp < wm)
                    {
                        src_new += cn;
                    }
#ifdef CV_SIMD
                    v_stack[ stack_start ] = vx_load_expand_q(src_new);
                    v_sumIn += v_stack[ stack_start ];
                    v_sum += v_sumIn;
#elif
                    stack[stack_start*cn] = *(src_new);
                    stack[stack_start*cn + 1] = *(src_new + 1);
                    stack[stack_start*cn + 2] = *(src_new + 2);
                    stack[stack_start*cn + 3] = *(src_new + 3);

                    sumIn[0] += static_cast<int>(*(src_new));
                    sumIn[1] += static_cast<int>(*(src_new + 1));
                    sumIn[2] += static_cast<int>(*(src_new + 2));
                    sumIn[3] += static_cast<int>(*(src_new + 3));

                    sum[0] += sumIn[0];
                    sum[1] += sumIn[1];
                    sum[2] += sumIn[2];
                    sum[3] += sumIn[3];
#endif

                    int sp1 = sp + 1;
                    sp1 &= -(sp1 < div);

#ifdef CV_SIMD
                    v_temp = v_stack[sp1];
                    v_sumOut += v_temp;
                    v_sumIn -= v_temp;
#elif
                    temp = static_cast<int>(stack[sp1*cn]);
                    sumOut[0] += temp;
                    sumIn[0] -= temp;

                    temp = static_cast<int>(stack[sp1*cn + 1]);
                    sumOut[1] += temp;
                    sumIn[1] -= temp;

                    temp = static_cast<int>(stack[sp1*cn + 2]);
                    sumOut[2] += temp;
                    sumIn[2] -= temp;

                    temp = static_cast<int>(stack[sp1*cn + 3]);
                    sumOut[3] += temp;
                    sumIn[3] -= temp;
#endif

                    dst_ptr += cn;

                    if (xp < wm)
                    {
                        xp++;
                        src_ptr += cn;
                    }

                    ++sp;
                    if (sp >= div) sp = 0;
                }
            }
        }
    }

private:
    const Mat &src;
    Mat &dst;
    int radius;
    int width;
    int wm;
    unsigned int mul_sum;
    int shr_sum;

};

// Stack Blur on Vertical Direction
class ParallelStackBlurColumn : public ParallelLoopBody
{
public:
    ParallelStackBlurColumn (const Mat & _src, Mat &_dst, int radius):src(_src), dst(_dst) ,radius(radius)
    {

        cn = src.channels();
        widthElem = dst.step.p[0];
        height = src.size().height;
        hm = height - 1;
        mul_sum = stackblur_mul[radius];
        shr_sum = stackblur_shr[radius];
    }

    ~ParallelStackBlurColumn() {}

    ParallelStackBlurColumn& operator=(const ParallelStackBlurColumn &) { return *this; }

    virtual void operator ()(const Range& range) const CV_OVERRIDE
    {
        const int div = 2*radius + 1;
        int col = range.start;
        int i, temp, yp, sp;
        uchar *dst_ptr;
        ushort *src_ptr, *src_new;

#if CV_SIMD128
        const int vlen = v_uint16x8::nlanes;

        v_uint32x4 v_sum00 = v_setzero_u32();
        v_uint32x4 v_sum01 = v_setzero_u32();
        v_uint32x4 v_sum02 = v_setzero_u32();
        v_uint32x4 v_sum03 = v_setzero_u32();

        v_uint32x4 v_sumIn00 = v_setzero_u32();
        v_uint32x4 v_sumIn01 = v_setzero_u32();
        v_uint32x4 v_sumIn02 = v_setzero_u32();
        v_uint32x4 v_sumIn03 = v_setzero_u32();

        v_uint32x4 v_sumOut00 = v_setzero_u32();
        v_uint32x4 v_sumOut01 = v_setzero_u32();
        v_uint32x4 v_sumOut02 = v_setzero_u32();
        v_uint32x4 v_sumOut03 = v_setzero_u32();

        v_uint32x4 v_temp00 = v_setzero_u32();
        v_uint32x4 v_temp01 = v_setzero_u32();
        v_uint32x4 v_temp02 = v_setzero_u32();
        v_uint32x4 v_temp03 = v_setzero_u32();

        v_uint16x8 v_stack0[div];
        v_uint16x8 v_stack1[div];

        for(int k = 0; k < div; k++)
        {
            v_stack0[k] = v_setzero_u16();
            v_stack1[k] = v_setzero_u16();
        }

        v_uint32x4 v_mul_sum = v_setall_u32((int) mul_sum);

        for (; col < range.end * cn; col += 2 * vlen) {
            src_ptr = (ushort * )(src.data) + col;

            v_sum00 = v_setzero_u32();
            v_sum01 = v_setzero_u32();
            v_sum02 = v_setzero_u32();
            v_sum03 = v_setzero_u32();

            v_sumIn00 = v_setzero_u32();
            v_sumIn01 = v_setzero_u32();
            v_sumIn02 = v_setzero_u32();
            v_sumIn03 = v_setzero_u32();

            v_sumOut00 = v_setzero_u32();
            v_sumOut01 = v_setzero_u32();
            v_sumOut02 = v_setzero_u32();
            v_sumOut03 = v_setzero_u32();

            v_temp00 = v_setzero_u32();
            v_temp01 = v_setzero_u32();
            v_temp02 = v_setzero_u32();
            v_temp03 = v_setzero_u32();

            for (i = 0; i <= radius; i++) {

                v_stack0[i] = v_load(src_ptr);
                v_stack1[i] = v_load(src_ptr + vlen);

                v_expand(v_stack0[i], v_temp00, v_temp01 );
                v_expand(v_stack1[i], v_temp02, v_temp03 );

                v_sum00 += v_temp00 * v_setall_u32(i + 1);
                v_sum01 += v_temp01 * v_setall_u32(i + 1);
                v_sum02 += v_temp02 * v_setall_u32(i + 1);
                v_sum03 += v_temp03 * v_setall_u32(i + 1);

                v_sumOut00 += v_temp00;
                v_sumOut01 += v_temp01;
                v_sumOut02 += v_temp02;
                v_sumOut03 += v_temp03;

            }

            for (i = 1; i <= radius; i++) {
                if (i <= hm) src_ptr += widthElem;

                v_stack0[i + radius] = v_load(src_ptr);
                v_stack1[i + radius] = v_load(src_ptr + vlen);

                v_expand(v_stack0[i + radius], v_temp00, v_temp01 );
                v_expand(v_stack1[i + radius], v_temp02, v_temp03 );

                v_sum00 += v_temp00 * v_setall_u32(radius + 1 - i);
                v_sum01 += v_temp01 * v_setall_u32(radius + 1 - i);
                v_sum02 += v_temp02 * v_setall_u32(radius + 1 - i);
                v_sum03 += v_temp03 * v_setall_u32(radius + 1 - i);

                v_sumIn00 += v_temp00;
                v_sumIn01 += v_temp01;
                v_sumIn02 += v_temp02;
                v_sumIn03 += v_temp03;
            }

            sp = radius;
            yp = radius;

            if (yp > hm) yp = hm;

            dst_ptr = dst.data + col;
            src_ptr = (ushort *)(src.data) + col + yp * widthElem;

            int stack_start = 0;

            for (i = 0; i < height; i++)
            {
                stack_start = sp + div - radius;

                if (stack_start >= div) stack_start -= div;

                v_uint16x8 res1 = v_pack((v_sum00 * v_mul_sum) >> shr_sum, (v_sum01 * v_mul_sum) >> shr_sum);
                v_uint16x8 res2 = v_pack((v_sum02 * v_mul_sum) >> shr_sum, (v_sum03 * v_mul_sum) >> shr_sum);
                v_uint8x16 res = v_pack(res1, res2);

                v_store(dst_ptr, res);
                v_sum00 -= v_sumOut00;
                v_sum01 -= v_sumOut01;
                v_sum02 -= v_sumOut02;
                v_sum03 -= v_sumOut03;

                v_expand(v_stack0[stack_start], v_temp00, v_temp01 );
                v_expand(v_stack1[stack_start], v_temp02, v_temp03 );

                v_sumOut00 -= v_temp00;
                v_sumOut01 -= v_temp01;
                v_sumOut02 -= v_temp02;
                v_sumOut03 -= v_temp03;

                src_new = src_ptr;
                if (yp < hm) {
                    src_new += widthElem;
                }

                v_stack0[stack_start] = v_load(src_new);
                v_stack1[stack_start] = v_load(src_new + vlen);

                v_expand(v_stack0[stack_start], v_temp00, v_temp01);
                v_expand(v_stack1[stack_start], v_temp02, v_temp03);

                v_sumIn00 += v_temp00;
                v_sumIn01 += v_temp01;
                v_sumIn02 += v_temp02;
                v_sumIn03 += v_temp03;

                v_sum00 += v_sumIn00;
                v_sum01 += v_sumIn01;
                v_sum02 += v_sumIn02;
                v_sum03 += v_sumIn03;

                int sp1 = sp + 1;
                sp1 &= -(sp1 < div);

                v_expand(v_stack0[sp1], v_temp00, v_temp01);
                v_expand(v_stack1[sp1], v_temp02, v_temp03);

                v_sumOut00 += v_temp00;
                v_sumOut01 += v_temp01;
                v_sumOut02 += v_temp02;
                v_sumOut03 += v_temp03;

                v_sumIn00 -= v_temp00;
                v_sumIn01 -= v_temp01;
                v_sumIn02 -= v_temp02;
                v_sumIn03 -= v_temp03;

                dst_ptr += widthElem;

                if (yp < hm)
                {
                    yp++;
                    src_ptr += widthElem;
                }

                ++sp;
                if (sp >= div) sp = 0;
            }
        }
#endif

        for (; col < range.end * cn; col++) {
            src_ptr = (ushort *)(src.data)+ col;
            unsigned int sum = 0;
            unsigned int sumIn = 0;
            unsigned int sumOut = 0;
            ushort stack[div];

            memset(stack, 0, sizeof(ushort) * div);

            for (i = 0; i <= radius; i++)
            {
                temp = static_cast<int>(*(src_ptr));
                stack[i] = *(src_ptr);
                sum += temp * (i + 1);
                sumOut += temp;
            }

            for (i = 1; i <= radius; i++)
            {
                if (i <= hm) src_ptr += src.step.p[0];
                temp = static_cast<int>(*(src_ptr));
                stack[ i + radius] = *(src_ptr);
                sum += temp * (radius + 1 - i);
                sumIn += temp;
            }

            sp = radius;
            yp = radius ;

            if (yp > hm) yp = hm;

            dst_ptr = dst.data + col;
            src_ptr = (ushort *)(src.data) + col + yp * widthElem;

            int stack_start= 0;

            for(i = 0; i < dst.size().height; i++)
            {
                stack_start = sp + div - radius;
                if (stack_start >= div) stack_start -= div;

                    *(dst_ptr) = static_cast<uchar>((sum * mul_sum) >> shr_sum);

                    sum -= sumOut;
                    sumOut -= static_cast<int>(stack[stack_start]);
                    src_new = src_ptr;

                    if(yp < hm)
                    {
                        src_new += widthElem;
                    }

                    stack[stack_start] = *(src_new);

                    sumIn += static_cast<int>(*(src_new));
                    sum += sumIn;

                    int sp1 = sp + 1;
                    sp1 &= -(sp1 < div);
                    temp = static_cast<int>(stack[sp1]);

                    sumOut += temp;
                    sumIn -= temp;

                dst_ptr += widthElem;

                if (yp < hm)
                {
                    yp++;
                    src_ptr += widthElem;
                }

                ++sp;
                if (sp >= div) sp = 0;
            }
        }
    }

private:
    const Mat &src;
    Mat &dst;
    int radius;
    int cn;
    int height;
    int widthElem;
    int hm;
    unsigned int mul_sum;
    int shr_sum;
};

void stackBlur(InputArray _src, OutputArray _dst, int radius)
{
    CV_INSTRUMENT_REGION();
    CV_Assert(!_src.empty());

    // The current version only supports CV_16U depth of InputArray.
    CV_Assert(_src.depth() == CV_8U);
    Mat src = _src.getMat();
    int stype = src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype);
    _dst.create( src.size(), CV_MAKETYPE(sdepth, cn) );

    Mat temp(src.size(), CV_MAKETYPE(CV_16U, cn)), gray;

    Mat dst = _dst.getMat();
    int numOfThreads = std::max(1, std::min(getNumThreads(), getNumberOfCPUs()));

    if (dst.rows / numOfThreads < 3)
        numOfThreads = std::max(1, dst.rows / 3);

    // Output of StackBlurRow is CV_16U.
    parallel_for_(Range(0, src.rows), ParallelStackBlurRow(src, temp, radius), numOfThreads);

    // The vertical branch is time consuming. And multiple threads cannot achieve guaranteed speedup ratio.
    parallel_for_(Range(0, temp.cols), ParallelStackBlurColumn(temp, dst, radius), numOfThreads);

// Horizontal only
//        parallel_for_(Range(0, src.rows), ParallelStackBlurRow(src, temp, radius), numOfThreads);
//        temp.convertTo(dst, CV_8U);

// Vertical Only
//        src.convertTo(temp, CV_16U);
//        parallel_for_(Range(0, temp.cols), ParallelStackBlurColumn(temp, dst, radius), 1);
}

} //namespace