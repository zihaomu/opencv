// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include <opencv2/imgproc.hpp>


namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN


Mat blobFromImage(InputArray image, double scalefactor, const Size& size,
        const Scalar& mean, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    Mat blob;
    blobFromImage(image, blob, scalefactor, size, mean, swapRB, crop, ddepth);
    return blob;
}

void blobFromImage(InputArray image, OutputArray blob, double scalefactor,
        const Size& size, const Scalar& mean, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    std::vector<Mat> images(1, image.getMat());
    blobFromImages(images, blob, scalefactor, size, mean, swapRB, crop, ddepth);
}

Mat blobFromImages(InputArrayOfArrays images, double scalefactor, Size size,
        const Scalar& mean, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    Mat blob;
    blobFromImages(images, blob, scalefactor, size, mean, swapRB, crop, ddepth);
    return blob;
}

void blobFromImages(InputArrayOfArrays images_, OutputArray blob_, double scalefactor,
        Size size, const Scalar& mean_, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    CV_CheckType(ddepth, ddepth == CV_32F || ddepth == CV_8U, "Blob depth should be CV_32F or CV_8U");
    if (ddepth == CV_8U)
    {
        CV_CheckEQ(scalefactor, 1.0, "Scaling is not supported for CV_8U blob depth");
        CV_Assert(mean_ == Scalar() && "Mean subtraction is not supported for CV_8U blob depth");
    }

    std::vector<Mat> images;
    images_.getMatVector(images);
    CV_Assert(!images.empty());
    for (size_t i = 0; i < images.size(); i++)
    {
        Size imgSize = images[i].size();
        if (size == Size())
            size = imgSize;
        if (size != imgSize)
        {
            if (crop)
            {
                float resizeFactor = std::max(size.width / (float)imgSize.width,
                        size.height / (float)imgSize.height);
                resize(images[i], images[i], Size(), resizeFactor, resizeFactor, INTER_LINEAR);
                Rect crop(Point(0.5 * (images[i].cols - size.width),
                                  0.5 * (images[i].rows - size.height)),
                        size);
                images[i] = images[i](crop);
            }
            else
                resize(images[i], images[i], size, 0, 0, INTER_LINEAR);
        }
        if (images[i].depth() == CV_8U && ddepth == CV_32F)
            images[i].convertTo(images[i], CV_32F);
        Scalar mean = mean_;
        if (swapRB)
            std::swap(mean[0], mean[2]);

        images[i] -= mean;
        images[i] *= scalefactor;
    }

    size_t nimages = images.size();
    Mat image0 = images[0];
    int nch = image0.channels();
    CV_Assert(image0.dims == 2);
    if (nch == 3 || nch == 4)
    {
        int sz[] = { (int)nimages, nch, image0.rows, image0.cols };
        blob_.create(4, sz, ddepth);
        Mat blob = blob_.getMat();
        Mat ch[4];

        for (size_t i = 0; i < nimages; i++)
        {
            const Mat& image = images[i];
            CV_Assert(image.depth() == blob_.depth());
            nch = image.channels();
            CV_Assert(image.dims == 2 && (nch == 3 || nch == 4));
            CV_Assert(image.size() == image0.size());

            for (int j = 0; j < nch; j++)
                ch[j] = Mat(image.rows, image.cols, ddepth, blob.ptr((int)i, j));
            if (swapRB)
                std::swap(ch[0], ch[2]);
            split(image, ch);
        }
    }
    else
    {
        CV_Assert(nch == 1);
        int sz[] = { (int)nimages, 1, image0.rows, image0.cols };
        blob_.create(4, sz, ddepth);
        Mat blob = blob_.getMat();

        for (size_t i = 0; i < nimages; i++)
        {
            const Mat& image = images[i];
            CV_Assert(image.depth() == blob_.depth());
            nch = image.channels();
            CV_Assert(image.dims == 2 && (nch == 1));
            CV_Assert(image.size() == image0.size());

            image.copyTo(Mat(image.rows, image.cols, ddepth, blob.ptr((int)i, 0)));
        }
    }
}

void imagesFromBlob(const cv::Mat& blob_, OutputArrayOfArrays images_)
{
    CV_TRACE_FUNCTION();

    // A blob is a 4 dimensional matrix in floating point precision
    // blob_[0] = batchSize = nbOfImages
    // blob_[1] = nbOfChannels
    // blob_[2] = height
    // blob_[3] = width
    CV_Assert(blob_.depth() == CV_32F);
    CV_Assert(blob_.dims == 4);

    images_.create(cv::Size(1, blob_.size[0]), blob_.depth());

    std::vector<Mat> vectorOfChannels(blob_.size[1]);
    for (int n = 0; n < blob_.size[0]; ++n)
    {
        for (int c = 0; c < blob_.size[1]; ++c)
        {
            vectorOfChannels[c] = getPlane(blob_, n, c);
        }
        cv::merge(vectorOfChannels, images_.getMatRef(n));
    }
}


// dataconvert

// TODEL
void printblob(InputArray blob_) {
    Mat blob = blob_.getMat();
    auto shapeV = shape(blob);
//    CV_Assert(shapeV[0] == 1);
    auto typeMat = blob.type();

    std::cout << "data type = " << typeMat << std::endl;
    float *ptrf;
    uchar *ptru;
    char *ptrc;
    int len = std::min(int(blob.total()), 100);
    if (typeMat == 0) {
        ptru = (uchar *) blob.data;
        for (int i = 0; i < len; i++) {
            std::cout << (int) *(ptru + i) << ", ";
        }
    }else if (typeMat == 1) {
        ptrc = (char *)blob.data;
        for(int i = 0; i<len; i++) {
            std::cout<<(int) *(ptrc + i)<<", ";}

    }else if (typeMat == 5) {
        ptrf = (float *)blob.data;
        for(int i = 0; i<len; i++) {
            std::cout<<*(ptrf + i)<<", ";
        }
    }
    std::cout<<std::endl;
}

void shapePrint(InputArray blob_)
{
    Mat blob = blob_.getMat();
    auto shapeV = shape(blob);

    std::cout<<"Mat shape = ";
    for(auto & i : shapeV) {
        std::cout<<i<<" x ";
    }
    std::cout<<std::endl;
}

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))   // 这个是将一个数对齐4位。
#define ALIGN_UP4(x) ROUND_UP((x), 4)
#define ALIGN_UP8(x) ROUND_UP((x), 8)
void NHWC2NCHW(InputArray _src, OutputArray _dst)
{
    Mat src = _src.getMat();
    MatShape srcShape = shape(src);
    MatShape dstShape;
    dstShape.assign(srcShape.begin(), srcShape.end());

    // Move the channel to the second.
    for (int i = dstShape.size() - 1; 1 < i; i--)
    {
        std::swap(dstShape[i - 1], dstShape[i]);
    }

    _dst.create(src.dims, &dstShape[0], src.type());
    Mat dst = _dst.getMat();

//    CV_Assert(srcShape.size() == 4);
    int batchSize = srcShape[0];
    int channel = srcShape[srcShape.size() - 1];
    int depth = src.depth();
    int area = 0;

    if (srcShape.size() == 4)
    {
        area = srcShape[1] * srcShape[2];
    }
    else if (srcShape.size() == 5)
    {
        area = srcShape[1] * srcShape[2] * srcShape[3];
    }
    else if (srcShape.size() == 3)
    {
        area = srcShape[1];
    }

    // TODO! support int8_t
    CV_Assert(depth == CV_32F);
    float * inptr = src.ptr<float>();
    float * outptr = dst.ptr<float>();
    for (int bi = 0; bi < batchSize; bi++)
    {
        inptr = src.ptr<float>() + bi * channel * area;
        outptr = dst.ptr<float>() + bi * channel * area;

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int cur_area = 0; cur_area < area; cur_area++)
        {
            auto inptrD = inptr + channel * cur_area;
            auto outptrD = outptr + cur_area;
            for (int ci = 0; ci < channel; ci++)
            {
                outptrD[ci*area] = inptrD[ci];
            }
        }
    }
}

void NCHW2NHWC(InputArray _src, OutputArray _dst) {
    Mat src = _src.getMat();
    MatShape srcShape = shape(src);
    MatShape dstShape;
    dstShape.assign(srcShape.begin(), srcShape.end());
    // Move the channel to the last.
    // Move the channel to the second.
    for (int i = 1; i < dstShape.size() - 1; i++)
    {
        std::swap(dstShape[i], dstShape[i + 1]);
    }
    _dst.create(src.dims, &dstShape[0], src.type());
    Mat dst = _dst.getMat();

//    CV_Assert(srcShape.size() == 4);
    int batchSize = srcShape[0];
    int channel = srcShape[1];
    int depth = src.depth();
    int area = 0;

    if (srcShape.size() == 4)
    {
        area = srcShape[2] * srcShape[3];
    }
    else if (srcShape.size() == 5)
    {
        area = srcShape[2] * srcShape[3] * srcShape[4];
    }
    else if (srcShape.size() == 3)
    {
        area = srcShape[2];
    }

    // TODO! support int8_t
    CV_Assert(depth == CV_32F);
    float * inptr = src.ptr<float>();
    float * outptr = dst.ptr<float>();
    for (int bi = 0; bi < batchSize; bi++)
    {
        inptr = src.ptr<float>() + bi * channel * area;
        outptr = dst.ptr<float>() + bi * channel * area;

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int cur_area = 0; cur_area < area; cur_area++)
        {
            auto inptrD = inptr + cur_area;
            auto outptrD = outptr + channel * cur_area;
            for (int ci = 0; ci < channel; ci++)
            {
                outptrD[ci] = inptrD[ci*area];
            }
        }
    }


//    int sourceBatchsize = c * area;
//    int destBatchSize   = sourceBatchsize;
//    for (int bi = 0; bi < b; ++bi) {
//        auto srcBatch = source + bi * sourceBatchsize;
//        auto dstBatch = dest + bi * destBatchSize;
//        for (int i = 0; i < area; ++i) {
//            auto srcArea = srcBatch + i * c;
//            auto dstArea = dstBatch + i;
//            for (int ci = 0; ci < c; ++ci) {
//                dstArea[ci * area] = srcArea[ci];
//            }
//        }
//    }
}

// Default is that dst need be put in to the right Output.
void UnpackToNCHW(InputArray _src, OutputArray _dst, const int packElem = 4)
{
    Mat src = _src.getMat();
    MatShape srcShape = shape(src);
    CV_Assert(!_dst.empty());
    Mat dst = _dst.getMat();
    MatShape dstShape = shape(dst);

//    dstShape.assign(srcShape.begin(), srcShape.end() - 1);
//    dstShape[1] = srcShape[1] * src;
//    _dst.create(dstShape.size(), &dstShape[0], src.depth());
//    Mat dst = _dst.getMat();

    int batchSize = dstShape[0];
    int channel = dstShape[1];
    int depth = dst.depth();
    int area = 0;

    if (dstShape.size() == 4)
    {
        area = dstShape[2] * dstShape[3];
    }
    else if (dstShape.size() == 5)
    {
        area = dstShape[2] * dstShape[3] * dstShape[4];
    }
    else if (dstShape.size() == 3)
    {
        area = dstShape[2];
    }

    // TODO! only for NCHW
    int batchSrc = area * srcShape[1] * packElem;
    int batchDst = area * dstShape[1];

    // TODO! support int8_t
    CV_Assert(depth == CV_32F);
    float * inptr = src.ptr<float>();
    float * outptr = dst.ptr<float>();
    int ci, cur_area, idx;
//    int remain = channel % packElem;

    for (int bi = 0; bi < batchSize; bi++)
    {
        inptr = src.ptr<float>() + bi * batchSrc;
        outptr = dst.ptr<float>() + bi * batchDst;
//        idx = 0;

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (ci = 0;  ci < channel; ci++)
        {
            int idx = ci * area;
            int plane      = channel / packElem;
            auto srcPlane = inptr + plane * area * packElem;
            int offset     = ci % packElem;
            for (cur_area = 0; cur_area < area; ++cur_area)
            {
                outptr[idx++] = srcPlane[packElem * cur_area + offset];
            }
        }
    }
}

// If the channel % packElem != 0, we use zero padding. If the input shape is [N, C, H, W].
// The output shape should be [N, RoundUp(C, 4), H, W, 4].
void Pack4FromNCHW(InputArray _src, OutputArray _dst, const int packElem = 4)
{
    Mat src = _src.getMat();
    MatShape srcShape = shape(src);
    MatShape dstShape;
    dstShape.assign(srcShape.begin(), srcShape.end());
    dstShape.push_back(packElem);
    dstShape[1] = UP_DIV(dstShape[1], packElem);
    _dst.create(dstShape.size(), &dstShape[0], src.depth());
    Mat dst = _dst.getMat();

    int batchSize = srcShape[0];
    int channel = srcShape[1];
    int depth = src.depth();
    int area = 0;

    if (srcShape.size() == 4)
    {
        area = srcShape[2] * srcShape[3];
    }
    else if (srcShape.size() == 5)
    {
        area = srcShape[2] * srcShape[3] * srcShape[4];
    }
    else if (srcShape.size() == 3)
    {
        area = srcShape[2];
    }

    int batchSrc = area * srcShape[1];
    int batchDst = area * dstShape[1] * packElem;
    // TODO! support int8_t
    CV_Assert(depth == CV_32F);
    float * inptr = src.ptr<float>();
    float * outptr = dst.ptr<float>();
    int remain = channel % packElem;
    int ci, cur_area, idx;
    for (int bi = 0; bi < batchSize; bi++)
    {
        inptr = src.ptr<float>() + bi * batchSrc;
        outptr = dst.ptr<float>() + bi * batchDst;
//        idx = 0;

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (ci = 0;  ci < channel; ci++)
        {
            int idx = ci * area;
            int plane = ci / packElem;
            int offset = ci % packElem;
            auto dstPlane = outptr + plane * area * packElem;

            for (cur_area = 0; cur_area < area; ++cur_area)
            {
                dstPlane[packElem * cur_area + offset] = inptr[idx++];
            }
        }

        // zero padding for remain channel.
        if (remain > 0)
        {
            for (ci = channel; ci < ROUND_UP(channel, packElem); ci ++)
            {
                int plane = ci / packElem;
                int offset= ci % packElem;
                auto dstPlane = outptr + plane * area * packElem;

                for (cur_area = 0; cur_area < area; ++cur_area)
                {
                    dstPlane[packElem * cur_area + offset] = 0;
                }
            }
        }
    }
}

// For Mat
void dataLayoutConvert(InputArray _src, OutputArray _dst, DataLayout inpLayout, DataLayout outLayout)
{
    // TODO: support data type: float 16, int 8.
    // get input data depth.
    Mat src = _src.getMat();

    if (inpLayout == DNN_DATALAYOUT_NCHW && outLayout == DNN_DATALAYOUT_NHWC)
    {
        NCHW2NHWC(_src, _dst);
    }
    else if (inpLayout == DNN_DATALAYOUT_NHWC && outLayout == DNN_DATALAYOUT_NCHW)
    {
        NHWC2NCHW(_src, _dst);
    }

    // Unpack the Blob
    if (inpLayout == DNN_DATALAYOUT_NC4HW4)
    {
        if (outLayout == DNN_DATALAYOUT_NCHW)
        {
            UnpackToNCHW(_src, _dst, 4);
        }
        else // TODO! outLayout = DNN_DATALAYOUT_NHWC
            CV_Error(Error::StsNotImplemented, "Not Implemet DataLayOut");
    }

    // Pack the Blob
    if (outLayout == DNN_DATALAYOUT_NC4HW4)
    {
        if (inpLayout == DNN_DATALAYOUT_NCHW)
        {
            Pack4FromNCHW(_src, _dst, 4);
        }
        else // inpLayout = DNN_DATALAYOUT_NHWC
            CV_Error(Error::StsNotImplemented, "Not Implemet DataLayOut");
    }
}




CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
