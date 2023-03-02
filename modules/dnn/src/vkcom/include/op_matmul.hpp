//
// Created by Z Moo on 2023/3/1.
//

#ifndef OPENCV_OP_MATMUL_HPP
#define OPENCV_OP_MATMUL_HPP

#include "vkcom.hpp"
#include "op_base.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

struct MatMulShaderConfig
{
    int local_size_x;
    int local_size_y;
    int local_size_z;
    int block_height;
    int block_width;
    int block_depth;
};

// Current Vulkan Convolution layer only support Conv2D.
class OpMatMul : public OpBase
{
public:
    OpMatMul(std::vector<Mat>& matBlobs, const int M, const int K, const int N);

    virtual bool forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs) CV_OVERRIDE;
    Ptr<Tensor> weightTensorPtr;
private:
    bool computeGroupCount();

    FusedActivationType activ;
    const int M, K, N;

    int Hi, Wi;
    int H0, W0;
    int batch;

    MatMulShaderConfig config;
};

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
#endif //OPENCV_OP_MATMUL_HPP
