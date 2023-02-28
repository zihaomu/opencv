// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "common.hpp"
//#include "internal.hpp"
#include "../include/op_conv.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

#define DEFAULT_LOCAL_SZ 128
#define MAX_COMPUTE_GFLOPS 10
// TODO: query group count from vulkan device
#define MAX_GROUP_COUNT_X 65535
#define MAX_GROUP_COUNT_Y 65535
#define MAX_GROUP_COUNT_Z 65535
#define VEC_LEN 4
#define GPU_WORK_GROUP_SIZE 32

struct ShaderConstant {
    int lsz_x;
    int lsz_y;
    int lsz_z;
};

struct ShaderParam {
    int in_h;
    int in_w;
    int out_h;
    int out_w;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    int filter_h;
    int filter_w;
    int dilation_h;
    int dilation_w;
    int channels;
    int batch;
    int has_bias;
    int M;
    int K;
    int N;
    int basic_shader_batch_idx;
    int basic_shader_partition_idx;
    int basic_shader_partition_size;
};

struct ShaderParam4x4 {
    int out_h;
    int out_w;
    int CgHkWk;
    int Kg;
    int group;
    int batch;
};

// TODO add check for fuse activ!
// TODO change the tensorBlobs as Mat, and add the weight Transform at the constructor func.
OpConv::OpConv(std::vector<Mat>& matBlobs, int _activType, const int _ngroups, const int _K,
               const int _C, const int _Hk, const int _Wk, const int _stride_h, const int _stride_w,
               const int _dilation_h, const int _dilation_w, const int _pad_left, const int _pad_top, bool _fusedAdd):
               activ((FusedActivationType)_activType), ngroups(_ngroups), K(_K), C(_C), Hk(_Hk), Wk(_Wk), stride_h(_stride_h), stride_w(_stride_w),
               dilation_h(_dilation_h), dilation_w(_dilation_w), pad_left(_pad_left), pad_top(_pad_top), fusedAdd(_fusedAdd)
{
    bool testMode = false;

    type_ = kOpTypeConv;

    Kg = K/ngroups, Cg = max(C/ngroups, 1);
    ksize = Hk * Wk;
    CgHkWk = Cg * ksize;

    if (testMode)
    {
        shaderType = kConvShaderTest;
    }
    else if (ngroups > 1 && ngroups == K && ngroups == C)
    {
        shaderType = kConvShaderTypeDepthWise;
    }
    else
    {
        shaderType = kConvShaderTypeGeneric;
    }

    CV_Assert(matBlobs.size() >= 1);

    // TODO add 1x4 and 4x1 branch for some extreme case.
    // repack the weight. The shape is from [K, C, Hk, Wk] to [ngroups, Ceil(K/group), (Cg*Hk*Wk), VEC_LEN]
    // 4x4, brnach
    if (0 && shaderType == kConvShaderTypeGeneric)
    {
        int numStripsKg = (Kg + VEC_LEN - 1) / VEC_LEN;
        int Kg_aligned = numStripsKg * VEC_LEN;
        std::vector<int> repackWeightShape = {ngroups, numStripsKg, CgHkWk, VEC_LEN};

        Mat repackWeight = Mat(repackWeightShape, CV_32FC1, Scalar_<float>(0.0f));
        float* weightsBufPtr = repackWeight.ptr<float>();
        const float* srcWeight = matBlobs[0].ptr<float>();
        const size_t wstep = matBlobs[0].step1();
        
        // Pack the weight.
        parallel_for_(Range(0, ngroups * numStripsKg), [&](const Range& r0){
        for (int gsi = r0.start; gsi < r0.end; gsi++)
        {
            int g = gsi / numStripsKg;
            int si = gsi - g * numStripsKg;

            int startK = si * VEC_LEN;
            CV_Assert(startK < Kg_aligned);

            float* packed_wptr = weightsBufPtr + CgHkWk * (startK + g * Kg_aligned);
            int dk = Kg - startK < VEC_LEN ? Kg - startK : VEC_LEN;

            int k_idx = g*Kg + startK;

            for(int c = 0; c < Cg; c++)
            {
                for(int hwd = 0; hwd < Hk*Wk; hwd++, packed_wptr += VEC_LEN)
                {
                    const float* wptr = srcWeight + wstep * k_idx + c*Hk*Wk + hwd;
                    int k = 0;
                    for(; k < dk; k++, wptr += wstep)
                        packed_wptr[k] = *wptr;
                    for(; k < VEC_LEN; k++)
                        packed_wptr[k] = 0.f;
                }
            }
        }});

        // Create weightTensor
        Tensor weightTensor;
        CV_Assert(repackWeight.isContinuous() && matBlobs[0].type() == CV_32F);
        weightTensor.reshape((const char*)repackWeight.data, repackWeightShape);
        weightTensorPtr = makePtr<Tensor>(weightTensor);
    }
    else
    {
        // Create weightTensor
        Tensor weightTensor;
        CV_Assert(matBlobs[0].isContinuous() && matBlobs[0].type() == CV_32F);
        std::vector<int> matShape = shape(matBlobs[0]);
        weightTensor.reshape((const char*)matBlobs[0].data, matShape); // This code will copy the src data from Mat to VkBuffer.

        weightTensorPtr = makePtr<Tensor>(weightTensor);
    }

    if (matBlobs.size() == 2)
    {
        Tensor biasTensor;
        CV_Assert(matBlobs[1].isContinuous() && matBlobs[1].type() == CV_32F);
        std::vector<int> matShape = shape(matBlobs[1]);
        biasTensor.reshape((const char*)matBlobs[1].data, matShape); // This code will copy the src data from Mat to VkBuffer.

        biasTensorPtr = makePtr<Tensor>(biasTensor);
    }
    // TODO set the 0 as the bias Tensor, so that we can remove the if branch in the kernel
    else
    {
        std::vector<int> shape = {K};
        Tensor bias(0, shape);
        biasTensorPtr = makePtr<Tensor>(bias);
    }
    
//#define BUFFER_NUM 4
//OpBase::initVulkanThing(BUFFER_NUM);
//    OpBase::initVulkanThing(4);
}

void setXYZ(unsigned int* dstcode, const unsigned int* code, const size_t size,
            uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z, size_t* dstsize)
{
    uint32_t local_size_x_id = -1;
    uint32_t local_size_y_id = -1;
    uint32_t local_size_z_id = -1;
    uint32_t gl_WorkGroupSize_id = -1;

    const uint32_t* p = code;
    uint32_t* dp = dstcode;

    // skip magic version generator bound schema
    memcpy(dp, p, 5 * sizeof(uint32_t));
    p += 5;
    dp += 5;

    // foreach op
    while ((const unsigned char*)p < (const unsigned char*)code + size)
    {
        uint32_t opcode = p[0];

        uint16_t wordcount = opcode >> 16;
        uint16_t op = opcode & 0xffff;

        if (op == 16) // OpExecutionMode
        {
            uint32_t mode = p[2];
            if (mode == 17) // LocalSize
            {
                memcpy(dp, p, wordcount * sizeof(uint32_t));

                // set local_size_xyz
                // 替换掉原始的 xyz
                dp[3] = local_size_x;
                dp[4] = local_size_y;
                dp[5] = local_size_z;

                p += wordcount;
                dp += wordcount;
                continue;
            }
        }
        else if (op == 50) // OpSpecConstant
        {
            uint32_t id = p[2];
            if (id == local_size_x_id || id == local_size_y_id || id == local_size_z_id)
            {
                p += wordcount;
                continue;
            }
        }
        else if (op == 51) // OpSpecConstantComposite
        {
            uint32_t id = p[2];
            if (id == gl_WorkGroupSize_id)
            {
                if (wordcount == 6 && (p[3] == local_size_x_id || p[4] == local_size_y_id || p[5] == local_size_z_id))
                {
                    p += wordcount;
                    continue;
                }
            }
        }
        else if (op == 71) // OpDecorate
        {
            uint32_t id = p[1];
            uint32_t decoration = p[2];
            if (decoration == 1) // SpecId
            {
                uint32_t specid = p[3];
                if (specid == 233) local_size_x_id = id;
                if (specid == 234) local_size_y_id = id;
                if (specid == 235) local_size_z_id = id;
                if (specid == 233 || specid == 234 || specid == 235)
                {
                    p += wordcount;
                    continue;
                }
            }
            else if (decoration == 11) // BuiltIn
            {
                uint32_t builtin = p[3];
                if (builtin == 25) // WorkgroupSize
                {
                    gl_WorkGroupSize_id = id;
                    p += wordcount;
                    continue;
                }
            }
        }

        memcpy(dp, p, wordcount * sizeof(uint32_t));
        p += wordcount;
        dp += wordcount;
    }

    *dstsize = (unsigned char*)dp - (unsigned char*)dstcode;
}

void printTensor(vkcom::Tensor &src)
{
    std::vector<int> shape = src.getShape();
    void *data = src.map();
    Mat tmp(shape, CV_32F, data);

    std::cout<<"print data from tensor!"<<std::endl;
    printblob(tmp);
    src.unMap();
}

bool OpConv::forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs)
{
    Layer::t0.start();
    CV_Assert(ins.size() == 1 && outs.size() == 1);
    Shape inputShape = ins[0].getShape();
    Shape outputShape = outs[0].getShape();
    CV_Assert(inputShape.size() == outputShape.size());

    batch = inputShape[kShapeIdxBatch];
    Hi = inputShape[kShapeIdxHeight];
    Wi = inputShape[kShapeIdxWidth];
    const int inPlanesize = Hi*Wi;

    H0 = outputShape[kShapeIdxHeight];
    W0 = outputShape[kShapeIdxWidth];
    const int outPlanesize = H0*W0;

    // ------------ repack input
    // one conv type, one shader type.
    // Step 1: repack the input data.
    // A new Op for data repack.
    // repacked data shape is [ngroups, ceil(H0*W0/VEL_LEN), Cg*Hk*Wk, VEC_LEN]
    int hwStripe = (H0*W0 + VEC_LEN - 1) / VEC_LEN;

    config.local_size_x = DEFAULT_LOCAL_SZ;
    config.local_size_y = 1;
    config.local_size_z = 1;

//    std::vector<int> param = {0, 1, 2, 3, 4, 5, 6, 7, 8};
//    std::vector<int> shape = {sizeof(param)};
//    Tensor paramTensor = Tensor(reinterpret_cast<const char *>(param.data()), shape, kFormatInt32, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
//    computeGroupCount();
//
//    std::string key = "moo_test2_spv";
//    destTypes = {
//            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // input
//            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
//    };
//
//    Ptr<Pipeline> pipeline = pipelineFactoryPtr->getPipeline(key, destTypes);
//    Ptr<Descriptor> desSet = pipeline->createSet();
//    Ptr<CommandBuffer> cmdBuffer = cmdPoolPtr->allocBuffer();
//
//    VkCommandBuffer cmdBufferReal = cmdBuffer->get();
//    desSet->writeTensor(outs[0], 0);
//    desSet->writeTensor(paramTensor, 1); // TODO change the parameter from pushconstance to buffer.
//
//    cmdBuffer->beginRecord();
//    pipeline->bind(cmdBufferReal, desSet->get());
//    vkCmdDispatch(cmdBufferReal, group_x_, group_y_, group_z_);
//    cmdBuffer->endRecord();
//
//    cmdPoolPtr->submitAndWait(cmdBufferReal);

    computeGroupCount();
    std::vector<int> param = {Hi, Wi,
                              H0, W0,
                              stride_h, stride_w,
                              pad_top, pad_left,
                              Hk, Wk,
                              dilation_h, dilation_w,
                              C, batch, true,
                              outPlanesize, CgHkWk, K, 0, 0, 0};

    std::vector<int> shape = {(int)param.size()};
    Tensor paramTensor = Tensor(reinterpret_cast<const char *>(param.data()), shape, kFormatInt32, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    std::string key = "conv_spv";
    destTypes = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // input
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // output
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // weight
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // bias
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
    };
    Layer::t0.stop();

    Layer::t1.start();
    Ptr<Pipeline> pipeline = pipelineFactoryPtr->getPipeline(key, destTypes);
    Ptr<Descriptor> desSet = pipeline->createSet();
    Ptr<CommandBuffer> cmdBuffer = cmdPoolPtr->allocBuffer();

    VkCommandBuffer cmdBufferReal = cmdBuffer->get();
    desSet->writeTensor(ins[0], 0);
    desSet->writeTensor(*biasTensorPtr, 1);
    desSet->writeTensor(*weightTensorPtr, 2);
    desSet->writeTensor(outs[0], 3);
    desSet->writeTensor(paramTensor, 4); // TODO change the parameter from pushconstance to buffer.

    cmdBuffer->beginRecord();
    pipeline->bind(cmdBufferReal, desSet->get());
    vkCmdDispatch(cmdBufferReal, group_x_, group_y_, group_z_);
    cmdBuffer->endRecord();

    cmdPoolPtr->submitAndWait(cmdBufferReal);
    Layer::t1.stop();
//    printTensor(outs[0]);
    return true;
}

bool OpConv::computeGroupCount()
{
    int outplan = H0 * W0;
    if (shaderType == kConvShaderTypeDepthWise)
    {
        group_x_ = alignSize(W0, config.local_size_x) / config.local_size_x;
        group_y_ = alignSize(H0, config.local_size_y) / config.local_size_y;
        group_z_ = alignSize(C, config.local_size_z) / config.local_size_z;
        return true;
    }
    else if (shaderType == kConvShaderTypeGeneric)
    {
//        int hwStripe = (H0*W0 + VEC_LEN - 1) / VEC_LEN;
//        group_x_ = alignSize(hwStripe * alignSize(Kg, VEC_LEN)/VEC_LEN, config.local_size_x) / config.local_size_x;
//        group_y_ = batch * ngroups;
//        group_z_ = 1;

//        group_x_ = 1;
//        group_y_ = 1;
//        group_z_ = 1;

        group_x_ = alignSize(outplan, config.local_size_x) / config.local_size_x;
        CV_Assert(config.local_size_y == 1);
        float GFLOPS = (2.0 * C *Hk*Wk + 1) *
                       (K * H0*W0) / 1000 / 1000 / 1000;
        group_y_ = std::min(MAX_GROUP_COUNT_Y, (int)floor(MAX_COMPUTE_GFLOPS / (GFLOPS / K)));
        group_z_ = 1;

    }
    else if (shaderType == kConvShaderTest)
    {
        group_x_ = 1;
        group_y_ = 1;
        group_z_ = 1;
    }
    else
        CV_Error(CV_StsNotImplemented, "shader type is not supported at compute GroupCount.");

    CV_Assert(group_x_ <= MAX_GROUP_COUNT_X);
    CV_Assert(group_y_ <= MAX_GROUP_COUNT_Y);
    CV_Assert(group_z_ <= MAX_GROUP_COUNT_Z);

    return true;
}

//struct RepackShaderConstant {
//        int in_h;
//        int in_w;
//        int out_h;
//        int out_w;
//        int group;
//        int batch;
//        int kernel_h;
//        int kernel_w;
//        int stride_h;
//        int stride_w;
//        int pad_top;
//        int pad_left;
//        int dilation_h;
//        int dilation_w;
//        int Cg;
//};
//
//SubOpConvRepackInput::SubOpConvRepackInput(const int _ngroups, const int _batch, const int _Hi, const int _Wi,
//                                           const int _H0, const int _W0, const int _C, const int _Hk,
//                                           const int _Wk, const int _stride_h, const int _stride_w,
//                                           const int _dilation_h, const int _dilation_w, const int _pad_left,
//                                           const int _pad_top) :
//                                           ngroups(_ngroups), batch(_batch), Hi(_Hi), Wi(_Wi), H0(_H0), W0(_W0), C(_C),
//                                           Hk(_Hk), Wk(_Wk), stride_h(_stride_h), stride_w(_stride_w), dilation_h(_dilation_h),
//                                           dilation_w(_dilation_w), pad_left(_pad_left), pad_top(_pad_top)
//{
//    type_ = kOpTypeConvInternel;
//    Cg = max(C/ngroups, 1);
//    OpBase::initVulkanThing(2);
//}
//
//bool SubOpConvRepackInput::forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs)
//{
//    if (pipeline_ == VK_NULL_HANDLE)
//    {
//        config.local_size_x = GPU_WORK_GROUP_SIZE;
//        config.local_size_y = 1;
//        config.local_size_z = 1;
//
//        size_t dataLen = sizeof(conv_in_repack_spv);
//        unsigned int* conv_in_repack_spv_modified = (unsigned int*)malloc(dataLen);
//
//        setXYZ(conv_in_repack_spv_modified, conv_in_repack_spv, dataLen, config.local_size_x, config.local_size_y, config.local_size_z, &dataLen);
//        createShaderModule(conv_in_repack_spv_modified, dataLen); // TODO: move this step out of the forward stage.
//        int pSize = sizeof(RepackShaderConstant);
//        createPipeline(pSize);
//    }
//
//    // TODO finish the repack shader code.
//    bindTensor(device_, ins[0], 0, descriptor_set_);
//    bindTensor(device_, outs[0], 1, descriptor_set_);
//
//    computeGroupCount();
//    RepackShaderConstant param = {
//            Hi, Wi, H0, W0, ngroups, batch, Hk, Wk, stride_h, stride_w,
//            pad_top, pad_left, dilation_h, dilation_w, Cg
//    };
//
//    recordCommandBuffer((void *)&param, sizeof(RepackShaderConstant));
//    runCommandBuffer();
//}
//
//// TODO find the best way to run the input repack.
//bool SubOpConvRepackInput::computeGroupCount()
//{
//    int hwStripe = (H0*W0 + VEC_LEN - 1) / VEC_LEN;
//    group_x_ = alignSize(hwStripe * Cg, config.local_size_x) / config.local_size_x;
//    group_y_ = batch * ngroups;
//    group_z_ = 1;
//
//    std::cout<<"group_x_ = "<<group_x_<<",group_y_ = "<<group_y_<<", group_z_="<<group_z_<<std::endl;
//
//    CV_Assert(group_x_ <= MAX_GROUP_COUNT_X);
//    CV_Assert(group_y_ <= MAX_GROUP_COUNT_Y);
//    CV_Assert(group_z_ <= MAX_GROUP_COUNT_Z);
//
//    return true;
//}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
