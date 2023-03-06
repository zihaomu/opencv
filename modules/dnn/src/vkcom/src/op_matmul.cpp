//
// Created by Z Moo on 2023/3/1.

#include "../../precomp.hpp"
#include "common.hpp"
//#include "internal.hpp"
#include "../include/op_matmul.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

#define KSTRIP_LEN 8
#define BLOCK_SIZE 32

#define MAX_COMPUTE_GFLOPS 10
// TODO: query group count from vulkan device
#define MAX_GROUP_COUNT_X 65535
#define MAX_GROUP_COUNT_Y 65535
#define MAX_GROUP_COUNT_Z 65535
#define VEC_LEN 4

OpMatMul::OpMatMul(std::vector<Mat>& matBlobs, const int _M, const int _K, const int _N) : M(_M), K(_K), N(_N)
{
    // Convert Weight to GPU Tensor.
    type_ = kOpTypeMatMul;
    CV_Assert(matBlobs.empty() || matBlobs.size() == 1);

    if (matBlobs.size() == 1)
    {
        Tensor weightTensor;
        CV_Assert(matBlobs[0].isContinuous() && matBlobs[0].type() == CV_32F);
        std::vector<int> matShape = shape(matBlobs[0]);
        weightTensor.reshape((const char*)matBlobs[0].data, matShape); // This code will copy the src data from Mat to VkBuffer.

        weightTensorPtr = makePtr<Tensor>(weightTensor);
    }
}

bool OpMatMul::forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs)
{
    Layer::t0.start();

    CV_Assert((ins.size() == 1 || ins.size() == 2) && outs.size() == 1);
    Shape inputShape = ins[0].getShape();
    Shape outputShape = outs[0].getShape();
    CV_Assert(inputShape.size() == outputShape.size());

    batch = inputShape[kShapeIdxBatch];
    Hi = inputShape[kShapeIdxHeight];
    Wi = inputShape[kShapeIdxWidth];

    H0 = outputShape[kShapeIdxHeight];
    W0 = outputShape[kShapeIdxWidth];

    config.local_size_x = BLOCK_SIZE;
    config.local_size_y = BLOCK_SIZE;
    config.local_size_z = 1;

    int KStrip = K/BLOCK_SIZE;
    int KStripRemain = K - KStrip * BLOCK_SIZE;
    computeGroupCount();
    std::vector<int> param = {M, K, N, KStrip, KStripRemain};

    std::vector<int> shape = {(int)param.size()};
    Tensor paramTensor = Tensor(reinterpret_cast<const char *>(param.data()), shape, kFormatInt32, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    std::string key = "gemm_v7_spv";
    destTypes = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // input
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // weight
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // out
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  // param
    };
    Layer::t0.stop();

    Layer::t1.start();
    Ptr<Pipeline> pipeline = pipelineFactoryPtr->getPipeline(key, destTypes);
    Ptr<Descriptor> desSet = pipeline->createSet();
    Ptr<CommandBuffer> cmdBuffer = cmdPoolPtr->allocBuffer();

    VkCommandBuffer cmdBufferReal = cmdBuffer->get();
    desSet->writeTensor(ins[0], 0);

    if (weightTensorPtr)
        desSet->writeTensor(*weightTensorPtr, 1);
    else
    {
        CV_Assert(ins.size() == 2);
        desSet->writeTensor(ins[1], 1);
    }

    desSet->writeTensor(outs[0], 2);
    desSet->writeTensor(paramTensor, 3); // TODO change the parameter from pushconstance to buffer.

    cmdBuffer->beginRecord();
    pipeline->bind(cmdBufferReal, desSet->get());
    vkCmdDispatch(cmdBufferReal, group_x_, group_y_, group_z_);
    cmdBuffer->endRecord();

    cmdPoolPtr->submitAndWait(cmdBufferReal);
    Layer::t1.stop();

    return true;
}


bool OpMatMul::computeGroupCount()
{
    group_x_ = alignSize(M, BLOCK_SIZE) / BLOCK_SIZE;
    group_y_ = alignSize(N, BLOCK_SIZE) / BLOCK_SIZE;
    group_z_ = 1;
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom