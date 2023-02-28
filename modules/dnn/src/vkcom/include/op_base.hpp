// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_VKCOM_OP_BASE_HPP
#define OPENCV_DNN_VKCOM_OP_BASE_HPP

#include "../../precomp.hpp"
#include "vkcom.hpp"
#include "context_vulkan.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

// Forward declare
class Context;

class OpBase
{
public:
    OpBase();
    virtual ~OpBase();
    virtual bool forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs) = 0;

protected:
    std::vector<VkCommandBuffer> kCmdBuffers;
    std::vector<VkDescriptorType> destTypes;
    std::string shader_name; // the key which is used for retrieve Pipeline from PipelineFactory.
    std::string type_;
    int group_x_;
    int group_y_;
    int group_z_;
};

//class OpBase
//{
//public:
//    OpBase();
//    virtual ~OpBase();
//    virtual bool forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs) = 0;
//protected:
//    void initVulkanThing(int buffer_num);
//    void createDescriptorSetLayout(int buffer_num);
//    void createDescriptorSet(int buffer_num);
//    void createShaderModule(const uint32_t* spv, size_t sz, const std::string& source = std::string());
//    void createPipeline(size_t push_constants_size = 0, VkSpecializationInfo* specialization_info = 0);
//    void createCommandBuffer();
//    void recordCommandBuffer(void* push_constants = NULL, size_t push_constants_size = 0);
//    void runCommandBuffer();
//
//    std::vector<VkDescriptorType> destTypes;
//    std::string shader_name;
//    std::string type_;
//    int group_x_;
//    int group_y_;
//    int group_z_;
//};

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom

#endif // OPENCV_DNN_VKCOM_OP_BASE_HPP
