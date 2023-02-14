// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/buffer.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

// 这串代码是从vk官方教程直接抄的。
static uint32_t findMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memoryProperties;

    // 查询到当前设备支持到内存属性。
    vkGetPhysicalDeviceMemoryProperties(kPhysicalDevice, &memoryProperties);

    // 从中找出满足要求到内存属性。
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        if ((memoryTypeBits & (1 << i)) &&
                ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
            return i;
    }
    return -1;
}

bool Buffer::init(size_t size_in_bytes, const char* data)
{
    if (buffer_ != VK_NULL_HANDLE)
    {
        printf("Warn: Buffer object already inited\n");
        return false;
    }

    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = size_in_bytes;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK_RESULT(vkCreateBuffer(device_, &bufferCreateInfo, NULL, &buffer_));

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device_, buffer_, &memoryRequirements);

    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(device_, &allocateInfo, NULL, &memory_));

    if (data)
    {
        // 这个操作是首先将数据复制到CPU上到一块内存，然后映射到GPU的内存上，最后解除映射，这样GPU上到buffer就有了正确到内存数据。
        char* dst;
        VK_CHECK_RESULT(vkMapMemory(device_, memory_, 0, size_in_bytes, 0, (void **)&dst));
        memcpy(dst, data, size_in_bytes);
        vkUnmapMemory(device_, memory_);
    }

    // 关联buffer和memory，分配到内存，和缓冲对象。
    VK_CHECK_RESULT(vkBindBufferMemory(device_, buffer_, memory_, 0));
    return true;
}

Buffer::Buffer(VkDevice& device, size_t size_in_bytes, const char* data)
{
    device_ = device;
    buffer_ = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
    init(size_in_bytes, data);
}

Buffer::~Buffer()
{
    vkFreeMemory(device_, memory_, NULL);
    vkDestroyBuffer(device_, buffer_, NULL);
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
