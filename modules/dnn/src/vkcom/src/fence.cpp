//
// Created by Z Moo on 2023/2/22.
//

#include "../../precomp.hpp"
#include "common.hpp"
#include "../include/fence.hpp"

namespace cv { namespace dnn { namespace vkcom {

Fence::Fence()
{
    // TODO check if the following code is needed.
//#ifdef VK_USE_PLATFORM_WIN32_KHR
//    // which one is correct on windows ?
//    VkExportFenceCreateInfoKHR efci;
//    // VkExportFenceWin32HandleInfoKHR efci;
//    efci.sType = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO;
//    efci.pNext = NULL;
//    efci.sType = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
//#else
//    VkExportFenceCreateInfoKHR efci;
//    efci.sType       = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO;
//    efci.pNext       = NULL;
//#if VK_USE_PLATFORM_ANDROID_KHR // current android only support VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT
//    efci.handleTypes = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT;
//#else
//    efci.handleTypes = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT;
//#endif
//#endif
    VkFenceCreateInfo fci{
            /* .sType = */ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            /* .pNext = */ nullptr,
            /* .flags = */ 0,
    };
    vkCreateFence(kDevice, &fci, nullptr, &fence);
}

VkFence Fence::get() const
{
    return fence;
}

VkResult Fence::reset() const
{
    return vkResetFences(kDevice, 1, &fence);
}

VkResult Fence::wait() const
{
    auto status = VK_TIMEOUT;

    do {
        status = vkWaitForFences(kDevice, 1, &fence, VK_TRUE, 5000000000);
    } while (status == VK_TIMEOUT);

    return status;
}

Fence::~Fence()
{
    vkDestroyFence(kDevice, fence, nullptr);
}

}}} // namespace cv::dnn::vkcom