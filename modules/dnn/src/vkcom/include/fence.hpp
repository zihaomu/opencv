//
// Created by Z Moo on 2023/2/22.
//

#ifndef OPENCV_FENCE_HPP
#define OPENCV_FENCE_HPP

#include "../../precomp.hpp"
#include <vulkan/vulkan.h>

namespace cv { namespace dnn { namespace vkcom {

// Used for synchronize and wait
class Fence
{
public:
    Fence();
    ~Fence();

    VkFence get() const;
    VkResult reset() const;
    VkResult wait() const;

private:
    VkFence fence;
};

}}} // namespace cv::dnn::vkcom

#endif //OPENCV_FENCE_HPP
