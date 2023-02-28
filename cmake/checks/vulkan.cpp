#include <iostream>
#include <vulkan/vulkan.h>

int main(int /*argc*/, char** /*argv*/)
{
    VkInstance instance;
    VkInstanceCreateInfo createInfo = {};
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
    {
        std::cout<<"failed to create instance!"<<std::endl;
        return 0;
    }
    else
    {
        std::cout<<"create instance successfully!"<<std::endl;
        return 1;
    }
}
