// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_VKCOM_VULKAN_VK_LOADER_HPP
#define OPENCV_DNN_VKCOM_VULKAN_VK_LOADER_HPP

#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif // HAVE_VULKAN

#if defined(_WIN32)
#include <windows.h>
typedef HMODULE VulkanHandle;
#define DEFAULT_VK_LIBRARY_PATH "vulkan-1.dll"
#define LOAD_VK_LIBRARY(path) LoadLibrary(path)
#define FREE_VK_LIBRARY(handle) FreeLibrary(handle)
#define GET_VK_ENTRY_POINT(handle) \
        (PFN_vkGetInstanceProcAddr)GetProcAddress(handle, "vkGetInstanceProcAddr");
#endif // _WIN32

#if defined(__linux__)
#include <dlfcn.h>
#include <stdio.h>
typedef void* VulkanHandle;
#define DEFAULT_VK_LIBRARY_PATH "libvulkan.so"
#define LOAD_VK_LIBRARY(path) dlopen(path, RTLD_LAZY | RTLD_GLOBAL)
#define FREE_VK_LIBRARY(handle) dlclose(handle)
#define GET_VK_ENTRY_POINT(handle) \
        (PFN_vkGetInstanceProcAddr)dlsym(handle, "vkGetInstanceProcAddr");
#endif // __linux__

#if defined(__APPLE__)
#include <dlfcn.h>
#include <stdio.h>
typedef void* VulkanHandle;

#if defined(__x86_64__)
#define DEFAULT_VK_LIBRARY_PATH "libvulkan.dylib"
#else // For Apple ARM chip, we use MoltenVK lib.
#define DEFAULT_VK_LIBRARY_PATH "libMoltenVK.dylib"
#endif

#define LOAD_VK_LIBRARY(path) dlopen(path, RTLD_LAZY | RTLD_GLOBAL)
#define FREE_VK_LIBRARY(handle) dlclose(handle)
#define GET_VK_ENTRY_POINT(handle) \
        (PFN_vkGetInstanceProcAddr)dlsym(handle, "vkGetInstanceProcAddr");
#endif // Macos

#ifndef DEFAULT_VK_LIBRARY_PATH
#define DEFAULT_VK_LIBRARY_PATH ""
#define LOAD_VK_LIBRARY(path) nullptr
#define FREE_VK_LIBRARY(handle)
#define GET_VK_ENTRY_POINT(handle) nullptr
#endif

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN
bool loadVulkanLibrary(VulkanHandle& handle);
bool loadVulkanEntry(VulkanHandle& handle);
bool loadVulkanGlobalFunctions();
bool loadVulkanFunctions(VkInstance& instance);
#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
#endif // OPENCV_DNN_VKCOM_VULKAN_VK_LOADER_HPP
