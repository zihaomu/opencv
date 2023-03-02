#set(VULKAN_INCLUDE_DIRS "${OpenCV_SOURCE_DIR}/3rdparty/include" CACHE PATH "Vulkan include directory")
#set(VULKAN_LIBRARIES "")

# Ubuntu
#set(VULKAN_INCLUDE_DIRS "/home/moo/vulkan_sdk/1.3.236.0/x86_64/include" CACHE PATH "Vulkan include directory")
#set(VULKAN_LIBRARIES_DIRS "/home/moo/vulkan_sdk/1.3.236.0/x86_64/lib" CACHE PATH "Path to Vulkan Libraries.")

# MacOS
set(VULKAN_INCLUDE_DIRS "/Users/zihao/VulkanSDK/1.3.231.1/MoltenVK/include" CACHE PATH "Vulkan include directory")
set(VULKAN_LIBRARIES_DIRS "/Users/zihao/VulkanSDK/1.3.231.1/MoltenVK/dylib/macOS" CACHE PATH "Path to Vulkan Libraries.")

# For Apple ARM, we use MoltenVK to call vulkan library.
if(CMAKE_HOST_SYSTEM_NAME MATCHES Darwin AND CMAKE_HOST_SYSTEM_PROCESSOR MATCHES arm64)
  # TODEL
  message("Find lib at MoltenVK")
  find_library(VULKAN_LIBRARIES MoltenVK PATHS ${VULKAN_LIBRARIES_DIRS} NO_DEFAULT_PATH)
else()
  # TODEL
  message("Find lib at Vulkan")
  find_library(VULKAN_LIBRARIES vulkan PATHS ${VULKAN_LIBRARIES_DIRS} NO_DEFAULT_PATH)
endif()

if(NOT VULKAN_LIBRARIES)
  message("Please Set right VULKAN_LIBRARIES_DIRS")
  return()
else()
  message("inVK cmake VULKAN_LIBRARIES Path = ${VULKAN_LIBRARIES}")
endif()

#try_compile(VALID_VULKAN
#      "${OpenCV_BINARY_DIR}"
#      "${OpenCV_SOURCE_DIR}/cmake/checks/vulkan.cpp"
#      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${VULKAN_INCLUDE_DIRS}"
#      OUTPUT_VARIABLE TRY_OUT
#      )

try_run(VALID_VULKAN_RUN VALID_VULKAN_COMPILE
        "${OpenCV_BINARY_DIR}"
        "${OpenCV_SOURCE_DIR}/cmake/checks/vulkan.cpp"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${VULKAN_INCLUDE_DIRS}"
        "-DLINK_LIBRARIES:STRING=${VULKAN_LIBRARIES}"
        OUTPUT_VARIABLE TRY_OUT
        )

if(${OUTPUT_VARIABLE})
  message("OUTPUT_VARIABLE1 = ${TRY_OUT}")
else()
  message("OUTPUT_VARIABLE2 = ${TRY_OUT}")
endif()


#if(NOT ${VALID_VULKAN})
#  message(WARNING "Can't use Vulkan")
#  return()
#endif()

set(HAVE_VULKAN 1)

if(HAVE_VULKAN)
  message("HAVE_VULKAN is ON")
#  add_definitions(-DVK_NO_PROTOTYPES)
  include_directories(${VULKAN_INCLUDE_DIRS})
endif()

#include_directories(${VULKAN_INCLUDE_DIRS})

MARK_AS_ADVANCED(
        VULKAN_INCLUDE_DIRS
        VULKAN_LIBRARIES
)
