cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
cmake_policy(VERSION 3.6)
project(Vulkan-Edu)

find_package(Vulkan REQUIRED)

include_directories(include)
include_directories(external)
include_directories(external/glm)
include_directories(external/glfw/include)

SET (SRC_LIST "include/Vulkan_Edu.h"
              "src/Vulkan_Edu.cpp")


if (NOT MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")
endif()

option(USE_BUILT_IN_SDK "Use built-in Vulkan headers. If disabled, VK_SDK_PATH and VULKAN_SDK env vars will be assumed to hold the location where the headers can be found." OFF)

add_library(Vulkan_Edu STATIC ${SRC_LIST})