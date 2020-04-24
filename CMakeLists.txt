cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
cmake_policy(VERSION 3.7)
project(Vulkan-Edu)

option(USE_BUILT_IN_SDK "Use built-in Vulkan headers. If disabled, VK_SDK_PATH and VULKAN_SDK env vars will be assumed to hold the location where the headers can be found." OFF)
option(USE_BUILT_IN_GLSLANG "Use built in glslang. If disabled, VK_GLSLANG_PATH env vars will be assumed to hold the location where the header can be found." OFF)

message(STATUS "Options")
message(STATUS "-----------------------------------------")
message(STATUS "USE THE BUILT IN SDK:     ${USE_BUILT_IN_SDK}")
message(STATUS "USE THE BUILT IN GLSLANG: ${USE_BUILT_IN_GLSLANG}")
message(STATUS "")

#Checks the cmake version. Since 3.7 introduced Vulkan Module
if(NOT CMAKE_VERSION VERSION_LESS 3.7.0 AND NOT USE_BUILT_IN_SDK)
    message(STATUS "Using module to find Vulkan")
    find_package(Vulkan)
    set(GLSLANG_DIR ${Vulkan_INCLUDE_DIRS}/../glslang)
    message(STATUS ${GLSLANG_DIR})
else()
    message(STATUS "Using the built in SDK for Vulkan")
endif()

IF (NOT Vulkan_FOUND)
	message(FATAL_ERROR "Could not find Vulkan library!")
ELSE()
	message(STATUS ${Vulkan_INCLUDE_DIRS})
    message(STATUS ${Vulkan_SDK_DIRS})
ENDIF()

set(GLSLANGLIBS
    glslang
    HLSL
    OSDependent
    OGLCompiler
    SPIRV)

#Took from Anvil Vulkan
if (MSVC)
    set (DEFAULT_DYNAMIC_VK_DLL "vulkan-1.dll")
else()
    set (DEFAULT_DYNAMIC_VK_DLL "libvulkan.so")
endif()

include_directories(include)
include_directories(external)
include_directories(external/glm)
include_directories(external/glfw/include)

SET (SRC_LIST "include/Vulkan_Edu.h"
              "src/Vulkan_Edu.cpp")


if (NOT MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")
endif()

if (WIN32)
    include_directories($ENV{VK_SDK_PATH}/Include
                        $ENV{VULKAN_SDK}/Include)
else()
    include_directories($ENV{VK_SDK_PATH}/x86_64/include
                        $ENV{VULKAN_SDK}/include
                        $ENV{VULKAN_SDK}/x86_64/include)
endif()

add_library(Vulkan_Edu STATIC ${SRC_LIST})