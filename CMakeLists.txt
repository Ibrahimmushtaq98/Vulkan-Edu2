cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
cmake_policy(VERSION 3.7)
project(Vulkan_Edu)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)
include_directories(external)
include_directories(external/glm)
include_directories(external/glfw/include)

add_subdirectory(external/glfw)
add_subdirectory(external/glslang)
if(WIN32)
    find_package(vulkan REQUIRED)
else()
    find_package(Vulkan REQUIRED)
endif()

SET (SRC_LIST "include/Vulkan_Edu.h"
              "src/Vulkan_Edu.cpp")

if (NOT MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG -std=c++17")
endif()

if (WIN32)
    include_directories($ENV{VK_SDK_PATH}/Include
                        $ENV{VULKAN_SDK}/Include
                        $ENV{VK_SDK_PATH}/SPIRV
                        $ENV{VULKAN_SDK}/SPIRV
                        $ENV{VK_SDK_PATH}/glslang
                        $ENV{VULKAN_SDK}/glslang)
else()
    include_directories($ENV{VK_SDK_PATH}/x86_64/include
                        $ENV{VULKAN_SDK}/include
                        $ENV{VULKAN_SDK}/x86_64/include)
endif()

add_library(${PROJECT_NAME} STATIC ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} glslang HLSL OSDependent OGLCompiler SPIRV Vulkan::Vulkan)
add_subdirectory("examples" "${CMAKE_BINARY_DIR}/examples")