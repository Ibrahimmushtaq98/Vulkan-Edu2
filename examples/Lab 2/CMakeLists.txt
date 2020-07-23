include_directories(${CMAKE_SOURCE_DIR}include)
include_directories(${CMAKE_SOURCE_DIR}external)
include_directories(${CMAKE_SOURCE_DIR}external/glm)
include_directories(${CMAKE_SOURCE_DIR}external/glfw/include)

if(UNIX)
FIND_PACKAGE(X11 REQUIRED)
SET(OS_SPECIFIC_LIBS
   xcb
   X11-xcb
   ${X11_LIBRARIES}
)
endif()

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

SET (SRC_LIST "src/main.cpp" ${CMAKE_SOURCE_DIR}/include/tiny_obj_loader.cc)
add_executable(Lab_2 ${SRC_LIST})
target_link_libraries(Lab_2 Vulkan_Edu glfw glslang HLSL OSDependent OGLCompiler SPIRV Vulkan::Vulkan)
if(UNIX)
target_link_libraries(Lab_2 ${OS_SPECIFIC_LIBS})
endif()
file(COPY "shaders" DESTINATION "./")
file(COPY "../../data/sphere.obj" DESTINATION "./data/")