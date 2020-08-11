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

SET (SRC_LIST "src/main.cpp" ${CMAKE_SOURCE_DIR}/include/tiny_obj_loader.cc "src/texture.cpp")
add_executable(Lab_6 ${SRC_LIST})
target_link_libraries(Lab_6 Vulkan_Edu glfw glslang HLSL OSDependent OGLCompiler SPIRV Vulkan::Vulkan ${CMAKE_SOURCE_DIR}/lib/FreeImage.lib)
if(UNIX)
target_link_libraries(Lab_6 ${OS_SPECIFIC_LIBS})
endif()

if(WIN32)
configure_file("${CMAKE_SOURCE_DIR}/lib/FreeImage.dll" "${CMAKE_BINARY_DIR}/examples/Lab\ 6/Debug/FreeImage.dll" COPYONLY)
endif()

file(COPY "shaders" DESTINATION "./")
file(COPY "../../data/cube.obj" DESTINATION "./data/")
file(COPY "../../data/crate1.jpg" DESTINATION "./data/")
file(COPY "../../data/crate2.jpg" DESTINATION "./data/")