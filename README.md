# Vulkan_Edu Library
This library is aimed to tackle the amount of boilerplate code that comes with Vulkan. This project was for my undergraduate thesis, however I am still continuing this project as there are bugs to be worked out.

This project works on 
 - Windows OS
 - Linux OS (Using the X11)

I currently do have plan for making the project work on Mac OS

## Cloning 
This repository contains submodules for external dependencies, so when doing a fresh clone you need to clone recursively:
```
git clone --recursive https://github.com/Ibrahimmushtaq98/Vulkan-Edu2.git
```
If you have an existing repository, you can update it manually:
```
git submodule init
git submodule update
```

## Things to finish

 - [ ] Make the project work on all platform
 - [ ] Add staging buffers to map data efficiently on the GPU
 - [ ] Add all the remaining labs
 - [ ] Add more examples

## Credits
Special thanks to author of these library:
 - [OpenGL Mathematics (GLM)](https://github.com/g-truc/glm)
 - [GLFW](https://github.com/glfw/glfw)
 - [FreeImage](http://freeimage.sourceforge.net/)
 - [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader)