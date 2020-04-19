# vulkan-edu
to start, i place the `shaders` folder and `.obj` files  in the debug folder

or you can change the working directory to `../Lab\ 1` and `../Lab\ 2` and such, just the ones containing the `main.cpp`

environment path for u mac users:
```commandline
VK_ICD_FILENAMES = /Users/mechs/Desktop/vulkan-edu-master/vulkansdk-macos/macOS/share/vulkan/icd.d/MoltenVK_icd.json;
VK_LAYER_PATH = /Users/mechs/Desktop/vulkan-edu-master/vulkansdk-macos/macOS/share/vulkan/explicit_layer.d
```
or perhaps run `setup-env.sh` as described in https://vulkan.lunarg.com/doc/sdk/latest/mac/getting_started.html

i have to set the environment path for every lab, maybe theres a better way out there

for cloning, try
```sh
git clone --depth 1 --recurse-submodules https://github.com/Ibrahimmushtaq98/Vulkan-Edu2
cd Vulkan-Edu2
git submodule update --remote
```

theres also a one-liner on a newer version of git
```sh
git clone --depth 1 --recurse-submodules --remote-submodules https://github.com/Ibrahimmushtaq98/Vulkan-Edu2
```
then do the directory/file shuffle

## Linking FreeImage
First I downloaded the **Source distribution** on http://freeimage.sourceforge.net/download.html and followed the instructions on `README.osx`. I linked with the static library `FreeImage/Dist/libfreeimage.a`
After that I ran into a problem like on https://stackoverflow.com/q/22922585 , both the answers worked for me