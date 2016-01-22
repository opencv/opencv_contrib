## Experimental OpenCV's addons example

This folder contains **experimental** scripts for demonstration of OpenCV's addons system.

Note: Some opencv_contrib modules are not wrapped into OpenCV's
addons, because they use some API which is not public. These modules should be
build via general procedure with whole OpenCV.


### How to build OpenCV addons for opencv_contrib

1. Build and install OpenCV (refer to OpenCV addons wiki page) without `opencv_contrib` modules.
2. Start new CMake build, pointing to this directory as source.
3. Build and install this OpenCV Addon package.
4. Build your CMake application with usual `find_package(OpenCV)` command and using addon modules.
