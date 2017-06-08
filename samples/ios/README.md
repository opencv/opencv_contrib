## A sample build script for iOS

This folder contains some useful scripts and samples using "extra" modules for the iOS platform.

### How to build an iOS OpenCV framework with extra modules

You can build OpenCV, so it will include the modules from this repository.
The python script `extra_build_framework.py` can be used instead of the default one located at `opencv_directory/platforms/ios/build_framework.py` to build the OpenCV iOS framework with the module `xfeatures2d` (previously the nonfree folder under OpenCV 2.x). The framework will be build with the deployment target equal to 7.0 only for the following architectures:
- armv7
- arm64
- i386"
- x86_64
 
Architectures can be removed or added directly in the script.

If you want to enable another module, use CMake's `BUILD_opencv_*=ON` options. For example, for the module `adas`:
```
args = (" -DCMAKE_SKIP_INSTALL_ALL_DEPENDENCY=OFF " + 
                     " -DOPENCV_EXTRA_MODULES_PATH=%s " + 
                     " -DBUILD_opencv_adas=ON " +
                     ...
```
To disable it:
```
args = (" -DCMAKE_SKIP_INSTALL_ALL_DEPENDENCY=OFF " + 
                     " -DOPENCV_EXTRA_MODULES_PATH=%s " + 
                     " -DBUILD_opencv_adas=OFF " +
                     ...
```