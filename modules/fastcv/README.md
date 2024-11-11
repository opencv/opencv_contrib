FastCV extension for OpenCV
===========================

This module provides wrappers for several FastCV functions not covered by the corresponding HAL in OpenCV or have implementation incompatible with OpenCV.
Please note that:
1. This module supports ARM architecture only. This means that CMake script aborts configuration under x86 platform even if you don't want to build binaries for your machine and just want to build docs or enable code analysis in your IDE. In that case you should fix CMakeLists.txt file as told inside it.
2. Test data is stored in misc folder. Before running tests on a device you should copy the content of `misc/` folder to `$YOUR_TESTDATA_PATH/fastcv/` folder on a device.
