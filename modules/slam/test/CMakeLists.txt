# Standard OpenCV module test directory.
#
# Tests are discovered automatically by the OpenCV build system.
# This file is intentionally minimal: the parent CMakeLists.txt
# uses ocv_add_module() which sets up the test target.
#
# Build with: cmake -DBUILD_TESTS=ON ... && make opencv_test_slam
# Run with:   ./bin/opencv_test_slam --gtest_filter=SLAM.*