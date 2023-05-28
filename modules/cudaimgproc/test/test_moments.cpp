/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

#ifdef HAVE_CUDA

namespace opencv_test { namespace {

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Moments

PARAM_TEST_CASE(Moments, cv::cuda::DeviceInfo, cv::Size, IsBinary)
{
    static void drawCircle(cv::Mat& dst, const cv::Vec3f& circle, bool fill)
    {
        dst.setTo(cv::Scalar::all(0));
        cv::circle(dst, cv::Point2f(circle[0], circle[1]), (int)circle[2],
                   cv::Scalar::all(255), fill ? -1 : 1, cv::LINE_AA);
    }

    static void drawRectangle(cv::Mat& dst, const cv::Vec4f& rectangle, bool fill)
    {
        dst.setTo(cv::Scalar::all(0));
        cv::rectangle(dst, cv::Point2f(rectangle[0], rectangle[1]),
                      cv::Point2f(rectangle[2], rectangle[3]),
                      cv::Scalar::all(255), fill ? -1 : 1, cv::LINE_AA);
    }

    static void drawEllipse(cv::Mat& dst, const cv::Vec6f& ellipse, bool fill)
    {
        dst.setTo(cv::Scalar::all(0));
        cv::ellipse(dst, cv::Point2f(ellipse[0], ellipse[1]),
                    cv::Size2f(ellipse[2], ellipse[3]), ellipse[4], 0, 360,
                    cv::Scalar::all(255), fill ? -1 : 1, cv::LINE_AA);
    }
};

CUDA_TEST_P(Moments, Accuracy)
{
    const cv::cuda::DeviceInfo devInfo = GET_PARAM(0);
    cv::cuda::setDevice(devInfo.deviceID());
    const cv::Size size = GET_PARAM(1);
    const bool isBinary = GET_PARAM(2);

    const int shapeType = randomInt(0, 3);
    const int shapeIndex = randomInt(0, 4);
    printf("shapeType=%d, shapeIndex=%d\n", shapeType, shapeIndex);

    std::vector<cv::Vec3f> circles(4);
    circles[0] = cv::Vec3i(20, 20, 10);
    circles[1] = cv::Vec3i(90, 87, 15);
    circles[2] = cv::Vec3i(30, 70, 20);
    circles[3] = cv::Vec3i(80, 10, 25);

    std::vector<cv::Vec4f> rectangles(4);
    rectangles[0] = cv::Vec4i(20, 20, 30, 40);
    rectangles[1] = cv::Vec4i(40, 47, 65, 60);
    rectangles[2] = cv::Vec4i(30, 70, 50, 100);
    rectangles[3] = cv::Vec4i(80, 10, 100, 50);

    std::vector<cv::Vec6f> ellipses(4);
    ellipses[0] = cv::Vec6i(20, 20, 10, 15, 0, 0);
    ellipses[1] = cv::Vec6i(90, 87, 15, 30, 30, 0);
    ellipses[2] = cv::Vec6i(30, 70, 20, 25, 60, 0);
    ellipses[3] = cv::Vec6i(80, 10, 25, 50, 75, 0);

    cv::Mat src_cpu(size, CV_8UC1);
    switch(shapeType) {
      case 0: {
        drawCircle(src_cpu, circles[shapeIndex], true);
        break;
      }
      case 1: {
        drawRectangle(src_cpu, rectangles[shapeIndex], true);
        break;
      }
      case 2: {
        drawEllipse(src_cpu, ellipses[shapeIndex], true);
        break;
      }
    }
    cv::cuda::GpuMat src_gpu = loadMat(src_cpu, false);

    const auto t0 = std::chrono::high_resolution_clock::now();
    const cv::Moments moments_cpu = cv::moments(src_cpu, isBinary);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const cv::Moments moments_gpu = cv::cuda::moments(src_gpu, isBinary);
    const auto t2 = std::chrono::high_resolution_clock::now();
    const auto elapsed_time_cpu = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const auto elapsed_time_gpu = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    printf("CPU's moments took %4ldus\n", elapsed_time_cpu);
    printf("GPU's moments took %4ldus\n", elapsed_time_gpu);

    ASSERT_EQ(moments_cpu.m00, moments_gpu.m00);
    ASSERT_EQ(moments_cpu.m10, moments_gpu.m10);
    ASSERT_EQ(moments_cpu.m01, moments_gpu.m01);
    ASSERT_EQ(moments_cpu.m20, moments_gpu.m20);
    ASSERT_EQ(moments_cpu.m11, moments_gpu.m11);
    ASSERT_EQ(moments_cpu.m02, moments_gpu.m02);
    ASSERT_EQ(moments_cpu.m30, moments_gpu.m30);
    ASSERT_EQ(moments_cpu.m21, moments_gpu.m21);
    ASSERT_EQ(moments_cpu.m12, moments_gpu.m12);
    ASSERT_EQ(moments_cpu.m03, moments_gpu.m03);

    ASSERT_NEAR(moments_cpu.mu20, moments_gpu.mu20, 1e-4);
    ASSERT_NEAR(moments_cpu.mu11, moments_gpu.mu11, 1e-4);
    ASSERT_NEAR(moments_cpu.mu02, moments_gpu.mu02, 1e-4);
    ASSERT_NEAR(moments_cpu.mu30, moments_gpu.mu30, 1e-4);
    ASSERT_NEAR(moments_cpu.mu21, moments_gpu.mu21, 1e-4);
    ASSERT_NEAR(moments_cpu.mu12, moments_gpu.mu12, 1e-4);
    ASSERT_NEAR(moments_cpu.mu03, moments_gpu.mu03, 1e-4);

    ASSERT_NEAR(moments_cpu.nu20, moments_gpu.nu20, 1e-4);
    ASSERT_NEAR(moments_cpu.nu11, moments_gpu.nu11, 1e-4);
    ASSERT_NEAR(moments_cpu.nu02, moments_gpu.nu02, 1e-4);
    ASSERT_NEAR(moments_cpu.nu30, moments_gpu.nu30, 1e-4);
    ASSERT_NEAR(moments_cpu.nu21, moments_gpu.nu21, 1e-4);
    ASSERT_NEAR(moments_cpu.nu12, moments_gpu.nu12, 1e-4);
    ASSERT_NEAR(moments_cpu.nu03, moments_gpu.nu03, 1e-4);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, Moments, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    GRAYSCALE_BINARY));

}} // namespace
#endif // HAVE_CUDA
