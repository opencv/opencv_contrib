/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudev;
using namespace cvtest;

TEST(PyrDown, _8uc1)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_8UC1);

    GpuMat_<uchar> d_src(src);

    GpuMat_<uchar> dst = pyrDown_(d_src);

    Mat dst_gold;
    cv::pyrDown(src, dst_gold);

    ASSERT_MAT_NEAR(dst_gold, dst, 1.0);
}

TEST(PyrUp, _32fc4)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_32FC4);

    GpuMat_<float4> d_src(src);

    GpuMat_<float4> dst = pyrDown_(d_src);

    Mat dst_gold;
    cv::pyrDown(src, dst_gold);

    ASSERT_MAT_NEAR(dst_gold, dst, 1e-4);
}

TEST(PyrDown, GridYOverflow)
{
    // dst_rows = divUp(131077, 2) = 65539 > 65535
    const Size size(8, 65536 * 2 + 5);

    Mat src = randomMat(size, CV_8UC1);

    GpuMat d_src_raw, d_dst_raw;
    createContinuous(size, CV_8UC1, d_src_raw);
    createContinuous(divUp(size.height, 2), divUp(size.width, 2), CV_8UC1, d_dst_raw);
    d_src_raw.upload(src);

    GpuMat_<uchar> d_src(d_src_raw);
    GpuMat_<uchar> dst(d_dst_raw);
    gridPyrDown(d_src, dst);

    Mat dst_gold;
    cv::pyrDown(src, dst_gold);

    ASSERT_MAT_NEAR(dst_gold, dst, 1.0);
}

TEST(PyrUp, GridYOverflow)
{
    // dst_rows = 524291 * 2 = 1048582; divUp(dst_rows, 16) = 65537 > 65535
    const Size size(8, 65536 * 8 + 3);

    Mat src = randomMat(size, CV_8UC1);

    GpuMat d_src_raw, d_dst_raw;
    createContinuous(size, CV_8UC1, d_src_raw);
    createContinuous(size.height * 2, size.width * 2, CV_8UC1, d_dst_raw);
    d_src_raw.upload(src);

    GpuMat_<uchar> d_src(d_src_raw);
    GpuMat_<uchar> dst(d_dst_raw);
    gridPyrUp(d_src, dst);

    Mat dst_gold;
    cv::pyrUp(src, dst_gold);

    ASSERT_MAT_NEAR(dst_gold, dst, 1.0);
}
