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
// Copyright (C) 2015, University of Ostrava, Institute for Research and Applications of Fuzzy Modeling,
// Pavel Vlasanek, all rights reserved. Third party copyrights are property of their respective owners.
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

namespace opencv_test { namespace {

TEST(fuzzy_image, inpainting)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "fuzzy/";
    Mat orig = imread(folder + "orig.png");
    Mat exp1 = imread(folder + "exp1.png");
    Mat exp2 = imread(folder + "exp2.png");
    Mat exp3 = imread(folder + "exp3.png");
    Mat mask1 = imread(folder + "mask1.png", IMREAD_GRAYSCALE);
    Mat mask2 = imread(folder + "mask2.png", IMREAD_GRAYSCALE);

    EXPECT_TRUE(!orig.empty() && !exp1.empty() && !exp2.empty() && !exp3.empty() && !mask1.empty() && !mask2.empty());

    Mat res1, res2, res3;
    ft::inpaint(orig, mask1, res1, 2, ft::LINEAR, ft::ONE_STEP);
    ft::inpaint(orig, mask2, res2, 2, ft::LINEAR, ft::MULTI_STEP);
    ft::inpaint(orig, mask2, res3, 2, ft::LINEAR, ft::ITERATIVE);

    res1.convertTo(res1, CV_8UC3);
    res2.convertTo(res2, CV_8UC3);
    res3.convertTo(res3, CV_8UC3);

    double n1 = cvtest::norm(exp1, res1, NORM_INF);
    double n2 = cvtest::norm(exp2, res2, NORM_INF);
    double n3 = cvtest::norm(exp3, res3, NORM_INF);

    EXPECT_LE(n1, 1);
    EXPECT_LE(n2, 1);
    EXPECT_LE(n3, 1);
}

TEST(fuzzy_image, filtering)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "fuzzy/";
    Mat orig = imread(folder + "orig.png");
    Mat exp4 = imread(folder + "exp4.png");

    EXPECT_TRUE(!orig.empty() && !exp4.empty());

    Mat kernel;
    ft::createKernel(ft::LINEAR, 20, kernel, 3);

    Mat res4;
    ft::filter(orig, kernel, res4);

    res4.convertTo(res4, CV_8UC3);

    double n1 = cvtest::norm(exp4, res4, NORM_INF);

    EXPECT_LE(n1, 1);
}

TEST(fuzzy_image, kernel)
{
    Mat kernel1;
    ft::createKernel(ft::LINEAR, 2, kernel1, 1);

    Mat vectorA = (Mat_<float>(1, 5) << 0, 0.5, 1, 0.5, 0);
    Mat vectorB = (Mat_<float>(5, 1) << 0, 0.5, 1, 0.5, 0);

    Mat kernel2;
    ft::createKernel(vectorA, vectorB, kernel2, 1);

    double diff = cvtest::norm(kernel1, kernel2, NORM_INF);

    EXPECT_DOUBLE_EQ(diff, 0);
}

}} // namespace
