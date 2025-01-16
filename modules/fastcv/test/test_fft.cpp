/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

class FFTExtTest : public ::testing::TestWithParam<cv::Size> {};

TEST_P(FFTExtTest, forward)
{
    Size size = GetParam();

    RNG& rng = cv::theRNG();
    Mat src(size, CV_8UC1);
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(256));

    Mat srcFloat;
    src.convertTo(srcFloat, CV_32F);

    Mat dst, ref;
    cv::fastcv::FFT(src, dst);

    cv::dft(srcFloat, ref, DFT_COMPLEX_OUTPUT);

    double normInf = cvtest::norm(dst, ref, cv::NORM_INF);
    double normL2  = cvtest::norm(dst, ref, cv::NORM_L2)  / dst.size().area();

    EXPECT_LT(normInf, 19.1); // for 512x512 case
    EXPECT_LT(normL2, 18.0 / 256.0 );
}

TEST_P(FFTExtTest, inverse)
{
    Size size = GetParam();

    RNG& rng = cv::theRNG();
    Mat src(size, CV_8UC1);
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(256));

    Mat srcFloat;
    src.convertTo(srcFloat, CV_32F);

    Mat fwd, back;
    cv::fastcv::FFT(src, fwd);
    cv::fastcv::IFFT(fwd, back);
    Mat backFloat;
    back.convertTo(backFloat, CV_32F);

    Mat fwdRef, backRef;
    cv::dft(srcFloat, fwdRef, DFT_COMPLEX_OUTPUT);
    cv::idft(fwdRef, backRef, DFT_REAL_OUTPUT);

    backRef *= 1./(src.size().area());

    double normInf = cvtest::norm(backFloat, backRef, cv::NORM_INF);
    double normL2  = cvtest::norm(backFloat, backRef, cv::NORM_L2)  / src.size().area();

    EXPECT_LT(normInf, 9.16e-05);
    EXPECT_LT(normL2,  1.228e-06);
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, FFTExtTest, ::testing::Values(Size(8, 8), Size(128, 128), Size(32, 256), Size(512, 512),
                                                                        Size(32, 1), Size(512, 1)));

}} // namespaces opencv_test, ::
