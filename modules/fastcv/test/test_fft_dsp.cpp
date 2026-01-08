/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

class FFT_DSPExtTest : public ::testing::TestWithParam<cv::Size> {};

TEST_P(FFT_DSPExtTest, forward)
{
    applyTestTag(CV_TEST_TAG_FASTCV_SKIP_DSP);

    //Initialize DSP
    int initStatus = cv::fastcv::dsp::fcvdspinit();
    ASSERT_EQ(initStatus, 0) << "Failed to initialize FastCV DSP";

    Size size = GetParam();

    RNG& rng = cv::theRNG();

    Mat src;
    src.allocator = cv::fastcv::getQcAllocator();
    src.create(size, CV_8UC1);

    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(256));

    Mat srcFloat;
    src.convertTo(srcFloat, CV_32F);

    Mat dst, ref;
    dst.allocator = cv::fastcv::getQcAllocator();
    cv::fastcv::dsp::FFT(src, dst);

    //De-Initialize DSP
    cv::fastcv::dsp::fcvdspdeinit();

    cv::dft(srcFloat, ref, DFT_COMPLEX_OUTPUT);

    double normInf = cvtest::norm(dst, ref, cv::NORM_INF);
    double normL2  = cvtest::norm(dst, ref, cv::NORM_L2)  / dst.size().area();

    EXPECT_LT(normInf, 19.1); // for 512x512 case
    EXPECT_LT(normL2, 18.0 / 256.0 );
}

TEST_P(FFT_DSPExtTest, inverse)
{
    applyTestTag(CV_TEST_TAG_FASTCV_SKIP_DSP);

    //Initialize DSP
    int initStatus = cv::fastcv::dsp::fcvdspinit();
    ASSERT_EQ(initStatus, 0) << "Failed to initialize FastCV DSP";

    Size size = GetParam();

    RNG& rng = cv::theRNG();

    Mat src;
    src.allocator = cv::fastcv::getQcAllocator();
    src.create(size, CV_8UC1);

    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(256));

    Mat srcFloat;
    src.convertTo(srcFloat, CV_32F);

    Mat fwd, back;
    fwd.allocator = cv::fastcv::getQcAllocator();
    back.allocator = cv::fastcv::getQcAllocator();

    cv::fastcv::dsp::FFT(src, fwd);
    cv::fastcv::dsp::IFFT(fwd, back);

    //De-Initialize DSP
    cv::fastcv::dsp::fcvdspdeinit();

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

INSTANTIATE_TEST_CASE_P(FastCV_Extension, FFT_DSPExtTest, ::testing::Values(Size(256, 256), Size(512, 512)));

}} // namespaces opencv_test, ::
