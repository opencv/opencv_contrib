/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

class DCTExtTest : public ::testing::TestWithParam<cv::Size> {};

TEST_P(DCTExtTest, forward)
{
    Size size = GetParam();

    RNG& rng = cv::theRNG();
    Mat src(size, CV_8UC1);
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(255));
    Mat srcFloat;
    src.convertTo(srcFloat, CV_32F);

    Mat dst, ref;
    cv::fastcv::DCT(src, dst);

    cv::dct(srcFloat, ref);

    Mat dstFloat;
    ref.convertTo(dstFloat, CV_32F);

    double normInf = cvtest::norm(dstFloat, ref, cv::NORM_INF);
    double normL2  = cvtest::norm(dstFloat, ref, cv::NORM_L2)  / dst.size().area();

    if (cvtest::debugLevel > 0)
    {
        std::cout << "dst:" << std::endl << dst << std::endl;
        std::cout << "ref:" << std::endl << ref << std::endl;
    }

    EXPECT_EQ(normInf, 0);
    EXPECT_EQ(normL2, 0);
}

TEST_P(DCTExtTest, inverse)
{
    Size size = GetParam();

    RNG& rng = cv::theRNG();
    Mat src(size, CV_8UC1);
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(256));

    Mat srcFloat;
    src.convertTo(srcFloat, CV_32F);

    Mat fwd, back;
    cv::fastcv::DCT(src, fwd);
    cv::fastcv::IDCT(fwd, back);
    Mat backFloat;
    back.convertTo(backFloat, CV_32F);

    Mat fwdRef, backRef;
    cv::dct(srcFloat, fwdRef);
    cv::idct(fwdRef, backRef);

    double normInf = cvtest::norm(backFloat, backRef, cv::NORM_INF);
    double normL2  = cvtest::norm(backFloat, backRef, cv::NORM_L2)  / src.size().area();

    if (cvtest::debugLevel > 0)
    {
        std::cout << "src:"     << std::endl << src     << std::endl;
        std::cout << "back:"    << std::endl << back    << std::endl;
        std::cout << "backRef:" << std::endl << backRef << std::endl;
    }

    EXPECT_LE(normInf, 7.00005);
    EXPECT_LT(normL2,  0.13);
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, DCTExtTest, ::testing::Values(Size(8, 8), Size(128, 128), Size(32, 256), Size(512, 512)));

}} // namespaces opencv_test, ::
