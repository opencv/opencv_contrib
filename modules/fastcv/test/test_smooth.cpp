/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef std::tuple<float, float> BilateralTestParams;
class BilateralRecursiveTest : public ::testing::TestWithParam<BilateralTestParams> {};

TEST_P(BilateralRecursiveTest, accuracy)
{
    auto p = GetParam();
    float sigmaColor = std::get<0>(p);
    float sigmaSpace = std::get<1>(p);

    cv::Mat src = imread(cvtest::findDataFile("cv/shared/baboon.png"), cv::IMREAD_GRAYSCALE);

    Mat dst;
    cv::fastcv::bilateralRecursive(src, dst, sigmaColor, sigmaSpace);

    // NOTE: test files should be manually loaded to folder on a device, for example like this:
    // adb push fastcv/misc/bilateral_recursive/ /sdcard/testdata/fastcv/bilateral/
    cv::Mat ref = imread(cvtest::findDataFile(cv::format("fastcv/bilateral/rec_%2f_%2f.png", sigmaColor, sigmaSpace)),
                         IMREAD_GRAYSCALE);

    if (cvtest::debugLevel > 0)
    {
        cv::imwrite(cv::format("rec_%2f_%2f.png", sigmaColor, sigmaSpace), dst);
    }

    double normInf = cvtest::norm(dst, ref, cv::NORM_INF);
    double normL2  = cvtest::norm(dst, ref, cv::NORM_L2);

    ASSERT_LT(normInf, 1);
    ASSERT_LT(normL2, 1.f / src.size().area());
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, BilateralRecursiveTest,
                        ::testing::Values(
                            BilateralTestParams {0.01f, 1.00f},
                            BilateralTestParams {0.10f, 0.01f},
                            BilateralTestParams {1.00f, 0.01f},
                            BilateralTestParams {1.00f, 1.00f},
                            BilateralTestParams {5.00f, 0.01f},
                            BilateralTestParams {5.00f, 0.10f},
                            BilateralTestParams {5.00f, 5.00f}
                        ));

}} // namespaces opencv_test, ::
