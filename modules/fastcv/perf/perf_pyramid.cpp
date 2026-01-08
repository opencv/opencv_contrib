/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef std::tuple<bool /*useFloat*/, int /*nLevels*/, bool /*scaleBy2*/> PyramidTestParams;
class PyramidTest : public ::perf::TestBaseWithParam<PyramidTestParams> { };

PERF_TEST_P(PyramidTest, checkAllVersions, // version, useFloat, nLevels
                        ::testing::Values(
                            PyramidTestParams { true, 2,  true}, PyramidTestParams { true, 3,  true}, PyramidTestParams { true, 4,  true},
                            PyramidTestParams {false, 2,  true}, PyramidTestParams {false, 3,  true}, PyramidTestParams {false, 4,  true},
                            PyramidTestParams {false, 2, false}, PyramidTestParams {false, 3, false}, PyramidTestParams {false, 4, false}
                            ))
{
    auto par = GetParam();

    bool useFloat = std::get<0>(par);
    int  nLevels  = std::get<1>(par);
    bool scaleBy2 = std::get<2>(par);

    cv::Mat src = imread(cvtest::findDataFile("cv/shared/baboon.png"), cv::IMREAD_GRAYSCALE);

    if (useFloat)
    {
        cv::Mat f;
        src.convertTo(f, CV_32F);
        src = f;
    }

    while(next())
    {
        std::vector<cv::Mat> pyr;
        startTimer();
        cv::fastcv::buildPyramid(src, pyr, nLevels, scaleBy2);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}


typedef std::tuple<MatType, size_t> SobelPyramidTestParams;
class SobelPyramidTest : public ::perf::TestBaseWithParam<SobelPyramidTestParams> {};

PERF_TEST_P(SobelPyramidTest, checkAllTypes,
    ::testing::Combine(::testing::Values(CV_8S, CV_16S, CV_32F),
                       ::testing::Values(3, 6)))
{
    auto p = GetParam();
    int    type    = std::get<0>(p);
    size_t nLevels = std::get<1>(p);

    // NOTE: test files should be manually loaded to folder on a device, for example like this:
    // adb push fastcv/misc/bilateral_recursive/ /sdcard/testdata/fastcv/bilateral/
    cv::Mat src = imread(cvtest::findDataFile("cv/shared/baboon.png"), cv::IMREAD_GRAYSCALE);

    std::vector<cv::Mat> pyr;
    cv::fastcv::buildPyramid(src, pyr, nLevels);

    while(next())
    {
        std::vector<cv::Mat> pyrDx, pyrDy;
        startTimer();
        cv::fastcv::sobelPyramid(pyr, pyrDx, pyrDy, type);
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
