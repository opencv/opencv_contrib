/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef testing::TestWithParam<tuple<Size, int, int>> Filter2DTest_DSP;

TEST_P(Filter2DTest_DSP, accuracy)
{
    applyTestTag(CV_TEST_TAG_FASTCV_SKIP_DSP);

    //Initialize DSP
    int initStatus = cv::fastcv::dsp::fcvdspinit();
    ASSERT_EQ(initStatus, 0) << "Failed to initialize FastCV DSP";

    Size srcSize = get<0>(GetParam());
    int ddepth   = get<1>(GetParam());
    int ksize    = get<2>(GetParam());

    cv::Mat src;
    src.allocator = cv::fastcv::getQcAllocator();
    src.create(srcSize, CV_8U);

    cv::Mat kernel;
    cv::Mat dst, ref;
    kernel.allocator = cv::fastcv::getQcAllocator();
    dst.allocator = cv::fastcv::getQcAllocator();

    switch (ddepth)
    {
        case CV_8U:
        case CV_16S:
        {
            kernel.create(ksize,ksize,CV_8S);
            break;
        }
        case CV_32F:
        {
            kernel.create(ksize,ksize,CV_32F);
            break;
        }
        default:
            return;
    }

    RNG& rng = cv::theRNG();
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(255));
    cvtest::randUni(rng, kernel, Scalar::all(INT8_MIN), Scalar::all(INT8_MAX));

    cv::fastcv::dsp::filter2D(src, dst, ddepth, kernel);

    //De-Initialize DSP
    cv::fastcv::dsp::fcvdspdeinit();

    cv::filter2D(src, ref, ddepth, kernel);
    cv::Mat difference;
    dst.convertTo(dst, CV_8U);
    ref.convertTo(ref, CV_8U);
    cv::absdiff(dst, ref, difference);

    int num_diff_pixels = cv::countNonZero(difference);
    EXPECT_LT(num_diff_pixels, (src.rows+src.cols)*ksize);
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, Filter2DTest_DSP, Combine(
/*image size*/      Values(perf::szVGA, perf::sz720p),
/*dst depth*/      Values(CV_8U,CV_16S,CV_32F),
/*kernel size*/    Values(3, 5, 7, 9, 11)
));

}} // namespaces opencv_test, ::
