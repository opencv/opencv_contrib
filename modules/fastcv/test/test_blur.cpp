/*
 * Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef testing::TestWithParam<tuple<Size, int, int, bool>> GaussianBlurTest;

TEST_P(GaussianBlurTest, accuracy)
{
    cv::Size srcSize = get<0>(GetParam());
    int depth = get<1>(GetParam());
    int ksize = get<2>(GetParam());
    bool border = get<3>(GetParam());

    // For some cases FastCV not support, so skip them
    if((ksize!=5) && (depth!=CV_8U))
        return;

    cv::Mat src(srcSize, depth);
    cv::Mat dst,ref;
    RNG& rng = cv::theRNG();
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(255));

    cv::fastcv::gaussianBlur(src, dst, ksize, border);

    if(depth == CV_32S)
        src.convertTo(src, CV_32F);
    cv::GaussianBlur(src,ref,Size(ksize,ksize),0,0,border);
    ref.convertTo(ref,depth);

    cv::Mat difference;
    cv::absdiff(dst, ref, difference);

    int num_diff_pixels = cv::countNonZero(difference);

    EXPECT_LT(num_diff_pixels, (src.rows+src.cols)*ksize);
}

typedef testing::TestWithParam<tuple<Size, int, int>> Filter2DTest;

TEST_P(Filter2DTest, accuracy)
{
    Size srcSize = get<0>(GetParam());
    int ddepth   = get<1>(GetParam());
    int ksize    = get<2>(GetParam());

    cv::Mat src(srcSize, CV_8U);
    cv::Mat kernel;
    cv::Mat dst, ref;

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

    cv::fastcv::filter2D(src, dst, ddepth, kernel);
    cv::filter2D(src, ref, ddepth, kernel);

    cv::Mat difference;
    dst.convertTo(dst, CV_8U);
    ref.convertTo(ref, CV_8U);
    cv::absdiff(dst, ref, difference);

    int num_diff_pixels = cv::countNonZero(difference);
    EXPECT_LT(num_diff_pixels, (src.rows+src.cols)*ksize);
}

typedef testing::TestWithParam<tuple<Size, int>> SepFilter2DTest;

TEST_P(SepFilter2DTest, accuracy)
{
    Size srcSize = get<0>(GetParam());
    int ksize    = get<1>(GetParam());

    cv::Mat src(srcSize, CV_8U);
    cv::Mat kernel(1,ksize,CV_8S);
    cv::Mat dst,ref;
    RNG& rng = cv::theRNG();
    cvtest::randUni(rng, src, Scalar::all(0), Scalar::all(255));
    cvtest::randUni(rng, kernel, Scalar::all(INT8_MIN), Scalar::all(INT8_MAX));

    cv::fastcv::sepFilter2D(src, dst, CV_8U, kernel, kernel);
    cv::sepFilter2D(src,ref,CV_8U,kernel,kernel);

    cv::Mat difference;
    cv::absdiff(dst, ref, difference);
    int num_diff_pixels = cv::countNonZero(difference);
    EXPECT_LT(num_diff_pixels, (src.rows+src.cols)*ksize);
}

typedef testing::TestWithParam<tuple<int>> NormalizeLocalBoxTest;

TEST_P(NormalizeLocalBoxTest, accuracy)
{
    bool use_stddev = get<0>(GetParam());
    cv::Mat src, dst;
    src = imread(cvtest::findDataFile("cv/shared/baboon.png"), cv::IMREAD_GRAYSCALE);

    cv::fastcv::normalizeLocalBox(src, dst, Size(5,5), use_stddev);
    Scalar s = cv::mean(dst);

    if(use_stddev)
       EXPECT_LT(s[0],1);
    else
       EXPECT_LT(s[0],50);
}


INSTANTIATE_TEST_CASE_P(FastCV_Extension, GaussianBlurTest, Combine(
/*image size*/     ::testing::Values(perf::szVGA, perf::sz720p, perf::sz1080p),
/*image depth*/    ::testing::Values(CV_8U,CV_16S,CV_32S),
/*kernel size*/    ::testing::Values(3, 5),
/*blur border*/    ::testing::Values(true,false)
));

INSTANTIATE_TEST_CASE_P(FastCV_Extension, Filter2DTest, Combine(
/*image sie*/      Values(perf::szVGA, perf::sz720p, perf::sz1080p),
/*dst depth*/      Values(CV_8U,CV_16S,CV_32F),
/*kernel size*/    Values(3, 5, 7, 9, 11)
));

INSTANTIATE_TEST_CASE_P(FastCV_Extension, SepFilter2DTest, Combine(
/*image size*/      Values(perf::szVGA, perf::sz720p, perf::sz1080p),
/*kernel size*/    Values(3, 5, 7, 9, 11)
));

INSTANTIATE_TEST_CASE_P(FastCV_Extension, NormalizeLocalBoxTest, Values(0,1));


}} // namespaces opencv_test, ::