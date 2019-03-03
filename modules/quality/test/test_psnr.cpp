// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#define TEST_CASE_NAME CV_Quality_PSNR

namespace opencv_test
{
namespace quality_test
{

const cv::Scalar
    PSNR_EXPECTED_1 = { 14.8347, INFINITY, INFINITY, INFINITY } // matlab: psnr('rock_1.bmp', 'rock_2.bmp') == 14.8347
    , PSNR_EXPECTED_2 = { 28.4542, 27.7402, 27.2886, INFINITY }  // matlab: psnr('rubberwhale1.png', 'rubberwhale2.png') == BGR: 28.4542, 27.7402, 27.2886,  avg 27.8015
;

// static method
TEST(TEST_CASE_NAME, static_)
{
    std::vector<cv::Mat> qMats = {};
    quality_expect_near(quality::QualityPSNR::compute(get_testfile_1a(), get_testfile_1a(), qMats), cv::Scalar(INFINITY,INFINITY,INFINITY,INFINITY)); // ref vs ref == inf
    EXPECT_EQ(qMats.size(), 1U);
}

// single channel, with/without opencl
TEST(TEST_CASE_NAME, single_channel)
{
    auto fn = []() { quality_test(quality::QualityPSNR::create(get_testfile_1a()), get_testfile_1b(), PSNR_EXPECTED_1); };
    OCL_OFF( fn() );
    OCL_ON( fn() );
}

// multi-channel
TEST(TEST_CASE_NAME, multi_channel)
{
    quality_test(quality::QualityPSNR::create(get_testfile_2a()), get_testfile_2b(), PSNR_EXPECTED_2);
}

// multi-frame test
TEST(TEST_CASE_NAME, multi_frame)
{
    cv::Scalar expected;
    cv::add(MSE_EXPECTED_1, MSE_EXPECTED_2, expected);
    expected /= 2.;

    expected = quality::quality_utils::mse_to_psnr(expected, quality::QualityPSNR::MAX_PIXEL_VALUE_DEFAULT );

    quality_test(quality::QualityPSNR::create(get_testfile_1a2a()), get_testfile_1b2b(), expected, 2);
}

// internal a/b test
/*
TEST(TEST_CASE_NAME, performance)
{
    auto ref = get_testfile_1a();
    auto cmp = get_testfile_1b();
    quality_performance_test("PSNR", [&]() { cv::quality::QualityPSNR::compute(ref, cmp, cv::noArray()); });
}
*/
}
} // namespace