// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#define TEST_CASE_NAME CV_Quality_SSIM

namespace opencv_test
{
namespace quality_test
{

// expected ssim per channel
const cv::Scalar
    SSIM_EXPECTED_1 = { .1501 }
    , SSIM_EXPECTED_2 = { .7541, .7742, .8095 }
    ;

// static method
TEST(TEST_CASE_NAME, static_)
{
    cv::Mat qMat = {};
    quality_expect_near(quality::QualitySSIM::compute(get_testfile_1a(), get_testfile_1a(), qMat), cv::Scalar(1.)); // ref vs ref == 1.
    check_quality_map(qMat);
}

// single channel, with/without opencl
TEST(TEST_CASE_NAME, single_channel)
{
    auto fn = []() { quality_test(quality::QualitySSIM::create(get_testfile_1a()), get_testfile_1b(), SSIM_EXPECTED_1); };
    OCL_OFF(fn());
    OCL_ON(fn());
}

// multi-channel
TEST(TEST_CASE_NAME, multi_channel)
{
    quality_test(quality::QualitySSIM::create(get_testfile_2a()), get_testfile_2b(), SSIM_EXPECTED_2);
}

// internal a/b test
/*
TEST(TEST_CASE_NAME, performance)
{
    auto ref = get_testfile_1a();
    auto cmp = get_testfile_1b();
    quality_performance_test("SSIM", [&]() { cv::quality::QualitySSIM::compute(ref, cmp, cv::noArray()); });
}
*/
}
} // namespace