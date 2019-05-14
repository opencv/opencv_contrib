// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#define TEST_CASE_NAME CV_Quality_GMSD

namespace opencv_test
{
namespace quality_test
{

// expected gmsd per channel
const cv::Scalar
    GMSD_EXPECTED_1 = { .2393 }
    , GMSD_EXPECTED_2 = { .0942, .1016, .0995 }
;

// static method
TEST(TEST_CASE_NAME, static_)
{
    cv::Mat qMat = {};
    quality_expect_near(quality::QualityGMSD::compute(get_testfile_1a(), get_testfile_1a(), qMat), cv::Scalar(0.)); // ref vs ref == 0.
    check_quality_map(qMat);
}

// single channel, with and without opencl
TEST(TEST_CASE_NAME, single_channel)
{
    auto fn = []() { quality_test(quality::QualityGMSD::create(get_testfile_1a()), get_testfile_1b(), GMSD_EXPECTED_1); };
    OCL_OFF(fn());
    OCL_ON(fn());
}

// multi-channel
TEST(TEST_CASE_NAME, multi_channel)
{
    quality_test(quality::QualityGMSD::create(get_testfile_2a()), get_testfile_2b(), GMSD_EXPECTED_2);
}

// internal A/B test
/*
TEST(TEST_CASE_NAME, performance)
{
    auto ref = get_testfile_1a();
    auto cmp = get_testfile_1b();
    quality_performance_test("GMSD", [&]() { cv::quality::QualityGMSD::compute(ref, cmp, cv::noArray()); });
}
*/

}
} // namespace