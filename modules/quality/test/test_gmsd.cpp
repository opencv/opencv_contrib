// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#define TEST_CASE_NAME CV_Quality_GMSD

namespace opencv_test
{
namespace quality_test
{

// gmsd per channel
const cv::Scalar
    GMSD_EXPECTED_1 = { .2393 }
    , GMSD_EXPECTED_2 = { .0942, .1016, .0995 }
;

// static method
TEST(TEST_CASE_NAME, static_)
{
    std::vector<cv::Mat> qMats = {};
    quality_expect_near(quality::QualityGMSD::compute(get_testfile_1a(), get_testfile_1a(), qMats), cv::Scalar(0.)); // ref vs ref == 0.
    EXPECT_EQ(qMats.size(), 1U );
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

// multi-frame test
TEST(TEST_CASE_NAME, multi_frame)
{
    // result == average of all frames
    cv::Scalar expected;
    cv::add(GMSD_EXPECTED_1, GMSD_EXPECTED_2, expected);
    expected /= 2.;

    quality_test(quality::QualityGMSD::create(get_testfile_1a2a()), get_testfile_1b2b(), expected, 2);
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