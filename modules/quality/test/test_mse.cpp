// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#define TEST_CASE_NAME CV_Quality_MSE

namespace opencv_test
{
namespace quality_test
{

// static method
TEST(TEST_CASE_NAME, static_ )
{
    cv::Mat qMat = {};
    quality_expect_near(quality::QualityMSE::compute(get_testfile_1a(), get_testfile_1a(), qMat), cv::Scalar(0.)); // ref vs ref == 0
    check_quality_map(qMat);
}

// single channel, with and without opencl
TEST(TEST_CASE_NAME, single_channel )
{
    auto fn = []() { quality_test(quality::QualityMSE::create(get_testfile_1a()), get_testfile_1b(), MSE_EXPECTED_1); };
    OCL_OFF( fn() );
    OCL_ON( fn() );
}

// multi-channel
TEST(TEST_CASE_NAME, multi_channel)
{
    quality_test(quality::QualityMSE::create(get_testfile_2a()), get_testfile_2b(), MSE_EXPECTED_2);
}

// internal a/b test
/*
TEST(TEST_CASE_NAME, performance)
{
    auto ref = get_testfile_1a();
    auto cmp = get_testfile_1b();

    quality_performance_test("MSE", [&]() { cv::quality::QualityMSE::compute(ref, cmp, cv::noArray()); });
}
*/
}
} // namespace