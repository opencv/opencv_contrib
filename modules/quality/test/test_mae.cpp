// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#define TEST_CASE_NAME CV_Quality_MAE

namespace opencv_test
{
namespace quality_test
{

namespace
{
const cv::Scalar
    MAE_MAX_EXPECTED_1 = { 203. },
    MAE_MEAN_EXPECTED_1 = { 33.5824 },
    MAE_MAX_EXPECTED_2 = { 138., 145., 156. },
    MAE_MEAN_EXPECTED_2 = { 5.7918, 6.0645, 5.5609}
    ;
} // anonymous

// static method
TEST(TEST_CASE_NAME, static_max )
{
    // Max
    cv::Mat qMat = {};
    quality_expect_near(quality::QualityMAE::compute(get_testfile_1a(), get_testfile_1a(), qMat, quality::MAE_MAX), cv::Scalar(0.)); // ref vs ref == 0
    check_quality_map(qMat);
}

// static method
TEST(TEST_CASE_NAME, static_mean )
{
    // Mean
    cv::Mat qMat = {};
    quality_expect_near(quality::QualityMAE::compute(get_testfile_1a(), get_testfile_1a(), qMat, quality::MAE_MEAN), cv::Scalar(0.)); // ref vs ref == 0
    check_quality_map(qMat);
}

// single channel, with and without opencl
TEST(TEST_CASE_NAME, single_channel_max )
{
    auto fn = []() { quality_test(quality::QualityMAE::create(get_testfile_1a(), quality::MAE_MAX), get_testfile_1b(), MAE_MAX_EXPECTED_1); };

    OCL_OFF( fn() );
    OCL_ON( fn() );
}

// single channel, with and without opencl
TEST(TEST_CASE_NAME, single_channel_mean )
{
    auto fn = []() { quality_test(quality::QualityMAE::create(get_testfile_1a(), quality::MAE_MEAN), get_testfile_1b(), MAE_MEAN_EXPECTED_1); };

    OCL_OFF( fn() );
    OCL_ON( fn() );
}

// multi-channel max
TEST(TEST_CASE_NAME, multi_channel_max)
{
    quality_test(quality::QualityMAE::create(get_testfile_2a(), quality::MAE_MAX), get_testfile_2b(), MAE_MAX_EXPECTED_2);
}

// multi-channel mean
TEST(TEST_CASE_NAME, multi_channel_mean)
{
    quality_test(quality::QualityMAE::create(get_testfile_2a(), quality::MAE_MEAN), get_testfile_2b(), MAE_MEAN_EXPECTED_2);
}

} // namespace quality_test
} // namespace opencv_test
