#include "test_precomp.hpp"

#define TEST_CASE_NAME CV_Quality_MSE

namespace opencv_test {
	namespace quality_test {

        // static method
        TEST(TEST_CASE_NAME, static_ )
        {
            std::vector<quality::quality_map_type> qMats = {};
            quality_expect_near(quality::QualityMSE::compute(get_testfile_1a(), get_testfile_1a(), qMats), cv::Scalar(0.)); // ref vs ref == 0
            EXPECT_EQ(qMats.size(), 1);
        }

        // single channel
        TEST(TEST_CASE_NAME, single_channel )
        {
            quality_test(quality::QualityMSE::create(get_testfile_1a()), get_testfile_1b(), MSE_EXPECTED_1);
        }

        // single channel, no opencl
        TEST(TEST_CASE_NAME, single_channel_no_ocl)
        {
            quality_test(quality::QualityMSE::create(get_testfile_1a()), get_testfile_1b(), MSE_EXPECTED_1, true, true );
        }

        // multi-channel
        TEST(TEST_CASE_NAME, multi_channel)
        {
            quality_test(quality::QualityMSE::create(get_testfile_2a()), get_testfile_2b(), MSE_EXPECTED_2);
        }

        // multi-frame test
        TEST(TEST_CASE_NAME, multi_frame)
        {
            // result mse == average mse of all frames
            cv::Scalar expected;
            cv::add(MSE_EXPECTED_1, MSE_EXPECTED_2, expected);
            quality::quality_utils::scalar_multiply(expected, .5);
            quality_test(quality::QualityMSE::create(get_testfile_1a2a()), get_testfile_1b2b(), expected, 2 );
        }

        // internal performance test
        TEST(TEST_CASE_NAME, performance)
        {
            auto ref = get_testfile_1a();
            auto cmp = get_testfile_1b();

            quality_performance_test("MSE", [&]() { cv::quality::QualityMSE::compute(ref, cmp, cv::noArray()); });
        }
	}
} // namespace