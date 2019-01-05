#include "test_precomp.hpp"
#include <chrono>
#include <opencv2/quality/quality_utils.hpp>

#define TEST_CASE_NAME CV_Quality_SSIM

namespace opencv_test {
    namespace quality_test {

        // ssim per channel
        const cv::Scalar
            SSIM_EXPECTED_1 = { .8511 }
            , SSIM_EXPECTED_2 = { .7541, .7742, .8095 }
            ;

        // static method
        TEST(TEST_CASE_NAME, static_)
        {
            std::vector<quality::quality_map_type> qMats = {};
            quality_expect_near(quality::QualitySSIM::compute(get_testfile_1a(), get_testfile_1a(), qMats), cv::Scalar(1.)); // ref vs ref == 1.
            EXPECT_EQ(qMats.size(), 1);
        }

        // single channel
        TEST(TEST_CASE_NAME, single_channel)
        {
            quality_test(quality::QualitySSIM::create(get_testfile_1a()), get_testfile_1b(), SSIM_EXPECTED_1);
        }

        // single channel, no opencl
        TEST(TEST_CASE_NAME, single_channel_no_ocl)
        {
            quality_test(quality::QualitySSIM::create(get_testfile_1a()), get_testfile_1b(), SSIM_EXPECTED_1, true, true);
        }

        // multi-channel
        TEST(TEST_CASE_NAME, multi_channel)
        {
            quality_test(quality::QualitySSIM::create(get_testfile_2a()), get_testfile_2b(), SSIM_EXPECTED_2);
        }

        // multi-frame test
        TEST(TEST_CASE_NAME, multi_frame)
        {
            // result == average of all frames
            cv::Scalar expected;
            cv::add(SSIM_EXPECTED_1, SSIM_EXPECTED_2, expected);
            quality::quality_utils::scalar_multiply(expected, .5);

            quality_test(quality::QualitySSIM::create(get_testfile_1a2a()), get_testfile_1b2b(), expected, 2);
        }

        // internal performance test
        TEST(TEST_CASE_NAME, performance)
        {
            auto ref = get_testfile_1a();
            auto cmp = get_testfile_1b();
            quality_performance_test("SSIM", [&]() { cv::quality::QualitySSIM::compute(ref, cmp, cv::noArray()); });
        }
    }
} // namespace