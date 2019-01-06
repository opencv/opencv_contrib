#include "test_precomp.hpp"

#define TEST_CASE_NAME CV_Quality_GMSD

namespace opencv_test {
    namespace quality_test {

        // gmsd per channel
        const cv::Scalar
            GMSD_EXPECTED_1 = { .2393 }
            , GMSD_EXPECTED_2 = { .0942, .1016, .0995 }
        ;

        // bug report, https://github.com/opencv/opencv/issues/13577
        /*
        TEST(TEST_CASE_NAME, cv_resize_bug )
        {
            cv::ocl::setUseOpenCL(false);
            UMat foo( 10, 10, CV_32FC1 );
            cv::resize(foo, foo, cv::Size(), .5, .5 );
        }
        */

        // single channel, no opencl
        TEST(TEST_CASE_NAME, single_channel_no_ocl)
        {
            quality_test(quality::QualityGMSD::create(get_testfile_1a()), get_testfile_1b(), GMSD_EXPECTED_1, true, true);
        }

        // static method
        TEST(TEST_CASE_NAME, static_)
        {
            std::vector<quality::quality_map_type> qMats = {};
            quality_expect_near(quality::QualityGMSD::compute(get_testfile_1a(), get_testfile_1a(), qMats), cv::Scalar(0.)); // ref vs ref == 0.
            EXPECT_EQ(qMats.size(), 1U );
        }

        // single channel
        TEST(TEST_CASE_NAME, single_channel)
        {
            quality_test(quality::QualityGMSD::create(get_testfile_1a()), get_testfile_1b(), GMSD_EXPECTED_1);
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
            quality::quality_utils::scalar_multiply(expected, .5);

            quality_test(quality::QualityGMSD::create(get_testfile_1a2a()), get_testfile_1b2b(), expected, 2);
        }

        // internal performance test
        TEST(TEST_CASE_NAME, performance)
        {
            auto ref = get_testfile_1a();
            auto cmp = get_testfile_1b();
            quality_performance_test("GMSD", [&]() { cv::quality::QualityGMSD::compute(ref, cmp, cv::noArray()); });
        }
    }
} // namespace