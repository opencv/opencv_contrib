// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#define TEST_CASE_NAME CV_Quality_BRISQUE

namespace opencv_test
{
namespace quality_test
{

// brisque per channel
const cv::Scalar
    BRISQUE_EXPECTED_1 = { 31.866388320922852 }  // testfile_1a
    , BRISQUE_EXPECTED_2 = { 9.7544803619384766 }   // testfile 2a
;

// default model and range file names
static const char* MODEL_FNAME = "brisque_model_live.yml";
static const char* RANGE_FNAME = "brisque_range_live.yml";

// instantiates a brisque object for testing
inline cv::Ptr<quality::QualityBRISQUE> create_brisque()
{
    // location of BRISQUE model and range file
    //  place these files in ${OPENCV_TEST_DATA_PATH}/quality/, or the tests will be skipped
    const auto model = cvtest::findDataFile(MODEL_FNAME, false);
    const auto range = cvtest::findDataFile(RANGE_FNAME, false);
    return quality::QualityBRISQUE::create(model, range);
}

// static method
TEST(TEST_CASE_NAME, static_ )
{
    quality_expect_near(
        quality::QualityBRISQUE::compute(
            get_testfile_1a()
            , cvtest::findDataFile(MODEL_FNAME, false)
            , cvtest::findDataFile(RANGE_FNAME, false)
        )
        , BRISQUE_EXPECTED_1
    );
}

// single channel, instance method, with and without opencl
TEST(TEST_CASE_NAME, single_channel )
{
    auto fn = []() { quality_test(create_brisque(), get_testfile_1a(), BRISQUE_EXPECTED_1, 0, true ); };
    OCL_OFF( fn() );
    OCL_ON( fn() );
}

// multi-channel
TEST(TEST_CASE_NAME, multi_channel)
{
    quality_test(create_brisque(), get_testfile_2a(), BRISQUE_EXPECTED_2, 0, true);
}

// multi-frame test
TEST(TEST_CASE_NAME, multi_frame)
{
    // result mse == average of all frames
    cv::Scalar expected;
    cv::add(BRISQUE_EXPECTED_1, BRISQUE_EXPECTED_2, expected);
    expected /= 2.;

    quality_test(create_brisque(), get_testfile_1a2a(), expected, 0, true );
}

// check brisque model/range persistence
TEST(TEST_CASE_NAME, model_persistence )
{
    auto ptr = create_brisque();
    auto fn = [&ptr]() { quality_test(ptr, get_testfile_1a(), BRISQUE_EXPECTED_1, 0, true); };
    fn();
    fn();   // model/range should persist with brisque ptr through multiple invocations
}

// check compute features interface method
TEST(TEST_CASE_NAME, compute_features)
{
    auto ptr = create_brisque();
    cv::Mat features;
    ptr->computeFeatures(get_testfile_1a(), features);

    EXPECT_EQ(features.rows, 1);
    EXPECT_EQ(features.cols, 36);
}

/*
// internal a/b test
TEST(TEST_CASE_NAME, performance)
{
    auto ref = get_testfile_1a();
    auto alg = create_brisque();

    quality_performance_test("BRISQUE", [&]() { alg->compute(ref); });
}
*/
}
} // namespace