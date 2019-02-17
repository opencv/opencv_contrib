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
// computed test file values via original brisque impl, libsvm 318, opencv 2.x
const cv::Scalar
    BRISQUE_EXPECTED_1 = { 31.155 }  // testfile_1a
    , BRISQUE_EXPECTED_2 = { 15.4114 }  // testfile_2a
;

inline cv::String readfile(const cv::String& path)
{
    std::ifstream is{ path };
    std::stringstream buffer;
    buffer << is.rdbuf();
    return buffer.str();
}

// location of BRISQUE model and range file
//  place these files in ${OPENCV_TEST_DATA_PATH}/quality/, or the tests will be skipped
inline cv::String readbrisquemodel() { return readfile(cvtest::findDataFile("brisque_allmodel.dat", false)); }
inline cv::String readbrisquerange() { return readfile(cvtest::findDataFile("brisque_allrange.dat", false)); }

// instantiates a brisque object for testing
inline cv::Ptr<quality::QualityBRISQUE> create_brisque()
{
    return quality::QualityBRISQUE::create(
        readbrisquemodel()
        , readbrisquerange()
    );
}

// static method
TEST(TEST_CASE_NAME, static_ )
{
    std::vector<cv::Mat> qMats = {};
    quality_expect_near(
        quality::QualityBRISQUE::compute( readbrisquemodel(), readbrisquerange(), get_testfile_1a(), qMats), BRISQUE_EXPECTED_1 
    );

    EXPECT_EQ(qMats.size(), 1U);
}

// single channel, instance method, with and without opencl
TEST(TEST_CASE_NAME, single_channel )
{
    auto fn = []() { quality_test(create_brisque(), get_testfile_1a(), BRISQUE_EXPECTED_1); };
    OCL_OFF( fn() );
    OCL_ON( fn() );
}

// multi-channel
TEST(TEST_CASE_NAME, multi_channel)
{
    quality_test(create_brisque(), get_testfile_2a(), BRISQUE_EXPECTED_2);
}

// multi-frame test
TEST(TEST_CASE_NAME, multi_frame)
{
    // result mse == average of all frames
    cv::Scalar expected;
    cv::add(BRISQUE_EXPECTED_1, BRISQUE_EXPECTED_2, expected);
    expected /= 2.;

    quality_test(create_brisque(), get_testfile_1a2a(), expected, 2 );
}

// internal a/b test
/*
TEST(TEST_CASE_NAME, performance)
{
    auto ref = get_testfile_1a();

    quality_performance_test("BRISQUE", [&]() { cv::quality::QualityBRISQUE::compute(ref, cv::noArray()); });
}
*/
}
} // namespace