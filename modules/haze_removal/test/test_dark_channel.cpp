// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
namespace opencv_test {
namespace {

TEST(dark_channel_haze_removal_test, accuracy)
{
    string inputFilepath = cvtest::findDataFile("haze_removal/input.png");
    string expectedOutputFilepath = cvtest::findDataFile("haze_removal/expectedOutput.png");

    cv::Mat input, expectedOutput, output;
    input = cv::imread(inputFilepath);
    expectedOutput = cv::imread(expectedOutputFilepath);

    ASSERT_FALSE(input.empty());
    ASSERT_FALSE(expectedOutput.empty());

    cv::haze_removal::darkChannelPriorHazeRemoval(input, output);

    ASSERT_EQ((int)output.rows, (int)expectedOutput.rows);
    ASSERT_EQ((int)output.cols, (int)expectedOutput.cols);

    EXPECT_LE(cvtest::norm(output, expectedOutput, NORM_INF), 5);
}

}} // opencv_test::namespace::
