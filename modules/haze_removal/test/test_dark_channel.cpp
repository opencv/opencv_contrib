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

    for (int r = 0; r < output.rows; r++)
    {
        for (int c = 0; c < output.cols; c++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                EXPECT_NEAR(output.at<cv::Vec3b>(r, c)[channel], expectedOutput.at<cv::Vec3b>(r, c)[channel], 5);
            }
        }
    }
}

}} // opencv_test::namespace::
