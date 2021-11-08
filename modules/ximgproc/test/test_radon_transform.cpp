// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test {namespace {

TEST(RadonTransformTest, output_size)
{
    Mat src(Size(256, 256), CV_8U, Scalar(0));
    circle(src, Point(128, 128), 64, Scalar(255), FILLED);
    Mat radon;
    cv::ximgproc::RadonTransform(src, radon);

    ASSERT_EQ(radon.rows, 180);
    ASSERT_EQ(radon.cols, 256);
}

TEST(RadonTransformTest, output_type)
{
    Mat src(Size(256, 256), CV_8U, Scalar(0));
    circle(src, Point(128, 128), 64, Scalar(255), FILLED);
    Mat radon, radon_norm;
    cv::ximgproc::RadonTransform(src, radon);
    cv::ximgproc::RadonTransform(src, radon_norm, 1, 0, 180, false, true);

    ASSERT_EQ(radon.type(), CV_32SC1);
    ASSERT_EQ(radon_norm.type(), CV_8U);
}

TEST(RadonTransformTest, accuracy_by_pixel)
{
    Mat src(Size(256, 256), CV_8U, Scalar(0));
    circle(src, Point(128, 128), 64, Scalar(255), FILLED);
    Mat radon;
    cv::ximgproc::RadonTransform(src, radon);

    ASSERT_EQ(radon.at<int>(0, 0), 0);

    ASSERT_GT(radon.at<int>(128, 128), 32000);
    ASSERT_LT(radon.at<int>(128, 128), 33000);
}

TEST(RadonTransformTest, accuracy_by_col_sum)
{
    Mat src(Size(256, 256), CV_8U, Scalar(0));
    circle(src, Point(128, 128), 64, Scalar(255), FILLED);
    Mat radon;
    cv::ximgproc::RadonTransform(src, radon, 1, 0, 180, false, true);
    Mat sum_col;
    cv::reduce(radon, sum_col, 0, REDUCE_SUM, CV_32SC1);

    ASSERT_EQ(sum_col.at<int>(0), 0);

    ASSERT_GT(sum_col.at<int>(128), 45000);
    ASSERT_LT(sum_col.at<int>(128), 46000);
}

TEST(RadonTransformTest, accuracy_by_row_sum)
{
    Mat src(Size(256, 256), CV_8U, Scalar(0));
    circle(src, Point(128, 128), 64, Scalar(255), FILLED);
    Mat radon;
    cv::ximgproc::RadonTransform(src, radon, 1, 0, 180, false, true);
    Mat sum_row;
    cv::reduce(radon, sum_row, 1, REDUCE_SUM, CV_32SC1);

    ASSERT_LT(abs(sum_row.at<int>(0) - sum_row.at<int>(128)), 1000);

    ASSERT_GT(sum_row.at<int>(0), 25000);
    ASSERT_LT(sum_row.at<int>(0), 26000);
}

} }
