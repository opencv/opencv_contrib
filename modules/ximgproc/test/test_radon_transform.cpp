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

    ASSERT_EQ(radon.rows, 363);
    ASSERT_EQ(radon.cols, 180);

    cv::ximgproc::RadonTransform(src, radon, 1, 0, 180, true);

    ASSERT_EQ(radon.rows, 256);
    ASSERT_EQ(radon.cols, 180);
}

TEST(RadonTransformTest, output_type)
{
    Mat src_int(Size(256, 256), CV_8U, Scalar(0));
    circle(src_int, Point(128, 128), 64, Scalar(255), FILLED);
    Mat radon, radon_norm;
    cv::ximgproc::RadonTransform(src_int, radon);
    cv::ximgproc::RadonTransform(src_int, radon_norm, 1, 0, 180, false, true);

    ASSERT_EQ(radon.type(), CV_32SC1);
    ASSERT_EQ(radon_norm.type(), CV_8U);

    Mat src_float(Size(256, 256), CV_32FC1, Scalar(0));
    Mat src_double(Size(256, 256), CV_32FC1, Scalar(0));
    cv::ximgproc::RadonTransform(src_float, radon);
    cv::ximgproc::RadonTransform(src_float, radon_norm, 1, 0, 180, false, true);
    ASSERT_EQ(radon.type(), CV_64FC1);
    ASSERT_EQ(radon_norm.type(), CV_8U);
    cv::ximgproc::RadonTransform(src_double, radon);
    ASSERT_EQ(radon.type(), CV_64FC1);
    ASSERT_EQ(radon_norm.type(), CV_8U);
}

TEST(RadonTransformTest, accuracy_by_pixel)
{
    Mat src(Size(256, 256), CV_8U, Scalar(0));
    circle(src, Point(128, 128), 64, Scalar(255), FILLED);
    Mat radon;
    cv::ximgproc::RadonTransform(src, radon);

    ASSERT_EQ(radon.at<int>(0, 0), 0);

    ASSERT_GT(radon.at<int>(128, 128), 18000);
    ASSERT_LT(radon.at<int>(128, 128), 19000);
}

TEST(RadonTransformTest, accuracy_uchar)
{
    Mat src(Size(10, 10), CV_8UC1, Scalar(1));
    cv::Mat radon;
    ximgproc::RadonTransform(src, radon, 45, 0, 180, false, false);

    ASSERT_EQ(sum(radon.col(0))[0], 100);
}

TEST(RadonTransformTest, accuracy_float)
{
    Mat src(Size(10, 10), CV_32FC1, Scalar(1.1));
    cv::Mat radon;
    ximgproc::RadonTransform(src, radon, 45, 0, 180, false, false);

    ASSERT_GT(sum(radon.col(0))[0], 109);
    ASSERT_LT(sum(radon.col(0))[0], 111);
}

} }
