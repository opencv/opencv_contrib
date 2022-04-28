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

    EXPECT_EQ(363, radon.rows);
    EXPECT_EQ(180, radon.cols);

    cv::ximgproc::RadonTransform(src, radon, 1, 0, 180, true);

    EXPECT_EQ(256, radon.rows);
    EXPECT_EQ(180, radon.cols);
}

TEST(RadonTransformTest, output_type)
{
    Mat src_int(Size(256, 256), CV_8U, Scalar(0));
    circle(src_int, Point(128, 128), 64, Scalar(255), FILLED);
    Mat radon, radon_norm;
    cv::ximgproc::RadonTransform(src_int, radon);
    cv::ximgproc::RadonTransform(src_int, radon_norm, 1, 0, 180, false, true);

    EXPECT_EQ(CV_32SC1, radon.type());
    EXPECT_EQ(CV_8U, radon_norm.type());

    Mat src_float(Size(256, 256), CV_32FC1, Scalar(0));
    Mat src_double(Size(256, 256), CV_32FC1, Scalar(0));
    cv::ximgproc::RadonTransform(src_float, radon);
    cv::ximgproc::RadonTransform(src_float, radon_norm, 1, 0, 180, false, true);
    EXPECT_EQ(CV_64FC1, radon.type());
    EXPECT_EQ(CV_8U, radon_norm.type());
    cv::ximgproc::RadonTransform(src_double, radon);
    EXPECT_EQ(CV_64FC1, radon.type());
    EXPECT_EQ(CV_8U, radon_norm.type());
}

TEST(RadonTransformTest, accuracy_by_pixel)
{
    Mat src(Size(256, 256), CV_8U, Scalar(0));
    circle(src, Point(128, 128), 64, Scalar(255), FILLED);
    Mat radon;
    cv::ximgproc::RadonTransform(src, radon);

    ASSERT_EQ(CV_32SC1, radon.type());

    EXPECT_EQ(0, radon.at<int>(0, 0));

    EXPECT_LT(18000, radon.at<int>(128, 128));
    EXPECT_GT(19000, radon.at<int>(128, 128));
}

TEST(RadonTransformTest, accuracy_uchar)
{
    Mat src(Size(10, 10), CV_8UC1, Scalar(1));
    cv::Mat radon;
    ximgproc::RadonTransform(src, radon, 45, 0, 180, false, false);

    EXPECT_EQ(100, sum(radon.col(0))[0]);
}

TEST(RadonTransformTest, accuracy_float)
{
    Mat src(Size(10, 10), CV_32FC1, Scalar(1.1));
    cv::Mat radon;
    ximgproc::RadonTransform(src, radon, 45, 0, 180, false, false);

    EXPECT_LT(109, sum(radon.col(0))[0]);
    EXPECT_GT(111, sum(radon.col(0))[0]);
}

} }
