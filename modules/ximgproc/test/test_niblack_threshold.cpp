// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(ximgproc_niBlackThreshold, sauvola)
{
    Mat src = (Mat_<uchar>(3, 3) << 1, 1, 1, 2, 2, 2, 3, 3, 3);
    Mat dst;
    cv::ximgproc::niBlackThreshold(src, dst, 255, THRESH_BINARY, 3, 1, BINARIZATION_SAUVOLA, 1);

    EXPECT_EQ(CV_8U, dst.type());
    EXPECT_EQ(3, dst.rows);
    EXPECT_EQ(3, dst.cols);

    EXPECT_EQ(0, dst.at<uchar>(0, 0));
    EXPECT_EQ(0, dst.at<uchar>(0, 1));
    EXPECT_EQ(0, dst.at<uchar>(0, 2));
    EXPECT_EQ(0, dst.at<uchar>(1, 0));
    EXPECT_EQ(0, dst.at<uchar>(1, 1));
    EXPECT_EQ(0, dst.at<uchar>(1, 2));
    EXPECT_EQ(255, dst.at<uchar>(2, 0));
    EXPECT_EQ(255, dst.at<uchar>(2, 1));
    EXPECT_EQ(255, dst.at<uchar>(2, 2));
}

}} // namespace
