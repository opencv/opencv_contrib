// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(ximgproc_Thinning, simple_ZHANGSUEN)
{
    string dir = cvtest::TS::ptr()->get_data_path();
    Mat src = imread(dir + "cv/ximgproc/sources/08.png", IMREAD_GRAYSCALE);
    Mat dst,check_img;

    thinning(src, dst, THINNING_ZHANGSUEN);

    check_img = imread(dir + "cv/ximgproc/results/Thinning_ZHANGSUEN.png", IMREAD_GRAYSCALE);
    EXPECT_EQ(0, cvtest::norm(check_img, dst, NORM_INF));

    dst = ~src;
    thinning(dst, dst, THINNING_ZHANGSUEN);

    check_img = imread(dir + "cv/ximgproc/results/Thinning_inv_ZHANGSUEN.png", IMREAD_GRAYSCALE);
    EXPECT_EQ(0, cvtest::norm(check_img, dst, NORM_INF));
}

TEST(ximgproc_Thinning, simple_GUOHALL)
{
    string dir = cvtest::TS::ptr()->get_data_path();
    Mat src = imread(dir + "cv/ximgproc/sources/08.png", IMREAD_GRAYSCALE);
    Mat dst,check_img;

    thinning(src, dst, THINNING_GUOHALL);

    check_img = imread(dir + "cv/ximgproc/results/Thinning_GUOHALL.png", IMREAD_GRAYSCALE);
    EXPECT_EQ(0, cvtest::norm(check_img, dst, NORM_INF));

    dst = ~src;
    thinning(dst, dst, THINNING_GUOHALL);

    check_img = imread(dir + "cv/ximgproc/results/Thinning_inv_GUOHALL.png", IMREAD_GRAYSCALE);
    EXPECT_EQ(0, cvtest::norm(check_img, dst, NORM_INF));
}

}} // namespace
