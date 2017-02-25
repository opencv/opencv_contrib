// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

using namespace cv;
using namespace cv::ximgproc;

namespace {

static int createTestImage(Mat& src)
{
    src = Mat::zeros(Size(256, 256), CV_8UC1);
    for (int x = 50; x < src.cols - 50; x += 50)
    {
        cv::circle(src, Point(x, x/2), 30 + x/2, Scalar(255), 5);
    }
    int src_pixels = countNonZero(src);
    EXPECT_GT(src_pixels, 0);
    return src_pixels;
}

TEST(ximpgroc_Thinning, simple_ZHANGSUEN)
{
    Mat src;
    int src_pixels = createTestImage(src);

    Mat dst;
    thinning(src, dst, THINNING_ZHANGSUEN);
    int dst_pixels = countNonZero(dst);
    EXPECT_LE(dst_pixels, src_pixels);

#if 0
    imshow("src", src); imshow("dst", dst); waitKey();
#endif
}

TEST(ximpgroc_Thinning, simple_GUOHALL)
{
    Mat src;
    int src_pixels = createTestImage(src);

    Mat dst;
    thinning(src, dst, THINNING_GUOHALL);
    int dst_pixels = countNonZero(dst);
    EXPECT_LE(dst_pixels, src_pixels);

#if 0
    imshow("src", src); imshow("dst", dst); waitKey();
#endif
}

}
