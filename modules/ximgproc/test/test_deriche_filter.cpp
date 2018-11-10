// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(ximgproc_DericheFilter, regression)
{
    Mat img = Mat::zeros(64, 64, CV_8UC3);
    Mat res = Mat::zeros(64, 64, CV_32FC3);
    img.at<Vec3b>(31, 31) = Vec3b(1, 2, 4);
    double a = 0.5;
    double w = 0.0005;
    Mat dst;

    ximgproc::GradientDericheX(img, dst, a, w);
    double c = pow(1 - exp(-a), 2.0) * exp(a);
    double k = pow(a*(1 - exp(-a)), 2.0) / (1 + 2 * a*exp(-a) - exp(-2 * a));
    for (int i = 0; i < img.rows; i++)
    {
        double n = -31 + i;
        for (int j = 0; j < img.cols; j++)
        {
	        double m = -31 + j;
	        double x = -c * exp(-a * fabs(m))*sin(w*m);
	        x = x * (k*(a*sin(w*fabs(n)) + w * cos(w*fabs(n)))*exp(-a * fabs(n))) / (a*a + w * w);
	        x = x / (w*w);
	        float xx=static_cast<float>(x);
	        res.at<Vec3f>(i, j) = Vec3f(xx, 2 * xx, 4 * xx);
        }
    }
    EXPECT_LE(cv::norm(res, dst, NORM_INF), 1e-5);

    Mat dst2;
    ximgproc::GradientDericheY(img, dst2, a, w);
    cv::transpose(dst2, dst2);
    EXPECT_LE(cv::norm(dst2, dst, NORM_INF), 1e-5);
}
}
} // namespace
