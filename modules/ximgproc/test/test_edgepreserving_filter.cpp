// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
//                    Created by Simon Reich
//
#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(ximgproc_EdgepreservingFilter, regression)
{
    // Load original image
    std::string filename = string(cvtest::TS::ptr()->get_data_path()) + "perf/320x260.png";
    cv::Mat src, dst, noise, original = imread(filename, 1);

    ASSERT_FALSE(original.empty()) << "Could not load input image " << filename;
    ASSERT_EQ(3, original.channels()) << "Load color input image " << filename;

    // add noise
    noise = Mat(original.size(), original.type());
    cv::randn(noise, 0, 5);
    src = original + noise;

    // Filter
    int kernel = 9;
    double threshold = 20;
    ximgproc::edgePreservingFilter(src, dst, kernel, threshold);

    double psnr = cvtest::PSNR(original, dst);
    //printf("psnr=%.2f\n", psnr);
    ASSERT_LT(psnr, 25.0);
}

}} // namespace
