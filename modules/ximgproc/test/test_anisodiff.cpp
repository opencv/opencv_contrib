// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(ximgproc_AnisotropicDiffusion, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "cv/shared/";
    string original_path = folder + "fruits.png";

    Mat original = imread(original_path, IMREAD_COLOR);

    ASSERT_FALSE(original.empty()) << "Could not load input image " << original_path;
    ASSERT_EQ(3, original.channels()) << "Load color input image " << original_path;

    Mat result;
    float alpha = 1.0f;
    float K = 0.02f;
    int niters = 10;
    ximgproc::anisotropicDiffusion(original, result, alpha, K, niters);

    double adiff_psnr = cvtest::PSNR(original, result);
    //printf("psnr=%.2f\n", adiff_psnr);
    ASSERT_GT(adiff_psnr, 25.0);
}

}} // namespace
