// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"
namespace opencv_test { namespace {

TEST(xphoto_inpainting, regression)
{
    string original_path = cvtest::findDataFile("cv/shared/lena.png");
    string mask_path = cvtest::findDataFile("cv/inpaint/mask.png");

    Mat original = imread(original_path);
    Mat mask = imread(mask_path);

    ASSERT_FALSE(original.empty()) << "Could not load input image " << original_path;
    ASSERT_FALSE(mask.empty()) << "Could not load error mask " << mask_path;

    Mat mask_;
    bitwise_not(mask, mask_);
    mask_ = mask_ / 255;
    Mat mask_resized;
    if (mask.rows != original.rows || mask.cols != original.cols)
    {
        resize(mask_, mask_resized, original.size(), 0.0, 0.0, cv::INTER_NEAREST);
    }
    Mat reconstructed_fast, reconstructed_best;
    Mat im_distorted = original.mul(mask_resized);
    Mat mask_1ch;
    if (mask.channels() == 3)
    {
        cvtColor(mask_resized, mask_1ch, cv::COLOR_BGR2GRAY);
    }
    cv::xphoto::inpaint(im_distorted, mask_1ch, reconstructed_fast, cv::xphoto::INPAINT_FSR_FAST);
    cv::xphoto::inpaint(im_distorted, mask_1ch, reconstructed_best, cv::xphoto::INPAINT_FSR_BEST);
    double adiff_psnr_fast = cvtest::PSNR(original, reconstructed_fast);
    double adiff_psnr_best = cvtest::PSNR(original, reconstructed_best);
    ASSERT_GT(adiff_psnr_fast, 39.5);
    ASSERT_GT(adiff_psnr_best, 39.6);
}

}} // namespace
