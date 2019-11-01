// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {
using namespace xphoto;


static void test_inpainting(const Size inputSize, InpaintTypes mode, double expected_psnr, ImreadModes inputMode = IMREAD_COLOR)
{
    string original_path = cvtest::findDataFile("cv/shared/lena.png");
    string mask_path = cvtest::findDataFile("cv/inpaint/mask.png");

    Mat original_ = imread(original_path, inputMode);
    ASSERT_FALSE(original_.empty()) << "Could not load input image " << original_path;

    Mat mask_ = imread(mask_path, IMREAD_GRAYSCALE);
    ASSERT_FALSE(mask_.empty()) << "Could not load error mask " << mask_path;

    Mat original, mask;
    resize(original_, original, inputSize, 0.0, 0.0, INTER_AREA);
    resize(mask_, mask, inputSize, 0.0, 0.0, INTER_NEAREST);

    Mat mask_valid = (mask == 0);
    Mat im_distorted(inputSize, original.type(), Scalar::all(0));
    original.copyTo(im_distorted, mask_valid);

    Mat reconstructed;
    xphoto::inpaint(im_distorted, mask_valid, reconstructed, mode);

    double adiff_psnr = cvtest::PSNR(original, reconstructed);
    EXPECT_LE(expected_psnr, adiff_psnr);

#if 0
    imshow("original", original);
    imshow("im_distorted", im_distorted);
    imshow("reconstructed", reconstructed);
    std::cout << "adiff_psnr=" << adiff_psnr << std::endl;
    waitKey();
#endif
}

TEST(xphoto_inpaint, smoke_FSR_FAST)  // fast smoke test, input doesn't fit well for tested algorithm
{
    test_inpainting(Size(128, 128), INPAINT_FSR_FAST, 30);
}
TEST(xphoto_inpaint, smoke_FSR_BEST)  // fast smoke test, input doesn't fit well for tested algorithm
{
    applyTestTag(CV_TEST_TAG_LONG);
    test_inpainting(Size(128, 128), INPAINT_FSR_BEST, 30);
}

TEST(xphoto_inpaint, smoke_grayscale_FSR_FAST)  // fast smoke test, input doesn't fit well for tested algorithm
{
    test_inpainting(Size(128, 128), INPAINT_FSR_FAST, 30, IMREAD_GRAYSCALE);
}
TEST(xphoto_inpaint, smoke_grayscale_FSR_BEST)  // fast smoke test, input doesn't fit well for tested algorithm
{
    test_inpainting(Size(128, 128), INPAINT_FSR_BEST, 30, IMREAD_GRAYSCALE);
}


TEST(xphoto_inpaint, regression_FSR_FAST)
{
    test_inpainting(Size(512, 512), INPAINT_FSR_FAST, 39.5);
}
TEST(xphoto_inpaint, regression_FSR_BEST)
{
    applyTestTag(CV_TEST_TAG_VERYLONG);  // add --test_tag_enable=verylong to run this test
    test_inpainting(Size(512, 512), INPAINT_FSR_BEST, 39.6);
}


}} // namespace
