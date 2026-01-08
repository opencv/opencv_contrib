// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

using namespace cv::xphoto;

#ifdef OPENCV_ENABLE_NONFREE

void loadImage(string path, Mat &img)
{
    img = imread(path, -1);
    ASSERT_FALSE(img.empty()) << "Could not load input image " << path;
}

void checkEqual(Mat img0, Mat img1, double threshold, const string& name)
{
    double max = 1.0;
    minMaxLoc(abs(img0 - img1), NULL, &max);
    ASSERT_FALSE(max > threshold) << "max=" << max << " threshold=" << threshold << " method=" << name;
}

TEST(Photo_Tonemap, Durand_regression)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "cv/hdr/tonemap/";

    Mat img, expected, result;
    loadImage(test_path + "image.hdr", img);
    float gamma = 2.2f;

    Ptr<TonemapDurand> durand = createTonemapDurand(gamma);
    durand->process(img, result);
    loadImage(test_path + "durand.png", expected);
    result.convertTo(result, CV_8UC3, 255);
    checkEqual(result, expected, 3, "Durand");
}

TEST(Photo_Tonemap, Durand_property_regression)
{
    const float gamma = 1.0f;
    const float contrast = 2.0f;
    const float saturation = 3.0f;
    const float sigma_color = 4.0f;
    const float sigma_space = 5.0f;

    const Ptr<TonemapDurand> durand1 = createTonemapDurand(gamma, contrast, saturation, sigma_color, sigma_space);
    ASSERT_EQ(gamma, durand1->getGamma());
    ASSERT_EQ(contrast, durand1->getContrast());
    ASSERT_EQ(saturation, durand1->getSaturation());
    ASSERT_EQ(sigma_space, durand1->getSigmaSpace());
    ASSERT_EQ(sigma_color, durand1->getSigmaColor());

    const Ptr<TonemapDurand> durand2 = createTonemapDurand();
    durand2->setGamma(gamma);
    durand2->setContrast(contrast);
    durand2->setSaturation(saturation);
    durand2->setSigmaColor(sigma_color);
    durand2->setSigmaSpace(sigma_space);
    ASSERT_EQ(gamma, durand2->getGamma());
    ASSERT_EQ(contrast, durand2->getContrast());
    ASSERT_EQ(saturation, durand2->getSaturation());
    ASSERT_EQ(sigma_color, durand2->getSigmaColor());
    ASSERT_EQ(sigma_space, durand2->getSigmaSpace());
}

#endif // OPENCV_ENABLE_NONFREE

}} // namespace
