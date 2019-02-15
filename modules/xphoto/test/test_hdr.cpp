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

#endif // OPENCV_ENABLE_NONFREE

}} // namespace
