// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(ximgproc_SuperpixelSLIC, smoke)
{
    Mat img = imread(cvtest::findDataFile("cv/shared/lena.png"), IMREAD_COLOR);
    Mat labImg;
    cvtColor(img, labImg, COLOR_BGR2Lab);
    Ptr< SuperpixelSLIC> slic = createSuperpixelSLIC(labImg);
    slic->iterate(5);
    Mat outLabels;
    slic->getLabels(outLabels);
    EXPECT_FALSE(outLabels.empty());
    int numSuperpixels = slic->getNumberOfSuperpixels();
    EXPECT_GT(numSuperpixels, 0);
}

}} // namespace
