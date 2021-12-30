// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

static void runScanSegment(int slices)

{
    Mat img = imread(cvtest::findDataFile("cv/shared/lena.png"), IMREAD_COLOR);
    Mat labImg;
    cvtColor(img, labImg, COLOR_BGR2Lab);
    Ptr<ScanSegment> ss = createScanSegment(labImg.cols, labImg.rows, 500, slices, true);
    ss->iterate(labImg);
    int numSuperpixels = ss->getNumberOfSuperpixels();
    EXPECT_GT(numSuperpixels, 100);
    EXPECT_LE(numSuperpixels, 500);
    Mat res;
    ss->getLabelContourMask(res, false);
    EXPECT_GE(cvtest::norm(res, NORM_L1), 1000000);

    if (cvtest::debugLevel >= 10)
    {
        imshow("ScanSegment", res);
        waitKey();
    }
}

TEST(ximgproc_ScanSegment, smoke) { runScanSegment(1); }
TEST(ximgproc_ScanSegment, smoke4) { runScanSegment(4); }
TEST(ximgproc_ScanSegment, smoke8) { runScanSegment(8); }

}} // namespace
