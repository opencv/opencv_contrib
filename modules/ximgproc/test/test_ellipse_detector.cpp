// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

using namespace cv;
using namespace std;

namespace opencv_test { namespace {


TEST(EllipseDetectorTest, EllipsesOnly)
{
    string picture_name = "cv/imgproc/stuff.jpg";
    string filename = cvtest::TS::ptr()->get_data_path() + picture_name;
    Mat src = imread(filename, IMREAD_GRAYSCALE);
    EXPECT_FALSE(src.empty()) << "Invalid test image: " << filename;

    vector<Vec6f> ells;
    ximgproc::ellipseDetector(src, ells, 0.7f, 0.7f, 0.02f);

    EXPECT_EQ(ells.size(), size_t(3)) << "Should find 3 ellipses";
}

TEST(EllipseDetectorTest, ManySmallEllipses)
{
    string picture_name = "cv/imgproc/beads.jpg";
    string filename = cvtest::TS::ptr()->get_data_path() + picture_name;
    Mat src = imread(filename, IMREAD_GRAYSCALE);
    EXPECT_FALSE(src.empty()) << "Invalid test image: " << filename;

    vector<Vec6f> ells;
    ximgproc::ellipseDetector(src, ells, 0.05f, 0.7f, 0.005f);

    EXPECT_GT(ells.size(), size_t(600)) << "Should find a lot of ellipses";
}

}}