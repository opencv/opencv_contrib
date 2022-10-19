// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

using namespace cv;
using namespace std;

namespace opencv_test { namespace {


TEST(FindEllipsesTest, EllipsesOnly)
{
    string picture_name = "cv/imgproc/stuff.jpg";
    string filename = cvtest::TS::ptr()->get_data_path() + picture_name;
    Mat src = imread(filename, IMREAD_GRAYSCALE);
    EXPECT_FALSE(src.empty()) << "Invalid test image: " << filename;

    vector<Vec6f> ells;
    ximgproc::findEllipses(src, ells, 0.7f, 0.5f, 0.01f);

    // number check
    EXPECT_EQ(ells.size(), size_t(3)) << "Should find 3 ellipses";
    // position check
    // first ellipse center
    EXPECT_TRUE((ells[0][0] >= 393) && (ells[0][0] <= 394)) << "First ellipse center x is wrong";
    EXPECT_TRUE((ells[0][1] >= 187) && (ells[0][1] <= 188)) << "First ellipse center y is wrong";
    // second ellipse center
    EXPECT_TRUE((ells[1][0] >= 208) && (ells[1][0] <= 209)) << "Second ellipse center x is wrong";
    EXPECT_TRUE((ells[1][1] >= 307) && (ells[1][1] <= 308)) << "Second ellipse center y is wrong";
    // third ellipse center
    EXPECT_TRUE((ells[2][0] >=  229) && (ells[2][0] <=  230)) << "Third ellipse center x is wrong";
    EXPECT_TRUE((ells[2][1] >=  57) && (ells[2][1] <=  58)) << "Third ellipse center y is wrong";
}
}}