// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

using namespace cv;

namespace opencv_test { namespace {


TEST(FindEllipsesTest, EllipsesOnly)
{
    std::string picture_name = "cv/imgproc/stuff.jpg";
    std::string filename = cvtest::TS::ptr()->get_data_path() + picture_name;
    Mat src = imread(filename, IMREAD_GRAYSCALE);
    EXPECT_FALSE(src.empty()) << "Invalid test image: " << filename;

    std::vector<Vec6f> ells;
    ximgproc::findEllipses(src, ells, 0.7f, 0.75f, 0.02f);

    // number check
    EXPECT_EQ(ells.size(), size_t(3)) << "Should find 3 ellipses";

    // position check
    // target centers
    Point2f center_1(226.9f, 57.2f);
    Point2f center_2(393.1f, 187.0f);
    Point2f center_3(208.5f, 307.5f);
    // matching
    for (auto ell: ells) {
        bool has_match = false;
        for (auto c: {center_1, center_2, center_3}) {
            Point2f diff = c - Point2f(ell[0], ell[1]);
            float distance = sqrt(diff.x * diff.x + diff.y * diff.y);
            if (distance < 5.0) {
                has_match = true;
                break;
            }
        }
        EXPECT_TRUE(has_match) << "Wrong ellipse center:" << Point2f(ell[0], ell[1]);
    }
}
}}