// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(ximgproc_letterboxResize, regression)
{

    string folder = string(cvtest::TS::ptr()->get_data_path()) + "cv/shared/";
    string img_path = folder + "fruits.png";

    Mat original = imread(img_path, IMREAD_COLOR);

    ASSERT_FALSE(original.empty()) << "Could not load input image " << img_path;

    Mat result;
    Size dsize;
    dsize.width = 400;
    dsize.height = 200;

    int interpolation = 0;
    int borderType = 0;
    Scalar value = {128, 128, 128};

    ximgproc::letterboxResize(original, result, dsize, interpolation, borderType, value);

    Size rsize = result.size();

    ASSERT_TRUE(rsize.width == dsize.width);
    ASSERT_TRUE(rsize.height == dsize.height);
}

}} // namespace
