// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test
{
namespace
{

TEST(CV_ccmUtils, test_gamma_correction)
{
    Mat x = (Mat_<double>(4, 3) <<
            0.8, -0.5, 0.6,
            0.2, 0.9, -0.9,
            1. , -0.2 , 0.4,
            -0.4, 0.1, 0.3);
    Mat y = (Mat_<double>(4, 3) <<
            0.6120656, -0.21763764, 0.32503696,
            0.02899119, 0.79311017, -0.79311017,
            1., -0.02899119, 0.13320851,
            -0.13320851, 0.00630957, 0.07074028);
    ASSERT_MAT_NEAR(gammaCorrection(x, 2.2), y, 1e-4);
}

TEST(CV_ccmUtils, test_saturate)
{
    Mat x = (Mat_<Vec3d>(5, 1) <<
            Vec3d(0., 0.5, 0.),
            Vec3d(0., 0.3, 0.4),
            Vec3d(0.3, 0.8, 0.4),
            Vec3d(0.7, 0.6, 0.2),
            Vec3d(1., 0.8, 0.5));
    Mat y = (Mat_<bool>(5, 1) <<false, false, true, true, false);
    ASSERT_MAT_NEAR(saturate(x, 0.2, 0.8), y, 0.0);
}

TEST(CV_ccmUtils, test_rgb2gray)
{
    Mat x = (Mat_<Vec3d>(1, 1) << Vec3d(0.2, 0.3, 0.4));
    Mat y = (Mat_<double>(1, 1) <<0.28596);
	std::cout << x << std::endl;
	std::cout << "vs" << std::endl;
	std::cout << y << std::endl;
    ASSERT_MAT_NEAR(rgb2gray(x), y, 1e-4);
}

} // namespace
} // namespace opencv_test
