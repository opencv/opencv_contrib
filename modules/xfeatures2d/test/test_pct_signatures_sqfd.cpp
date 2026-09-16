// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(XFeatures2d_PCTSignaturesSQFD, compute_quadratic_form_distances_multiple_signatures)
{
    Ptr<PCTSignaturesSQFD> sqfd = PCTSignaturesSQFD::create();
    ASSERT_FALSE(sqfd.empty());

    Mat source = Mat_<float>({1, 8}, {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});

    std::vector<Mat> imageSignatures;
    imageSignatures.push_back(Mat_<float>({1, 8}, {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}));
    imageSignatures.push_back(Mat_<float>({1, 8}, {1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f}));
    imageSignatures.push_back(Mat_<float>({1, 8}, {1.f, 2.f, 2.f, 0.f, 0.f, 0.f, 0.f, 0.f}));

    std::vector<float> distances;
    sqfd->computeQuadraticFormDistances(source, imageSignatures, distances);
    ASSERT_EQ(imageSignatures.size(), distances.size());

    // Each entry (including index > 0) must match the same computation done directly,
    // i.e. none of the non-first signatures may be treated as empty/garbage.
    for (size_t i = 0; i < imageSignatures.size(); i++)
    {
        float expected = sqfd->computeQuadraticFormDistance(source, imageSignatures[i]);
        EXPECT_NEAR(expected, distances[i], 1e-5) << "signature index " << i;
    }
}

}} // namespace
