// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(XFeatures2d_PCTSignatures, get_sample_count)
{
    std::vector<Point2f> initSamplingPoints;
    for (int i = 0; i < 10; i++)
    {
        initSamplingPoints.push_back(Point2f(0.1f * i, 0.1f * i));
    }

    Ptr<PCTSignatures> pctSignatures = PCTSignatures::create(initSamplingPoints, 5);
    ASSERT_FALSE(pctSignatures.empty());

    ASSERT_EQ((int)initSamplingPoints.size(), pctSignatures->getSampleCount());
    ASSERT_NE(pctSignatures->getSampleCount(), pctSignatures->getGrayscaleBits());
}

}} // namespace
