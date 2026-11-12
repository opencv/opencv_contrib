// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(HfsSegment_CPU, SyntheticQuadrantsSparseBoundarySegmentation)
{
    RNG& rng = theRNG();
    const int H = 96, W = 128;
    Mat img(H, W, CV_8UC3);
    rng.fill(img, RNG::UNIFORM, 0, 12);

    img(Rect(0, 0, W / 2, H / 2)) += Scalar(200, 30, 30);
    img(Rect(W / 2, 0, W / 2, H / 2)) += Scalar(30, 200, 30);
    img(Rect(0, H / 2, W / 2, H / 2)) += Scalar(30, 30, 200);
    img(Rect(W / 2, H / 2, W / 2, H / 2)) += Scalar(200, 200, 30);

    Ptr<HfsSegment> seg = HfsSegment::create(H, W, 0.08f, 30, 0.28f, 60, 0.6f, 8, 5);
    Mat labelMap = seg->performSegmentCpu(img, false);
    ASSERT_EQ(Size(W, H), labelMap.size());
    ASSERT_EQ(CV_16UC1, labelMap.type());

    double minLabel = 0.0, maxLabel = 0.0;
    minMaxLoc(labelMap, &minLabel, &maxLabel);
    EXPECT_GE(minLabel, 1.0);
    EXPECT_GE(maxLabel, 2.0);
    EXPECT_NE(labelMap.at<ushort>(16, 16), labelMap.at<ushort>(H - 16, W - 16));

    Mat drawn = seg->performSegmentCpu(img, true);
    ASSERT_EQ(Size(W, H), drawn.size());
    ASSERT_EQ(CV_8UC3, drawn.type());
}

}} // namespace
