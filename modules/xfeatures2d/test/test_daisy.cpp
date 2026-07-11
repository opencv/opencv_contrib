// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(XFeatures2d_DAISY, compute_roi_matches_full_image)
{
    // Deterministic, non-flat texture so descriptors are not trivially zero.
    Mat image(100, 100, CV_8UC1);
    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            image.at<uchar>(y, x) = static_cast<uchar>((x * 3 + y * 7) % 256);
        }
    }

    Ptr<DAISY> daisy = DAISY::create();
    ASSERT_FALSE(daisy.empty());

    Mat fullDescriptors;
    daisy->compute(image, fullDescriptors);
    ASSERT_EQ(image.rows * image.cols, fullDescriptors.rows);

    // roi offset away from the image origin, well clear of the border.
    Rect roi(30, 25, 10, 8);
    Mat roiDescriptors;
    daisy->compute(image, roi, roiDescriptors);
    ASSERT_EQ(roi.width * roi.height, roiDescriptors.rows);

    // Sampling at a given absolute pixel must produce the same descriptor regardless
    // of roi, only its storage row (roi-relative) differs. Before the fix, the roi
    // descriptors buffer was indexed with the full-image absolute offset, which for
    // any roi not starting at (0,0) writes past the end of the roi-sized buffer and
    // leaves the correct relative rows at their zero-initialized value.
    for (int y = roi.y; y < roi.y + roi.height; y++)
    {
        for (int x = roi.x; x < roi.x + roi.width; x++)
        {
            int fullIndex = y * image.cols + x;
            int roiIndex = (y - roi.y) * roi.width + (x - roi.x);

            double diff = cv::norm(fullDescriptors.row(fullIndex), roiDescriptors.row(roiIndex), NORM_INF);
            EXPECT_NEAR(0.0, diff, 1e-5) << "pixel (" << x << ", " << y << ")";
        }
    }
}

}} // namespace
