// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple<Size, int> ThinningPerfParam;
typedef TestBaseWithParam<ThinningPerfParam> ThinningPerfTest;

PERF_TEST_P(ThinningPerfTest, perf,
    Combine(
        Values(sz1080p, sz720p, szVGA),
        Values(THINNING_ZHANGSUEN, THINNING_GUOHALL)
    )
)
{
    ThinningPerfParam params = GetParam();
    Size size = get<0>(params);
    int type  = get<1>(params);

    Mat src = Mat::zeros(size, CV_8UC1);
    for (int x = 50; x < src.cols - 50; x += 50)
        cv::circle(src, Point(x, x/2), 30 + x/2, Scalar(255), 5);

    Mat dst;
    TEST_CYCLE()
    {
        thinning(src, dst, type);
    }

    SANITY_CHECK_NOTHING();
}

}} // namespace
