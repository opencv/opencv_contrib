// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {
namespace {

typedef tuple<int, Size, int> RLParams;

typedef TestBaseWithParam<RLParams> RLMorphologyPerfTest;

PERF_TEST_P(RLMorphologyPerfTest, perf, Combine(Values(1,7, 21), Values(sz720p, sz2160p),
    Values(MORPH_ERODE, MORPH_DILATE, MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT,MORPH_TOPHAT, MORPH_BLACKHAT)))
{
    RLParams params = GetParam();
    int seSize = get<0>(params);
    Size sz = get<1>(params);
    int op = get<2>(params);

    Mat src(sz, CV_8U);
    Mat thresholded, dstRLE;
    Mat se = rl::getStructuringElement(MORPH_ELLIPSE, cv::Size(2 * seSize + 1, 2 * seSize + 1));

    declare.in(src, WARMUP_RNG);

    TEST_CYCLE_N(4)
    {
        rl::threshold(src, thresholded, 100.0, THRESH_BINARY);
        rl::morphologyEx(thresholded, dstRLE, op, se);
    }

    SANITY_CHECK_NOTHING();
}

}
} // namespace