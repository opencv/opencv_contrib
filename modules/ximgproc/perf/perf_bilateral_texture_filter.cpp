// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple<int, double, double, Size, MatType, int> BTFTestParam;
typedef TestBaseWithParam<BTFTestParam> BilateralTextureFilterTest;

PERF_TEST_P(BilateralTextureFilterTest, perf,
    Combine(
    Values(2),
    Values(0.5),
    Values(0.5),
    SZ_TYPICAL,
    Values(CV_8U, CV_32F),
    Values(1, 3))
)
{
    BTFTestParam params = GetParam();
    int fr            = get<0>(params);
    double sigmaAlpha = get<1>(params);
    double sigmaAvg   = get<2>(params);
    Size sz           = get<3>(params);
    int depth         = get<4>(params);
    int srcCn         = get<5>(params);

    Mat src(sz, CV_MAKE_TYPE(depth,srcCn));
    Mat dst(sz, src.type());

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE_N(1)
    {
        bilateralTextureFilter(src, dst, fr, 1, sigmaAlpha, sigmaAvg);
    }

    SANITY_CHECK_NOTHING();
}

}} // namespace
