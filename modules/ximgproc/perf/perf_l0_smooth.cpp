// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple<Size, MatType, int> L0SmoothTestParam;
typedef TestBaseWithParam<L0SmoothTestParam> L0SmoothTest;

PERF_TEST_P(L0SmoothTest, perf,
    Combine(
    SZ_TYPICAL,
    Values(CV_8U, CV_16U, CV_32F, CV_64F),
    Values(1, 3))
)
{
    L0SmoothTestParam params = GetParam();
    Size sz         = get<0>(params);
    int depth       = get<1>(params);
    int srcCn        = get<2>(params);

    Mat src(sz, CV_MAKE_TYPE(depth, srcCn));
    Mat dst(sz, src.type());

    declare.in(src, WARMUP_RNG).out(dst);

    RNG rnd(sz.height + depth + srcCn);
    double lambda = rnd.uniform(0.01, 0.05);
    double kappa = rnd.uniform(1.0, 3.0);

    TEST_CYCLE_N(1)
    {
        l0Smooth(src, dst, lambda, kappa);
    }

    SANITY_CHECK_NOTHING();
}


}} // namespace
