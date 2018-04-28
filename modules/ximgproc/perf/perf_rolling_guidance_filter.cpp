// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple<double, Size, MatType, int> RGFTestParam;
typedef TestBaseWithParam<RGFTestParam> RollingGuidanceFilterTest;

PERF_TEST_P(RollingGuidanceFilterTest, perf,
    Combine(
    Values(2.0, 4.0, 6.0, 10.0),
    SZ_TYPICAL,
    Values(CV_8U, CV_32F),
    Values(1, 3))
)
{
    RGFTestParam params = GetParam();
    double sigmaS   = get<0>(params);
    Size sz         = get<1>(params);
    int depth       = get<2>(params);
    int srcCn       = get<3>(params);

    Mat src(sz, CV_MAKE_TYPE(depth, srcCn));
    Mat dst(sz, src.type());

    declare.in(src, WARMUP_RNG).out(dst);

    RNG rnd(cvRound(10*sigmaS) + sz.height + depth + srcCn);
    double sigmaC = rnd.uniform(1.0, 255.0);
    int iterNum = int(rnd.uniform(1.0, 5.0));

    TEST_CYCLE_N(1)
    {
        rollingGuidanceFilter(src, dst, -1, sigmaC, sigmaS, iterNum);
    }

    SANITY_CHECK_NOTHING();
}


}} // namespace
