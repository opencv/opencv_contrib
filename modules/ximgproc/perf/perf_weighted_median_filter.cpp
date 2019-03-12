// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple<Size, MatType, int, int, int, WMFWeightType> WMFTestParam;
typedef TestBaseWithParam<WMFTestParam> WeightedMedianFilterTest;

PERF_TEST_P(WeightedMedianFilterTest, perf,
    Combine(
    Values(szODD, szQVGA),
    Values(CV_8U, CV_32F),
    Values(1, 3),
    Values(1, 3),
    Values(3, 5),
    Values(WMF_EXP, WMF_COS))
)
{
    RNG rnd(1);

    WMFTestParam params = GetParam();

    double sigma   = rnd.uniform(20.0, 30.0);
    Size sz         = get<0>(params);
    int srcDepth       = get<1>(params);
    int jCn         = get<2>(params);
    int srcCn       = get<3>(params);
    int r = get<4>(params);
    WMFWeightType weightType = get<5>(params);

    Mat joint(sz, CV_MAKE_TYPE(CV_8U, jCn));
    Mat src(sz, CV_MAKE_TYPE(srcDepth, srcCn));
    Mat dst(sz, src.type());

    declare.in(joint, src, WARMUP_RNG).out(dst);

    TEST_CYCLE_N(1)
    {
        weightedMedianFilter(joint, src, dst, r, sigma, weightType);
    }

    SANITY_CHECK_NOTHING();
}


}} // namespace
