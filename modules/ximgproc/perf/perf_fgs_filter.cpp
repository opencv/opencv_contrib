// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple<MatType, MatType, Size> FGSParams;

typedef TestBaseWithParam<FGSParams> FGSFilterPerfTest;

PERF_TEST_P( FGSFilterPerfTest, perf,
             Combine(
                 Values(CV_8UC1, CV_8UC3),
                 Values(CV_8UC1, CV_8UC3, CV_16SC1, CV_16SC3, CV_32FC1, CV_32FC3),
                 Values(sz720p)
             )
)
{
    RNG rng(0);

    FGSParams params = GetParam();
    ElemType guideType   = get<0>(params);
    ElemType srcType     = get<1>(params);
    Size sz         = get<2>(params);

    Mat guide(sz, guideType);
    Mat src(sz, srcType);
    Mat dst(sz, srcType);

    declare.in(guide, src, WARMUP_RNG).out(dst);

    TEST_CYCLE_N(10)
    {
        double lambda = rng.uniform(500.0, 10000.0);
        double sigma  = rng.uniform(1.0, 100.0);
        fastGlobalSmootherFilter(guide,src,dst,lambda,sigma);
    }

    SANITY_CHECK_NOTHING();
}

}} // namespace
