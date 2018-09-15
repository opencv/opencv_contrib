// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple<MatType, MatType, Size> GFParams;

typedef TestBaseWithParam<GFParams> GuidedFilterPerfTest;

PERF_TEST_P( GuidedFilterPerfTest, perf, Combine
               (
                   Values(CV_8UC1, CV_8UC3, CV_32FC1, CV_32FC3),
                   Values(CV_8UC1, CV_8UC3, CV_32FC1, CV_32FC3),
                   Values(sz1080p, sz2K)
               )
           )
{
    RNG rng(0);

    GFParams params = GetParam();
    ElemType guideType   = get<0>(params);
    ElemType srcType = get<1>(params);
    Size sz         = get<2>(params);

    Mat guide(sz, guideType);
    Mat src(sz, srcType);
    Mat dst(sz, srcType);

    declare.in(guide, src, WARMUP_RNG).out(dst);

    TEST_CYCLE_N(3)
    {
        int radius = rng.uniform(5, 30);
        double eps = rng.uniform(0.1, 1e5);
        guidedFilter(guide, src, dst, radius, eps);
    }

    SANITY_CHECK_NOTHING();
}

}} // namespace
