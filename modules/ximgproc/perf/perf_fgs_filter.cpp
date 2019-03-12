// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

CV_ENUM(GuideTypes, CV_8UC1, CV_8UC3);
CV_ENUM(SrcTypes, CV_8UC1, CV_8UC3, CV_16SC1, CV_16SC3, CV_32FC1, CV_32FC3);
typedef tuple<GuideTypes, SrcTypes, Size> FGSParams;

typedef TestBaseWithParam<FGSParams> FGSFilterPerfTest;

PERF_TEST_P( FGSFilterPerfTest, perf, Combine(GuideTypes::all(), SrcTypes::all(), Values(sz720p)) )
{
    RNG rng(0);

    FGSParams params = GetParam();
    int guideType   = get<0>(params);
    int srcType     = get<1>(params);
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
