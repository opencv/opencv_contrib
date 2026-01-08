// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple<bool, Size, int, int, MatType> AMPerfTestParam;
typedef TestBaseWithParam<AMPerfTestParam> AdaptiveManifoldPerfTest;

PERF_TEST_P( AdaptiveManifoldPerfTest, perf,
    Combine(
    Values(true, false),            //adjust_outliers flag
    Values(sz1080p, sz720p),        //size
    Values(1, 3, 8),                //joint channels num
    Values(1, 3),                   //source channels num
    Values(CV_8U, CV_32F)           //source and joint depth
    )
)
{
    AMPerfTestParam params = GetParam();
    bool adjustOutliers = get<0>(params);
    Size sz             = get<1>(params);
    int jointCnNum      = get<2>(params);
    int srcCnNum        = get<3>(params);
    int depth           = get<4>(params);

    Mat joint(sz, CV_MAKE_TYPE(depth, jointCnNum));
    Mat src(sz, CV_MAKE_TYPE(depth, srcCnNum));
    Mat dst(sz, CV_MAKE_TYPE(depth, srcCnNum));

    declare.in(joint, src, WARMUP_RNG).out(dst);

    double sigma_s = 16;
    double sigma_r = 0.5;
    TEST_CYCLE_N(3)
    {
        Mat res;
        amFilter(joint, src, res, sigma_s, sigma_r, adjustOutliers);

        //at 5th cycle sigma_s will be five times more and tree depth will be 5
        sigma_s *= 1.38;
        sigma_r /= 1.38;
    }

    SANITY_CHECK_NOTHING();
}

}} // namespace
