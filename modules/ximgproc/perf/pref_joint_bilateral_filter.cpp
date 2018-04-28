// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple<double, Size, MatType, int, int> JBFTestParam;
typedef TestBaseWithParam<JBFTestParam> JointBilateralFilterTest;

PERF_TEST_P(JointBilateralFilterTest, perf,
    Combine(
    Values(4.0, 10.0),
    SZ_TYPICAL,
    Values(CV_8U, CV_32F),
    Values(1, 3),
    Values(1, 3))
)
{
    JBFTestParam params = GetParam();
    double sigmaS   = get<0>(params);
    Size sz         = get<1>(params);
    int depth       = get<2>(params);
    int jCn         = get<3>(params);
    int srcCn       = get<4>(params);

    Mat joint(sz, CV_MAKE_TYPE(depth, jCn));
    Mat src(sz, CV_MAKE_TYPE(depth, srcCn));
    Mat dst(sz, src.type());

    declare.in(joint, src, WARMUP_RNG).out(dst);

    RNG rnd(cvRound(10*sigmaS) + sz.height + depth + jCn + srcCn);
    double sigmaC = rnd.uniform(1.0, 255.0);

    TEST_CYCLE_N(1)
    {
        jointBilateralFilter(joint, src, dst, 0, sigmaC, sigmaS);
    }

    SANITY_CHECK_NOTHING();
}

}} // namespace
