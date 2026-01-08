// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace  {

typedef tuple<Size, MatType, int> FindEllipsesTestParam;
typedef TestBaseWithParam<FindEllipsesTestParam> FindEllipsesTest;

PERF_TEST_P(FindEllipsesTest, perf, Combine(SZ_TYPICAL, Values(CV_8U), Values(1, 3)))
{
    FindEllipsesTestParam params = GetParam();
    Size sz = get<0>(params);
    int matType = get<1>(params);
    int srcCn = get<2>(params);

    Mat src(sz, CV_MAKE_TYPE(matType, srcCn));
    Mat dst(sz, CV_32FC(6));

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() findEllipses(src, dst, 0.7f, 0.5f, 0.05f);

    SANITY_CHECK_NOTHING();
}
}} // namespace