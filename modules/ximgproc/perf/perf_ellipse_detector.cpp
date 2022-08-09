// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace  {

typedef tuple<Size, MatType, int> EllipseDetectorTestParam;
typedef TestBaseWithParam<EllipseDetectorTestParam> EllipseDetectorTest;

PERF_TEST_P(EllipseDetectorTest, perf, Combine(SZ_TYPICAL, Values(CV_8U, CV_16U, CV_32F, CV_64F), Values(1, 3)))
{
    EllipseDetectorTestParam params = GetParam();
    Size sz = get<0>(params);
    int matType = get<1>(params);
    int srcCn = get<2>(params);

    Mat src(sz, CV_MAKE_TYPE(matType, srcCn));
    std::vector<Vec6f> dst;

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE_N(1)
    {
        ellipseDetector(src, dst, 0.7f, 0.5f, 0.05f);
    }

    SANITY_CHECK_NOTHING();
}
}} // namespace