// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple<MatDepth, int, Size> RDFParams;
typedef TestBaseWithParam<RDFParams> RidgeDetectionFilterPerfTest;

PERF_TEST_P(RidgeDetectionFilterPerfTest, perf, Combine(
        Values(CV_32F),
        Values(3),
        SZ_TYPICAL
))
{
    RDFParams params = GetParam();
    ElemDepth ddepth = get<0>(params);
    int ksize = get<1>(params);
    Size sz = get<2>(params);

    Mat src(sz, CV_MAKETYPE(ddepth, 1));
    Mat out(sz, src.type());

    declare.in(src).out(out);

    Ptr<RidgeDetectionFilter> rdf = RidgeDetectionFilter::create(ddepth,1, 1, ksize);

    TEST_CYCLE() rdf->getRidgeFilteredImage(src, out);

    SANITY_CHECK_NOTHING();
}

}} // namespace
