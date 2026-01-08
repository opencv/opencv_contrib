// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple<Size> DFParams;
typedef TestBaseWithParam<DFParams> DenseOpticalFlow_DeepFlow;

PERF_TEST_P(DenseOpticalFlow_DeepFlow, perf, Values(szVGA, sz720p))
{
    DFParams params = GetParam();
    Size sz = get<0>(params);

    Mat frame1(sz, CV_8U);
    Mat frame2(sz, CV_8U);
    Mat flow;

    randu(frame1, 0, 255);
    randu(frame2, 0, 255);

    TEST_CYCLE_N(1)
    {
        Ptr<DenseOpticalFlow> algo = createOptFlow_DeepFlow();
        algo->calc(frame1, frame2, flow);
    }

    SANITY_CHECK_NOTHING();
}

}} // namespace
