// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
//
//                    Created by Simon Reich
//
#include "perf_precomp.hpp"

namespace opencv_test
{
namespace
{

/* 1. Define parameter type and test fixture */
typedef tuple<int, double> RGFTestParam;
typedef TestBaseWithParam<RGFTestParam> EdgepreservingFilterTest;

/* 2. Declare the testsuite */
PERF_TEST_P(EdgepreservingFilterTest, perf,
            Combine(Values(-20, 0, 10), Values(-100, 0, 20)))
{
    /* 3. Get actual test parameters */
    RGFTestParam params = GetParam();
    int kernelSize = get<0>(params);
    double threshold = get<1>(params);

    /* 4. Allocate and initialize arguments for tested function */
    std::string filename = getDataPath("perf/320x260.png");
    Mat src = imread(filename, 1);
    Mat dst(src.size(), src.type());

    /* 5. Manifest your expectations about this test */
    declare.in(src).out(dst);

    /* 6. Collect the samples! */
    PERF_SAMPLE_BEGIN();
        ximgproc::edgePreservingFilter(src, dst, kernelSize, threshold);
    PERF_SAMPLE_END();

    /* 7. Do not check anything */
    SANITY_CHECK_NOTHING();
}

} // namespace
} // namespace opencv_test
