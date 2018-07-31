// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

/* 1. Define parameter type and test fixture */
typedef tuple<int, double> RGFTestParam;
typedef TestBaseWithParam<RGFTestParam> EdgepreservingFilterTest;

/* 2. Declare the testsuite */
PERF_TEST_P(EdgepreservingFilterTest, perf,
    Combine(
    Values(-20, 0, 10),
    Values(-100, 0 , 20))
)
{
    /* 3. Get actual test parameters */
    RGFTestParam params = GetParam();
    int kernelSize    = get<0>(params);
    double threshold  = get<1>(params);


    /* 4. Allocate and initialize arguments for tested function */
    std::string filename = getDataPath("samples/data/corridor.jpg");
    Mat src = imread(filename, 1);
    Mat dst(src.size(), src.type());

    /* 5. Manifest your expectations about this test */
    declare.in(src, WARMUP_RNG).out(dst);

    /* 6. Collect the samples! */
    TEST_CYCLE_N(1)
    {
	      ximgproc::edgepreservingFilter(src, dst, kernelSize, threshold);
    }

    /* 7. Do not check anything */
    SANITY_CHECK_NOTHING();
}


}} // namespace
