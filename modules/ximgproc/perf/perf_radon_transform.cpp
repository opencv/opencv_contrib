// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test {
    namespace {

        typedef tuple<Size, MatType> RadonTransformPerfTestParam;
        typedef perf::TestBaseWithParam<RadonTransformPerfTestParam> RadonTransformPerfTest;

        PERF_TEST_P(RadonTransformPerfTest, perf,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::Values(CV_8U)
            )
        )
        {
            Size srcSize = get<0>(GetParam());
            int  srcType = get<1>(GetParam());

            Mat src(srcSize, srcType);
            Mat radon;

            declare.in(src, WARMUP_RNG);

            TEST_CYCLE_N(3)
            {
                RadonTransform(src, radon);
            }

            SANITY_CHECK_NOTHING();
        }
    }
}
