// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test {
    namespace {

        typedef tuple<Size, MatType> HoughSpaceTransformPerfTestParam;
        typedef perf::TestBaseWithParam<HoughSpaceTransformPerfTestParam> HoughSpaceTransformPerfTest;

        PERF_TEST_P(HoughSpaceTransformPerfTest, perf,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::Values(CV_8U)
            )
        )
        {
            Size srcSize = get<0>(GetParam());
            int  srcType = get<1>(GetParam());

            Mat src(srcSize, srcType);
            Mat hough;

            declare.in(src, WARMUP_RNG);

            TEST_CYCLE_N(3)
            {
                HoughSpaceTransform(src, hough);
            }

            SANITY_CHECK_NOTHING();
        }
    }
}
