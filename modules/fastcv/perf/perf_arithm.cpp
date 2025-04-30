/*
 * Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

typedef perf::TestBaseWithParam<tuple<Size, int>> IntegrateYUVPerfTest;

PERF_TEST_P(IntegrateYUVPerfTest, run,
    ::testing::Combine(::testing::Values(perf::szVGA, perf::sz720p, perf::sz1080p), // image size
                       ::testing::Values(CV_8U)                                     // image depth
                      )
           )
{
    cv::Size srcSize = get<0>(GetParam());
    int depth = get<1>(GetParam());

    cv::Mat Y(srcSize, depth), CbCr(srcSize.height/2, srcSize.width, depth);
    cv::Mat IY, ICb, ICr;
    RNG& rng = cv::theRNG();
    cvtest::randUni(rng, Y, Scalar::all(0), Scalar::all(255));
    cvtest::randUni(rng, CbCr, Scalar::all(0), Scalar::all(255));

    TEST_CYCLE() cv::fastcv::integrateYUV(Y, CbCr, IY, ICb, ICr);

    SANITY_CHECK_NOTHING();
}

} // namespace