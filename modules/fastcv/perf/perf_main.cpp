/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

static void initFastCVTests()
{
    cvtest::registerGlobalSkipTag(CV_TEST_TAG_FASTCV_SKIP_DSP);
}

CV_PERF_TEST_MAIN(imgproc, initFastCVTests())
