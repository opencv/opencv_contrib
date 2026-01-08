// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

static
void initTrackingTests()
{
    const char* extraTestDataPath =
#ifdef WINRT
        NULL;
#else
        getenv("OPENCV_DNN_TEST_DATA_PATH");
#endif
    if (extraTestDataPath)
        cvtest::addDataSearchPath(extraTestDataPath);

    cvtest::addDataSearchSubDirectory("");  // override "cv" prefix below to access without "../dnn" hacks
}

CV_PERF_TEST_MAIN(tracking, initTrackingTests())
