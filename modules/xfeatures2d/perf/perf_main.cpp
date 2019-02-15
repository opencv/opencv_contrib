// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"
#include "opencv2/ts/cuda_perf.hpp"

static const char * impls[] = {
#ifdef HAVE_CUDA
    "cuda",
#endif
    "plain"
};

CV_PERF_TEST_MAIN_WITH_IMPLS(xfeatures2d, impls, perf::printCudaInfo())
