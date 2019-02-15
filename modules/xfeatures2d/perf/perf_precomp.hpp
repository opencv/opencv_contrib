// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef __OPENCV_PERF_PRECOMP_HPP__
#define __OPENCV_PERF_PRECOMP_HPP__

#include "cvconfig.h"

#include "opencv2/ts.hpp"
#include "opencv2/xfeatures2d.hpp"

#ifdef HAVE_OPENCV_OCL
#  include "opencv2/ocl.hpp"
#endif

#ifdef HAVE_CUDA
#  include "opencv2/xfeatures2d/cuda.hpp"
#endif

namespace opencv_test {
using namespace cv::xfeatures2d;
using namespace perf;
}

#endif
