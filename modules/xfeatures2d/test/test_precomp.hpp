// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef __OPENCV_TEST_PRECOMP_HPP__
#define __OPENCV_TEST_PRECOMP_HPP__

#include "opencv2/ts.hpp"
#include "opencv2/ts/cuda_test.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"

#include "cvconfig.h"

#ifdef HAVE_OPENCL
#  include "opencv2/core/ocl.hpp"
#endif

#ifdef HAVE_CUDA
#  include "opencv2/xfeatures2d/cuda.hpp"
#endif

namespace opencv_test {
using namespace cv::xfeatures2d;
}

#endif
