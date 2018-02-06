// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef __OPENCV_TEST_PRECOMP_HPP__
#define __OPENCV_TEST_PRECOMP_HPP__

#include "opencv2/ximgproc.hpp"
#include "opencv2/ts.hpp"
#include <opencv2/ts/ts_perf.hpp>
#include <opencv2/core/utility.hpp>

namespace opencv_test {
using namespace cv::ximgproc;
using namespace perf; // szODD

Ptr<AdaptiveManifoldFilter> createAMFilterRefImpl(double sigma_s, double sigma_r, bool adjust_outliers);
}

#endif
