// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_TEST_PRECOMP_HPP__
#define __OPENCV_TEST_PRECOMP_HPP__

#include "opencv2/ts.hpp"
#include "opencv2/cann.hpp"
#include "opencv2/ts/cuda_test.hpp"
#include "opencv2/cann_interface.hpp"
#include "opencv2/ascendc_kernels.hpp"

using namespace cv;
using namespace cv::cann;
#undef EXPECT_MAT_NEAR
#define EXPECT_MAT_NEAR(m1, m2, eps) EXPECT_PRED_FORMAT3(cvtest::assertMatNear, m1, m2, eps)
#define ASSERT_MAT_NEAR(m1, m2, eps) ASSERT_PRED_FORMAT3(cvtest::assertMatNear, m1, m2, eps)

#define DEVICE_ID 0

Mat randomMat(int w, int h, int dtype, float min = 1.0f, float max = 10.0f);
Scalar randomScalar();
float randomNum();
int randomInterger();
Mat genMask();
AscendMat genNpuMask();

#endif //__OPENCV_TEST_PRECOMP_HPP__
