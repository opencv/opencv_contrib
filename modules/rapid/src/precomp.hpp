// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#include "opencv2/rapid.hpp"
#include <vector>
#include <opencv2/calib3d.hpp>

namespace cv
{
namespace rapid
{
void compute1DSobel(const Mat& src, Mat& dst);
}
} // namespace cv

#endif
