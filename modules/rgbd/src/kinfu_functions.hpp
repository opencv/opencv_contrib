// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#ifndef __OPENCV_RGBD_KINFU_FUNCTIONS_HPP__
#define __OPENCV_RGBD_KINFU_FUNCTIONS_HPP__

#include "utils.hpp"
#include "precomp.hpp"

namespace cv {

bool kinfuCommonUpdate(Odometry& odometry, Volume& volume, InputArray _depth, OdometryFrame& prevFrame, OdometryFrame& renderFrame, Matx44f& pose, int& frameCounter);

void kinfuCommonRender(const OdometryFrame& renderFrame, OutputArray image, const Vec3f& lightPose);

} // namespace cv

#endif