/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <opencv2/core.hpp>
#include <opencv2/rgbd.hpp>
#include "opencv2/core/utility.hpp"
#include "opencv2/core/private.hpp"

namespace cv
{
namespace rgbd
{
  CV_INIT_ALGORITHM(DepthCleaner, "RGBD.DepthCleaner",
      obj.info()->addParam(obj, "window_size", obj.window_size_);
      obj.info()->addParam(obj, "depth", obj.depth_);
      obj.info()->addParam(obj, "method", obj.method_))

  CV_INIT_ALGORITHM(RgbdNormals, "RGBD.RgbdNormals",
      obj.info()->addParam(obj, "rows", obj.rows_);
      obj.info()->addParam(obj, "cols", obj.cols_);
      obj.info()->addParam(obj, "window_size", obj.window_size_);
      obj.info()->addParam(obj, "depth", obj.depth_);
      obj.info()->addParam(obj, "K", obj.K_);
      obj.info()->addParam(obj, "method", obj.method_))

  CV_INIT_ALGORITHM(RgbdPlane, "RGBD.RgbdPlane",
      obj.info()->addParam(obj, "block_size", obj.block_size_);
      obj.info()->addParam(obj, "min_size", obj.min_size_);
      obj.info()->addParam(obj, "method", obj.method_);
      obj.info()->addParam(obj, "threshold", obj.threshold_);
      obj.info()->addParam(obj, "sensor_error_a", obj.sensor_error_a_);
      obj.info()->addParam(obj, "sensor_error_b", obj.sensor_error_b_);
      obj.info()->addParam(obj, "sensor_error_c", obj.sensor_error_c_))

  CV_INIT_ALGORITHM(RgbdOdometry, "RGBD.RgbdOdometry",
      obj.info()->addParam(obj, "cameraMatrix", obj.cameraMatrix);
      obj.info()->addParam(obj, "minDepth", obj.minDepth);
      obj.info()->addParam(obj, "maxDepth", obj.maxDepth);
      obj.info()->addParam(obj, "maxDepthDiff", obj.maxDepthDiff);
      obj.info()->addParam(obj, "iterCounts", obj.iterCounts);
      obj.info()->addParam(obj, "minGradientMagnitudes", obj.minGradientMagnitudes);
      obj.info()->addParam(obj, "maxPointsPart", obj.maxPointsPart);
      obj.info()->addParam(obj, "transformType", obj.transformType);
      obj.info()->addParam(obj, "maxTranslation", obj.maxTranslation);
      obj.info()->addParam(obj, "maxRotation", obj.maxRotation);)

  CV_INIT_ALGORITHM(ICPOdometry, "RGBD.ICPOdometry",
      obj.info()->addParam(obj, "cameraMatrix", obj.cameraMatrix);
      obj.info()->addParam(obj, "minDepth", obj.minDepth);
      obj.info()->addParam(obj, "maxDepth", obj.maxDepth);
      obj.info()->addParam(obj, "maxDepthDiff", obj.maxDepthDiff);
      obj.info()->addParam(obj, "maxPointsPart", obj.maxPointsPart);
      obj.info()->addParam(obj, "iterCounts", obj.iterCounts);
      obj.info()->addParam(obj, "transformType", obj.transformType);
      obj.info()->addParam(obj, "maxTranslation", obj.maxTranslation);
      obj.info()->addParam(obj, "maxRotation", obj.maxRotation);
      obj.info()->addParam<RgbdNormals>(obj, "normalsComputer", obj.normalsComputer, true, NULL, NULL);)

  CV_INIT_ALGORITHM(RgbdICPOdometry, "RGBD.RgbdICPOdometry",
      obj.info()->addParam(obj, "cameraMatrix", obj.cameraMatrix);
      obj.info()->addParam(obj, "minDepth", obj.minDepth);
      obj.info()->addParam(obj, "maxDepth", obj.maxDepth);
      obj.info()->addParam(obj, "maxDepthDiff", obj.maxDepthDiff);
      obj.info()->addParam(obj, "maxPointsPart", obj.maxPointsPart);
      obj.info()->addParam(obj, "iterCounts", obj.iterCounts);
      obj.info()->addParam(obj, "minGradientMagnitudes", obj.minGradientMagnitudes);
      obj.info()->addParam(obj, "transformType", obj.transformType);
      obj.info()->addParam(obj, "maxTranslation", obj.maxTranslation);
      obj.info()->addParam(obj, "maxRotation", obj.maxRotation);
      obj.info()->addParam<RgbdNormals>(obj, "normalsComputer", obj.normalsComputer, true, NULL, NULL);)

  bool
  initModule_rgbd(void);
  bool
  initModule_rgbd(void)
  {
    bool all = true;
    all &= !RgbdNormals_info_auto.name().empty();
    all &= !RgbdPlane_info_auto.name().empty();
    all &= !RgbdOdometry_info_auto.name().empty();
    all &= !ICPOdometry_info_auto.name().empty();
    all &= !RgbdICPOdometry_info_auto.name().empty();
    return all;
  }
}
}

