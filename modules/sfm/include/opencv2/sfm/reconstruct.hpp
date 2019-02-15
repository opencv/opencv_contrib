/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#ifndef __OPENCV_SFM_RECONSTRUCT_HPP__
#define __OPENCV_SFM_RECONSTRUCT_HPP__

#include <vector>
#include <string>

#include <opencv2/core.hpp>

namespace cv
{
namespace sfm
{

//! @addtogroup reconstruction
//! @{

#if defined(CV_DOXYGEN) || defined(CERES_FOUND)

/** @brief Reconstruct 3d points from 2d correspondences while performing autocalibration.
  @param points2d Input vector of vectors of 2d points (the inner vector is per image).
  @param Ps Output vector with the 3x4 projections matrices of each image.
  @param points3d Output array with estimated 3d points.
  @param K Input/Output camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$. Input parameters used as initial guess.
  @param is_projective if true, the cameras are supposed to be projective.

  This method calls below signature and extracts projection matrices from estimated K, R and t.

   @note
    - Tracks must be as precise as possible. It does not handle outliers and is very sensible to them.
*/
CV_EXPORTS
void
reconstruct(InputArrayOfArrays points2d, OutputArray Ps, OutputArray points3d, InputOutputArray K,
            bool is_projective = false);

/** @brief Reconstruct 3d points from 2d correspondences while performing autocalibration.
  @param points2d Input vector of vectors of 2d points (the inner vector is per image).
  @param Rs Output vector of 3x3 rotations of the camera.
  @param Ts Output vector of 3x1 translations of the camera.
  @param points3d Output array with estimated 3d points.
  @param K Input/Output camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$. Input parameters used as initial guess.
  @param is_projective if true, the cameras are supposed to be projective.

  Internally calls libmv simple pipeline routine with some default parameters by instatiating SFMLibmvEuclideanReconstruction class.

  @note
    - Tracks must be as precise as possible. It does not handle outliers and is very sensible to them.
    - To see a working example for camera motion reconstruction, check the following tutorial: @ref tutorial_sfm_trajectory_estimation.
*/
CV_EXPORTS
void
reconstruct(InputArrayOfArrays points2d, OutputArray Rs, OutputArray Ts, InputOutputArray K,
            OutputArray points3d, bool is_projective = false);

/** @brief Reconstruct 3d points from 2d images while performing autocalibration.
  @param images a vector of string with the images paths.
  @param Ps Output vector with the 3x4 projections matrices of each image.
  @param points3d Output array with estimated 3d points.
  @param K Input/Output camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$. Input parameters used as initial guess.
  @param is_projective if true, the cameras are supposed to be projective.

  This method calls below signature and extracts projection matrices from estimated K, R and t.

   @note
    - The images must be ordered as they were an image sequence. Additionally, each frame should be as close as posible to the previous and posterior.
    - For now DAISY features are used in order to compute the 2d points tracks and it only works for 3-4 images.
*/
CV_EXPORTS
void
reconstruct(const std::vector<String> images, OutputArray Ps, OutputArray points3d,
            InputOutputArray K, bool is_projective = false);

/** @brief Reconstruct 3d points from 2d images while performing autocalibration.
  @param images a vector of string with the images paths.
  @param Rs Output vector of 3x3 rotations of the camera.
  @param Ts Output vector of 3x1 translations of the camera.
  @param points3d Output array with estimated 3d points.
  @param K Input/Output camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$. Input parameters used as initial guess.
  @param is_projective if true, the cameras are supposed to be projective.

  Internally calls libmv simple pipeline routine with some default parameters by instatiating SFMLibmvEuclideanReconstruction class.

   @note
    - The images must be ordered as they were an image sequence. Additionally, each frame should be as close as posible to the previous and posterior.
    - For now DAISY features are used in order to compute the 2d points tracks and it only works for 3-4 images.
    - To see a working example for scene reconstruction, check the following tutorial: @ref tutorial_sfm_scene_reconstruction.
*/
CV_EXPORTS
void
reconstruct(const std::vector<String> images, OutputArray Rs, OutputArray Ts,
            InputOutputArray K, OutputArray points3d, bool is_projective = false);

#endif /* CV_DOXYGEN || CERES_FOUND */

//! @} sfm

} /* namespace cv */
} /* namespace sfm */

#endif

/* End of file. */
