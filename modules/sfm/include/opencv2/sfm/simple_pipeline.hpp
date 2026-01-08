/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
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

#ifndef __OPENCV_SFM_SIMPLE_PIPELINE_HPP__
#define __OPENCV_SFM_SIMPLE_PIPELINE_HPP__

#include <opencv2/core.hpp>

namespace cv
{
namespace sfm
{

//! @addtogroup simple_pipeline
//! @{

/** @brief Different camera models that libmv supports.
 */
enum {
  SFM_DISTORTION_MODEL_POLYNOMIAL = 0, // LIBMV_DISTORTION_MODEL_POLYNOMIAL
  SFM_DISTORTION_MODEL_DIVISION = 1,   // LIBMV_DISTORTION_MODEL_DIVISION
};

/** @brief Data structure describing the camera model and its parameters.
  @param _distortion_model Type of camera model.
  @param _focal_length_x focal length of the camera (in pixels).
  @param _focal_length_y focal length of the camera (in pixels).
  @param _principal_point_x principal point of the camera in the x direction (in pixels).
  @param _principal_point_y principal point of the camera in the y direction (in pixels).
  @param _polynomial_k1 radial distortion parameter.
  @param _polynomial_k2 radial distortion parameter.
  @param _polynomial_k3 radial distortion parameter.
  @param _polynomial_p1 radial distortion parameter.
  @param _polynomial_p2 radial distortion parameter.

  Is assumed that modern cameras have their principal point in the image center.\n
  In case that the camera model was SFM_DISTORTION_MODEL_DIVISION, it's only needed to provide
  _polynomial_k1 and _polynomial_k2 which will be assigned as division distortion parameters.
 */
class CV_EXPORTS_W_SIMPLE libmv_CameraIntrinsicsOptions
{
public:
  CV_WRAP
  libmv_CameraIntrinsicsOptions(const int _distortion_model=0,
                                const double _focal_length_x=0,
                                const double _focal_length_y=0,
                                const double _principal_point_x=0,
                                const double _principal_point_y=0,
                                const double _polynomial_k1=0,
                                const double _polynomial_k2=0,
                                const double _polynomial_k3=0,
                                const double _polynomial_p1=0,
                                const double _polynomial_p2=0)
    : distortion_model(_distortion_model),
      image_width(2*_principal_point_x),
      image_height(2*_principal_point_y),
      focal_length_x(_focal_length_x),
      focal_length_y(_focal_length_y),
      principal_point_x(_principal_point_x),
      principal_point_y(_principal_point_y),
      polynomial_k1(_polynomial_k1),
      polynomial_k2(_polynomial_k2),
      polynomial_k3(_polynomial_k3),
      division_k1(_polynomial_p1),
      division_k2(_polynomial_p2)
  {
    if ( _distortion_model == SFM_DISTORTION_MODEL_DIVISION )
    {
      division_k1 = _polynomial_k1;
      division_k2 = _polynomial_k2;
    }
  }

  // Common settings of all distortion models.
  CV_PROP_RW int distortion_model;
  CV_PROP_RW int image_width, image_height;
  CV_PROP_RW double focal_length_x;
  CV_PROP_RW double focal_length_y;
  CV_PROP_RW double principal_point_x, principal_point_y;

  // Radial distortion model.
  CV_PROP_RW double polynomial_k1, polynomial_k2, polynomial_k3;
  CV_PROP_RW double polynomial_p1, polynomial_p2;

  // Division distortion model.
  CV_PROP_RW double division_k1, division_k2;
};


/** @brief All internal camera parameters that libmv is able to refine.
 */
enum { SFM_REFINE_FOCAL_LENGTH         = (1 << 0),  // libmv::BUNDLE_FOCAL_LENGTH
       SFM_REFINE_PRINCIPAL_POINT      = (1 << 1),  // libmv::BUNDLE_PRINCIPAL_POINT
       SFM_REFINE_RADIAL_DISTORTION_K1 = (1 << 2),  // libmv::BUNDLE_RADIAL_K1
       SFM_REFINE_RADIAL_DISTORTION_K2 = (1 << 4),  // libmv::BUNDLE_RADIAL_K2
};


/** @brief Data structure describing the reconstruction options.
  @param _keyframe1 first keyframe used in order to initialize the reconstruction.
  @param _keyframe2 second keyframe used in order to initialize the reconstruction.
  @param _refine_intrinsics camera parameter or combination of parameters to refine.
  @param _select_keyframes allows to select automatically the initial keyframes. If 1 then autoselection is enabled. If 0 then is disabled.
  @param _verbosity_level verbosity logs level for Glog. If -1 then logs are disabled, otherwise the log level will be the input integer.
 */
class CV_EXPORTS_W_SIMPLE libmv_ReconstructionOptions
{
public:
  CV_WRAP
  libmv_ReconstructionOptions(const int _keyframe1=1,
                              const int _keyframe2=2,
                              const int _refine_intrinsics=1,
                              const int _select_keyframes=1,
                              const int _verbosity_level=-1)
    : keyframe1(_keyframe1), keyframe2(_keyframe2),
      refine_intrinsics(_refine_intrinsics),
      select_keyframes(_select_keyframes),
      verbosity_level(_verbosity_level) {}

  CV_PROP_RW int keyframe1, keyframe2;
  CV_PROP_RW int refine_intrinsics;
  CV_PROP_RW int select_keyframes;
  CV_PROP_RW int verbosity_level;
};


/** @brief base class BaseSFM declares a common API that would be used in a typical scene reconstruction scenario
 */
class CV_EXPORTS_W BaseSFM
{
public:
  virtual ~BaseSFM() {};

  CV_WRAP
  virtual void run(InputArrayOfArrays points2d) = 0;

  CV_WRAP
  virtual void run(InputArrayOfArrays points2d, InputOutputArray K, OutputArray Rs,
                   OutputArray Ts, OutputArray points3d) = 0;

  virtual void run(const std::vector<String> &images) = 0;
  virtual void run(const std::vector<String> &images, InputOutputArray K, OutputArray Rs,
                   OutputArray Ts, OutputArray points3d) = 0;

  CV_WRAP virtual double getError() const = 0;
  CV_WRAP virtual void getPoints(OutputArray points3d) = 0;
  CV_WRAP virtual cv::Mat getIntrinsics() const = 0;
  CV_WRAP virtual void getCameras(OutputArray Rs, OutputArray Ts) = 0;

  CV_WRAP
  virtual void
  setReconstructionOptions(const libmv_ReconstructionOptions &libmv_reconstruction_options) = 0;

  CV_WRAP
  virtual void
  setCameraIntrinsicOptions(const libmv_CameraIntrinsicsOptions &libmv_camera_intrinsics_options) = 0;
};

/** @brief SFMLibmvEuclideanReconstruction class provides an interface with the Libmv Structure From Motion pipeline.
 */
class CV_EXPORTS_W SFMLibmvEuclideanReconstruction : public BaseSFM
{
public:
  /** @brief Calls the pipeline in order to perform Eclidean reconstruction.
    @param points2d Input vector of vectors of 2d points (the inner vector is per image).

    @note
      - Tracks must be as precise as possible. It does not handle outliers and is very sensible to them.
  */
  CV_WRAP
  virtual void run(InputArrayOfArrays points2d) CV_OVERRIDE = 0;

  /** @brief Calls the pipeline in order to perform Eclidean reconstruction.
    @param points2d Input vector of vectors of 2d points (the inner vector is per image).
    @param K Input/Output camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$. Input parameters used as initial guess.
    @param Rs Output vector of 3x3 rotations of the camera.
    @param Ts Output vector of 3x1 translations of the camera.
    @param points3d Output array with estimated 3d points.

    @note
      - Tracks must be as precise as possible. It does not handle outliers and is very sensible to them.
  */
  CV_WRAP
  virtual void run(InputArrayOfArrays points2d, InputOutputArray K, OutputArray Rs,
                   OutputArray Ts, OutputArray points3d) CV_OVERRIDE = 0;

  /** @brief Calls the pipeline in order to perform Eclidean reconstruction.
    @param images a vector of string with the images paths.

    @note
      - The images must be ordered as they were an image sequence. Additionally, each frame should be as close as posible to the previous and posterior.
      - For now DAISY features are used in order to compute the 2d points tracks and it only works for 3-4 images.
  */
  virtual void run(const std::vector<String> &images) CV_OVERRIDE = 0;

  /** @brief Calls the pipeline in order to perform Eclidean reconstruction.
    @param images a vector of string with the images paths.
    @param K Input/Output camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$. Input parameters used as initial guess.
    @param Rs Output vector of 3x3 rotations of the camera.
    @param Ts Output vector of 3x1 translations of the camera.
    @param points3d Output array with estimated 3d points.

    @note
      - The images must be ordered as they were an image sequence. Additionally, each frame should be as close as posible to the previous and posterior.
      - For now DAISY features are used in order to compute the 2d points tracks and it only works for 3-4 images.
  */
  virtual void run(const std::vector<String> &images, InputOutputArray K, OutputArray Rs,
                   OutputArray Ts, OutputArray points3d) CV_OVERRIDE = 0;

  /** @brief Returns the computed reprojection error.
  */
  CV_WRAP
  virtual double getError() const CV_OVERRIDE = 0;

  /** @brief Returns the estimated 3d points.
    @param points3d Output array with estimated 3d points.
  */
  CV_WRAP
  virtual void getPoints(OutputArray points3d) CV_OVERRIDE = 0;

  /** @brief Returns the refined camera calibration matrix.
  */
  CV_WRAP
  virtual cv::Mat getIntrinsics() const CV_OVERRIDE = 0;

  /** @brief Returns the estimated camera extrinsic parameters.
    @param Rs Output vector of 3x3 rotations of the camera.
    @param Ts Output vector of 3x1 translations of the camera.
  */
  CV_WRAP
  virtual void getCameras(OutputArray Rs, OutputArray Ts) CV_OVERRIDE = 0;

  /** @brief Setter method for reconstruction options.
    @param libmv_reconstruction_options struct with reconstruction options such as initial keyframes,
      automatic keyframe selection, parameters to refine and the verbosity level.
  */
  CV_WRAP
  virtual void
  setReconstructionOptions(const libmv_ReconstructionOptions &libmv_reconstruction_options) CV_OVERRIDE = 0;

  /** @brief Setter method for camera intrinsic options.
    @param libmv_camera_intrinsics_options struct with camera intrinsic options such as camera model and
      the internal camera parameters.
  */
  CV_WRAP
  virtual void
  setCameraIntrinsicOptions(const libmv_CameraIntrinsicsOptions &libmv_camera_intrinsics_options) CV_OVERRIDE = 0;

  /** @brief Creates an instance of the SFMLibmvEuclideanReconstruction class. Initializes Libmv. */
  static Ptr<SFMLibmvEuclideanReconstruction>
    create(const libmv_CameraIntrinsicsOptions &camera_instrinsic_options=libmv_CameraIntrinsicsOptions(),
           const libmv_ReconstructionOptions &reconstruction_options=libmv_ReconstructionOptions());
  };

//! @} sfm

} /* namespace cv */
} /* namespace sfm */

#endif

/* End of file. */
