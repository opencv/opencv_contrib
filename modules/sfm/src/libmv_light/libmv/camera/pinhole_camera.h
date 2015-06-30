// Copyright (c) 2009 libmv authors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#ifndef LIBMV_CAMERA_PINHOLE_CAMERA_H_
#define LIBMV_CAMERA_PINHOLE_CAMERA_H_

#include "libmv/camera/camera.h"
#include "libmv/camera/lens_distortion.h"
#include "libmv/correspondence/feature.h"
#include "libmv/logging/logging.h"
#include "libmv/numeric/numeric.h"
#include "libmv/multiview/projection.h"

namespace libmv {

//
// A pinhole camera model
//
// Equation of the projection of a 3D point in the camera frame:
//  s x = K [ R | t ] X
// where
//  X is the homogeneous 3D point
//  K is the calibration matrix with the intrinsic parameters.
//  R is the orientation of the camera expressed in a world coordinate frame
//  t is the position of the camera expressed in a world coordinate frame
//  x is the 2D projection (pixel) of the homogeneous point X
//    in the camera frame
//  s is a 2D scale factor
//                             [ focal_x skew_factor principal_point_x]
//  K can be decomposed in K = [    0      focal_y   principal_point_y]
//                             [    0        0              1         ]
//  where
//    focal_x and focal_y are the camera focal (in pixels) along x and y axis
//    principal_point_x and principal_point_y are the coordinate of the
//      principal point (in pixels) expressed in the camera frame
//
class PinholeCamera : public Camera {
 public:
  PinholeCamera();
  PinholeCamera(const Mat34  &P);
  PinholeCamera(const Mat3  &R, const Vec3  &t);
  PinholeCamera(const Mat3  &K, const Mat3  &R, const Vec3  &t);
  PinholeCamera(double focal, const Vec2 &principal_point);
  virtual ~PinholeCamera();

  // The function computes the projection of a 3D point
  virtual bool ProjectPointStructure(const PointStructure &point_structure,
                                     PointFeature *feature) const;
  // The function computes the projection of a 3D point
  virtual bool ProjectPointStructure(const PointStructure &point_structure,
                                     Vec2 *q) const;

  // TODO(julien) A visibility test for a structure

  // The function returns the ray direction of a pixel
  virtual Vec3 Ray(const Vec2f &pixel) {
    return Vec3(pixel.x(), pixel.y(), -focal_x_);
  }

  PinholeCamera(const PinholeCamera &camera);

  const Mat34 &projection_matrix() const  { return projection_matrix_; }
  const Mat3 &intrinsic_matrix() const    { return intrinsic_matrix_; }
  const Mat3 &orientation_matrix() const  { return orientation_matrix_; }
  const Vec3 &position() const            { return position_; }

  double focal_x() const                  { return focal_x_; }
  double focal_y() const                  { return focal_y_; }
  double skew_factor() const              { return skew_factor_; }
  const Vec2 &principal_point() const     { return principal_point_; }
  const Vec2u &image_size() const          { return image_size_; }
  uint image_width() const          { return image_size_(0); }
  uint image_height() const          { return image_size_(1); }

  void set_projection_matrix(const Mat34 & p) {
    projection_matrix_ = p;
    // scale K so that K(2,2) = 1
    KRt_From_P(projection_matrix_,
               &intrinsic_matrix_,
               &orientation_matrix_,
               &position_);
    focal_x_            = intrinsic_matrix_(0, 0);
    focal_y_            = intrinsic_matrix_(1, 1);
    skew_factor_        = intrinsic_matrix_(0, 1);
    principal_point_(0) = intrinsic_matrix_(0, 2) ;
    principal_point_(1) = intrinsic_matrix_(1, 2) ;
  }

  void set_intrinsic_matrix(const Mat3 &intrinsic_matrix) {
    intrinsic_matrix_   = intrinsic_matrix;
    focal_x_            = intrinsic_matrix_(0, 0);
    focal_y_            = intrinsic_matrix_(1, 1);
    skew_factor_        = intrinsic_matrix_(0, 1);
    principal_point_(0) = intrinsic_matrix_(0, 2) ;
    principal_point_(1) = intrinsic_matrix_(1, 2) ;

    CHECK(intrinsic_matrix_(1, 0) == 0 &&
          intrinsic_matrix_(2, 0) == 0 &&
          intrinsic_matrix_(2, 1) == 0);

    UpdateProjectionMatrix();
  }

  void set_orientation_matrix(const Mat3 &orientation_matrix) {
    orientation_matrix_ = orientation_matrix;
    UpdateProjectionMatrix();
  }
  void set_position(const Vec3 &position) {
    position_ = position;
    UpdateProjectionMatrix();
  }
  // TODO(julien) clean the code: avoid to have SetXXX and set_XXX
  void SetExtrinsicParameters(const Mat3 &orientation_matrix,
                              const Vec3 &position) {
    orientation_matrix_ = orientation_matrix;
    position_ = position;
    UpdateProjectionMatrix();
  }
  void SetIntrinsicParameters(double focal, const Vec2 &principal_point) {
    focal_x_ = focal;
    focal_y_ = focal;
    principal_point_ = principal_point;
    UpdateIntrinsicMatrix();
    UpdateProjectionMatrix();
  }
  void SetIntrinsicExtrinsicParameters(const Mat3 &intrinsic_matrix,
                                       const Mat3 &orientation_matrix,
                                       const Vec3 &position) {
    intrinsic_matrix_   = intrinsic_matrix;
    focal_x_ = intrinsic_matrix_(0, 0);
    focal_y_ = intrinsic_matrix_(1, 1);
    principal_point_ << intrinsic_matrix_(0, 2), intrinsic_matrix_(1, 2);
    orientation_matrix_ = orientation_matrix;
    position_ = position;
    UpdateProjectionMatrix();
  }
  void GetIntrinsicExtrinsicParameters(Mat3 *intrinsic_matrix,
                                       Mat3 *orientation_matrix,
                                       Vec3 *position) {
    *intrinsic_matrix   = intrinsic_matrix_;
    *orientation_matrix = orientation_matrix_;
    *position = position_;
  }
  const Mat34 GetPoseMatrix() const  {
    Mat34 P;
    P.block<3, 3>(0, 0) = orientation_matrix_;
    P.col(3) = position_;
    return P;
  }
  void SetFocal(double focal) {
    focal_x_ = focal;
    focal_y_ = focal;
    UpdateIntrinsicMatrix();
    UpdateProjectionMatrix();
  }
  void SetFocal(double focal_x, double focal_y) {
    focal_x_ = focal_x;
    focal_y_ = focal_y;
    UpdateIntrinsicMatrix();
    UpdateProjectionMatrix();
  }
  void set_principal_point(const Vec2 &principal_point) {
    principal_point_ = principal_point;
    UpdateIntrinsicMatrix();
    UpdateProjectionMatrix();
  }
  void set_skew_factor(double skew_factor) {
    skew_factor_ = skew_factor;
    UpdateIntrinsicMatrix();
    UpdateProjectionMatrix();
  }
  // Sets the image size (width, height)
  void set_image_size(const Vec2u &size) {
    image_size_ = size;
  }

 private:
  void UpdateProjectionMatrix();
  void UpdateIntrinsicMatrix();

  Mat34 projection_matrix_;
  Mat3  intrinsic_matrix_;
  Mat3  orientation_matrix_;
  Vec3  position_;

  double  focal_x_;
  double  focal_y_;
  Vec2    principal_point_;
  double  skew_factor_;
  // Contains the image size (width, height)
  Vec2u    image_size_;
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// The class models a pinhole camera model with radial and/or tangential
// distortion.
class PinholeCameraDistortion : public PinholeCamera {
 public:
  PinholeCameraDistortion(LensDistortion *lens_distortion = NULL);
  PinholeCameraDistortion(const Mat3 &K,
                          const Mat3 &R,
                          const Vec3 &t,
                          LensDistortion *lens_distortion);
  PinholeCameraDistortion(const Mat3 &R,
                          const Vec3 &t,
                          LensDistortion *lens_distortion);

  PinholeCameraDistortion(const PinholeCameraDistortion &camera);

  // The function computes the projection of a 3D point
  virtual bool ProjectPointStructure(const PointStructure &point_structure,
                                     PointFeature *feature) const;

  // The function computes the undistorted feature using the camera distorsion
  // model
  virtual void ComputeUndistortedFeature(const Feature &feature,
                                         Feature *undistorted_feature) const;

  // The function undistort the feature using the camera distorsion model
  virtual void UndistortFeature(Feature *feature) const;

  void ComputeUndistortedCoordinates(const Vec2 &point,
                                     Vec2 *undistorted_point) const;
  void ComputeDistortedCoordinates(const Vec2 &point,
                                   Vec2 *distorted_point) const;

  const LensDistortion *lens_distortion() const {
    return lens_distortion_;
  }

  void set_lens_distortion(LensDistortion *lens_distortion) {
    lens_distortion_ = lens_distortion;
  }

 private:
  LensDistortion *lens_distortion_;
};

}  // namespace libmv

#endif  // LIBMV_CAMERA_PINHOLE_CAMERA_H_
