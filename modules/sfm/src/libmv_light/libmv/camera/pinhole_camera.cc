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

#include "libmv/camera/pinhole_camera.h"
#include "libmv/multiview/structure.h"

namespace libmv {

PinholeCamera::PinholeCamera() {
  orientation_matrix_.setIdentity();
  intrinsic_matrix_.setIdentity();
  UpdateProjectionMatrix();
}

PinholeCamera::PinholeCamera(const Mat34 &P) {
  projection_matrix_ = P;
  // TODO(julien) when the reconstruction is metric, we can call KRt_From_P();
  // but whent it is projective, what to do? add a boolean?
}

PinholeCamera::PinholeCamera(const Mat3 &R, const Vec3 &t) {
  orientation_matrix_ = R;
  position_ = t;
  UpdateProjectionMatrix();
}

PinholeCamera::PinholeCamera(const Mat3 &K,
                             const Mat3 &R,
                             const Vec3 &t) {
  intrinsic_matrix_ = K;
  orientation_matrix_ = R;
  position_ = t;
  UpdateProjectionMatrix();
}

PinholeCamera::PinholeCamera(double focal,
                             const Vec2 &principal_point) {
  orientation_matrix_.setIdentity();
  SetIntrinsicParameters(focal, principal_point);
}

PinholeCamera::~PinholeCamera() {
}

PinholeCamera::PinholeCamera(const PinholeCamera &camera) {
  this->set_intrinsic_matrix(camera.intrinsic_matrix());
}

// The function computes updates the projection matrix from intrinsic
// parameters (focal,...).
void PinholeCamera::UpdateIntrinsicMatrix() {
  intrinsic_matrix_ << focal_x_,  skew_factor_, principal_point_ (0),
                       0,         focal_y_,     principal_point_ (1),
                       0,         0,            1;
  UpdateProjectionMatrix();
}

// The function updates the projection matrix from intrinsic and
// extrinsic parameters.
void PinholeCamera::UpdateProjectionMatrix() {
  P_From_KRt(intrinsic_matrix_,
             orientation_matrix_,
             position_,
             &projection_matrix_);
}

// The function computes the projection of a 3D point.
bool PinholeCamera::ProjectPointStructure(
    const PointStructure &point_structure,
    PointFeature *feature) const {
  Vec2 q2;
  Project(projection_matrix_, point_structure.coords(), &q2);
  feature->coords << q2.cast<float>();
  return true;
}

// The function computes the projection of a 3D point.
bool PinholeCamera::ProjectPointStructure(
    const PointStructure &point_structure,
    Vec2 *q) const {
  Project(projection_matrix_, point_structure.coords(), q);
  return true;
}

PinholeCameraDistortion::PinholeCameraDistortion(
    LensDistortion *lens_distortion) {
  lens_distortion_ = lens_distortion;
}

PinholeCameraDistortion::PinholeCameraDistortion(
    const Mat3 &K,
    const Mat3 &R,
    const Vec3 &t,
    LensDistortion *lens_distortion) :
    PinholeCamera(K,R,t) {
  lens_distortion_ = lens_distortion;
}

PinholeCameraDistortion::PinholeCameraDistortion(
    const Mat3 &R,
    const Vec3 &t,
    LensDistortion *lens_distortion) :
    PinholeCamera(R,t) {
  lens_distortion_ = lens_distortion;
}

// The function copy the camera parameters from the camera provided
PinholeCameraDistortion::PinholeCameraDistortion(
    const PinholeCameraDistortion &camera) : PinholeCamera(camera){
  (*this) = camera;
  this->lens_distortion_ = camera.lens_distortion_ ;

}

// The function computes the projection of a 3D point
bool PinholeCameraDistortion::ProjectPointStructure(
    const PointStructure &point_structure,
    PointFeature *feature) const {
  // Project using the pin-hole model
  PinholeCamera::ProjectPointStructure(point_structure, feature);
  // correct the feature using the lens distortion
  Vec2 coords;
  ComputeDistortedCoordinates(feature->coords.cast<double>(), &coords);
  feature->coords = coords.cast<float>();
  return true;
}

// The function computes the undistorted feature using the camera distorsion
// model
void PinholeCameraDistortion::ComputeUndistortedFeature(
    const Feature &feature,
    Feature *undistorted_feature) const {
  // TODO(julien) call the 2D undistortion for every 2D component of a feature
  (void) feature;
  (void) undistorted_feature;
}

// The function undistorts the feature using the camera distorsion model
void PinholeCameraDistortion::UndistortFeature(Feature *feature) const {
  // TODO(julien) call the 2D undistortion for every 2D component of a feature
  (void) feature;
}

void PinholeCameraDistortion::ComputeUndistortedCoordinates(
    const Vec2 &point,
    Vec2 *undistorted_point) const {
  if (lens_distortion_) {
    lens_distortion_->ComputeUndistortedCoordinates(*this,
                                                    point,
                                                    undistorted_point);
  }
}

void PinholeCameraDistortion::ComputeDistortedCoordinates(
    const Vec2 &point,
    Vec2 *distorted_point) const {
  if (lens_distortion_) {
    lens_distortion_->ComputeDistortedCoordinates(*this,
                                                  point,
                                                  distorted_point);
  }
}

}  // namespace libmv
