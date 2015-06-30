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
#include "libmv/camera/lens_distortion.h"
#include "libmv/numeric/levenberg_marquardt.h"

namespace libmv {

class UndistortionOptimizerClass {
 public:
  typedef Vec2 FMatrixType;
  typedef Vec2 XMatrixType;

  UndistortionOptimizerClass(const LensDistortion &lens_distortion,
                             const PinholeCamera &camera,
                             const Vec2 &distorted_point) :
    lens_distortion_(lens_distortion),
    camera_(camera),
    distorted_point_(distorted_point) {}

  Vec2 operator()(const Vec2 &x) const {
    Vec2 fx;
    Vec2 point_x(x.x(), x.y());
    Vec2 distorted_point_x;
    lens_distortion_.ComputeUndistortedCoordinates(camera_,point_x,
                                                   &distorted_point_x);
    fx = (distorted_point_ - distorted_point_x).cast<double>();

    return fx;
  }

  const LensDistortion &lens_distortion_;
  const PinholeCamera &camera_;
  const Vec2 &distorted_point_;
};

LensDistortion::LensDistortion(const Vec &radial_distortion,
                               const Vec &tangential_distortion) {
  set_radial_distortion(radial_distortion);
  set_tangential_distortion(tangential_distortion);
}

// Compute the Undistortion of the lens
// (using Levenberg-marquardt)
void LensDistortion::ComputeDistortedCoordinates(
    const PinholeCamera &camera,
    const Vec2 &point,
    Vec2 *undistorted_point) const {
  Vec2 undistorted_point_wanted(point.x(), point.y());
  UndistortionOptimizerClass undistortion_optimizer(*this, camera, point);
  LevenbergMarquardt<UndistortionOptimizerClass>::SolverParameters params;
  LevenbergMarquardt<UndistortionOptimizerClass> lm(undistortion_optimizer);

  LevenbergMarquardt<UndistortionOptimizerClass>::Results results =
    lm.minimize(params, &undistorted_point_wanted);

  (*undistorted_point) = undistorted_point_wanted;
}

void LensDistortion::ComputeUndistortedCoordinates(
    const PinholeCamera &camera,
    const Vec2 &point,
    Vec2 *undistorted_point) const {
  Vec2 point_centered = point - camera.principal_point();

  double u = point_centered.x() / camera.focal_x();
  double v = point_centered.y() / camera.focal_y();
  double radius_squared = u * u + v * v;

  double coef_radial = 0;
  if (radial_distortion_.size() > 0) {
    for (int i = radial_distortion_.size() - 1; i >= 0; --i) {
      coef_radial = (coef_radial + radial_distortion_[i]) * radius_squared;
    }
  }

  undistorted_point->x() = point.x() + point_centered.x() * coef_radial;
  undistorted_point->y() = point.y() + point_centered.y() * coef_radial;

  if (tangential_distortion_.size() >= 2) {
    double coef_tangential = 1;

    for (size_t i = 2; i < tangential_distortion_.size(); ++i) {
      coef_tangential += tangential_distortion_[i] * radius_squared;
    }
    undistorted_point->x() += (tangential_distortion_.x() * (radius_squared +
                               2. * u * u) + 2. * tangential_distortion_.y() *
                               u * v) * coef_tangential;
    undistorted_point->y() += (tangential_distortion_.y() * (radius_squared +
                               2. * v * v) + 2. * tangential_distortion_.x() *
                               u * v) * coef_tangential;
  }
}

LensDistortionField::LensDistortionField(const Vec &radial_distortion,
                                         const Vec &tangential_distortion) :
    LensDistortion(radial_distortion, tangential_distortion) {
  is_precomputed_grid_done_ = false;
}

void LensDistortionField::ComputeUndistortedCoordinates(
    const PinholeCamera &camera,
    const Vec2 &point,
    Vec2 *undistorted_point) const {
  // TODO(julien) Computes the undistorted coordinates of a point using the
  // look-up table
  (void) camera;
  (void) point;
  (void) undistorted_point;
}

void LensDistortionField::ComputeDistortionMap(
    const PinholeCamera &camera) {
  // TODO(julien) add a look-up table with precomputed radius for instance
  (void) camera;
}

}  // namespace libmv
