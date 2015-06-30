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

#ifndef LIBMV_CAMERA_LENS_DISTORTION_H_
#define LIBMV_CAMERA_LENS_DISTORTION_H_

#include "libmv/base/vector.h"
#include "libmv/logging/logging.h"
#include "libmv/numeric/numeric.h"

namespace libmv {

class PinholeCamera;

//
// Lens distortion class.
//
// We use the Brown's distortion model
// Variables:
//   (x,y): 2D point in the image (pixel)
//   (u,v): the undistorted 2D point (pixel)
//   radial_distortion (k1, k2, k3, ...): vector containing the
//   radial distortion
//   tangential_distortion_ (p1, p2): vector containing the
//                                    tangential distortion
//   (cx,cy): camera principal point
//
// Equation:
//  u = x + (x - cx) * (k1 * r^2 + k2 * r^4 +...)
//    + (p1(r^2 + 2(x-cx)^2) + 2p2(x-cx)(y-cy))(1 + p3*r^2 +...)
//  v = y + (y - cy) * (k1 * r^2 + k2 * r^4 +...)
//   + (p2(r^2 + 2(y-cy)^2) + 2p1(x-cx)(y-cy))(1 + p3*r^2 +...)
 //
class LensDistortion {
 public:
  LensDistortion(const Vec &radial_distortion = Vec(),
                 const Vec &tangential_distortion = Vec());
  virtual ~LensDistortion() {};

  // Compute the distorted coordinates of a 2D point
  // \param[in] camera is a pinhole camera model
  // \param[in] point is the 2D point (in pixel) we need to undistort
  // \param[out] undistorted_point is the distort 2D point (pixel)
  //
  virtual void ComputeDistortedCoordinates(const PinholeCamera &camera,
                                           const Vec2 &point,
                                           Vec2 *distorted_point) const;

  // Compute the undistorted coordinates of a 2D point
  // \param[in] camera is a pinhole camera model
  // \param[in] point is the 2D point (in pixel) we need to undistort
  // \param[out] undistorted_point is the undistort 2D point (pixel)
  //
  virtual void ComputeUndistortedCoordinates(const PinholeCamera &camera,
                                             const Vec2 &point,
                                             Vec2 *undistorted_point) const;


  void set_radial_distortion(const Vec &radial_distortion) {
    radial_distortion_ = radial_distortion;
  }
  void set_tangential_distortion(const Vec &tangential_distortion) {
    tangential_distortion_ = tangential_distortion;
  }

  const Vec &radial_distortion() const     { return radial_distortion_; }
  const Vec &tangential_distortion() const { return tangential_distortion_; }

 private:
  Vec radial_distortion_;
  Vec tangential_distortion_;

};

//
// Use a precomputed map for fast undistortion computation
// WARNING: This is at best, barely started.
// This class is cleary not thought out yet!
class LensDistortionField : public LensDistortion {
 public:
  LensDistortionField(const Vec &radial_distortion = Vec(),
                      const Vec &tangential_distortion = Vec());

  // Compute the undistorted coordinates of a 2D point
  // \param[in] camera is a pinhole camera model
  // \param[in] point is the 2D point (in pixel) we need to undistort
  // \param[out] undistorted_point is the undistort 2D point (pixel)
  //
  void ComputeUndistortedCoordinates(const PinholeCamera &camera,
                                     const Vec2 &point,
                                     Vec2 *undistorted_point) const;

  // The function constructs two vectors containing the undistorted
  // coordinates for every pixels of an image
  void ComputeDistortionMap(const PinholeCamera &camera);

 private:
  // TODO(julien): Find an efficient index for the precomputed map
  // Contains the undistorted coordinates for every pixel
  vector<Vec> precomputed_undistortion_grid_;
  // Defines if the precomputed_distortion_grid is precomputed or not
  bool is_precomputed_grid_done_;
};

}

#endif  // LIBMV_CAMERA_LENS_DISTORTION_H_
