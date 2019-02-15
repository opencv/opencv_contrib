//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2014, OpenCV Foundation, all rights reserved.
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
// Author: Tolga Birdal <tbirdal AT gmail.com>

#ifndef __OPENCV_SURFACE_MATCHING_UTILS_HPP_
#define __OPENCV_SURFACE_MATCHING_UTILS_HPP_

#include <cmath>
#include <cstdio>

namespace cv
{
namespace ppf_match_3d
{

const float EPS = 1.192092896e-07F;        /* smallest such that 1.0+FLT_EPSILON != 1.0 */

#ifndef M_PI
#define M_PI  3.1415926535897932384626433832795
#endif

static inline void TNormalize3(Vec3d& v)
{
  double norm = cv::norm(v);
  if (norm > EPS)
  {
    v *= 1.0 / norm;
  }
}

/**
 *  \brief Calculate angle between two normalized vectors
 *
 *  \param [in] a normalized vector
 *  \param [in] b normalized vector
 *  \return angle between a and b vectors in radians
 */
static inline double TAngle3Normalized(const Vec3d& a, const Vec3d& b)
{
  /*
   angle = atan2(a dot b, |a x b|) # Bertram (accidental mistake)
   angle = atan2(|a x b|, a dot b) # Tolga Birdal (correction)
   angle = acos(a dot b)           # Hamdi Sahloul (simplification, a & b are normalized)
  */

  return acos(a.dot(b));
}

static inline void rtToPose(const Matx33d& R, const Vec3d& t, Matx44d& Pose)
{
  Matx34d P;
  hconcat(R, t, P);
  vconcat(P, Matx14d(0, 0, 0, 1), Pose);
}

static inline void poseToR(const Matx44d& Pose, Matx33d& R)
{
  Mat(Pose).rowRange(0, 3).colRange(0, 3).copyTo(R);
}

static inline void poseToRT(const Matx44d& Pose, Matx33d& R, Vec3d& t)
{
  poseToR(Pose, R);
  Mat(Pose).rowRange(0, 3).colRange(3, 4).copyTo(t);
}

/**
 *  \brief Axis angle to rotation
 */
static inline void aaToR(const Vec3d& axis, double angle, Matx33d& R)
{
  const double sinA = sin(angle);
  const double cosA = cos(angle);
  const double cos1A = (1 - cosA);
  uint i, j;

  Mat(cosA * Matx33d::eye()).copyTo(R);

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
    {
      if (i != j)
      {
        // Symmetry skew matrix
        R(i, j) += (((i + 1) % 3 == j) ? -1 : 1) * sinA * axis[3 - i - j];
      }
      R(i, j) += cos1A * axis[i] * axis[j];
    }
}

/**
 *  \brief Compute a rotation in order to rotate around X direction
 */
static inline void getUnitXRotation(double angle, Matx33d& Rx)
{
  const double sx = sin(angle);
  const double cx = cos(angle);

  Mat(Rx.eye()).copyTo(Rx);
  Rx(1, 1) = cx;
  Rx(1, 2) = -sx;
  Rx(2, 1) = sx;
  Rx(2, 2) = cx;
}

/**
*  \brief Compute a rotation in order to rotate around Y direction
*/
static inline void getUnitYRotation(double angle, Matx33d& Ry)
{
  const double sy = sin(angle);
  const double cy = cos(angle);

  Mat(Ry.eye()).copyTo(Ry);
  Ry(0, 0) = cy;
  Ry(0, 2) = sy;
  Ry(2, 0) = -sy;
  Ry(2, 2) = cy;
}

/**
*  \brief Compute a rotation in order to rotate around Z direction
*/
static inline void getUnitZRotation(double angle, Matx33d& Rz)
{
  const double sz = sin(angle);
  const double cz = cos(angle);

  Mat(Rz.eye()).copyTo(Rz);
  Rz(0, 0) = cz;
  Rz(0, 1) = -sz;
  Rz(1, 0) = sz;
  Rz(1, 1) = cz;
}

/**
*  \brief Convert euler representation to rotation matrix
*
*  \param [in] euler RPY angles
*  \param [out] R 3x3 Rotation matrix
*/
static inline void eulerToDCM(const Vec3d& euler, Matx33d& R)
{
  Matx33d Rx, Ry, Rz;

  getUnitXRotation(euler[0], Rx);
  getUnitYRotation(euler[1], Ry);
  getUnitZRotation(euler[2], Rz);

  Mat(Rx * (Ry * Rz)).copyTo(R);
}


/**
 *  \brief Compute the transformation needed to rotate n1 onto x axis and p1 to origin
 */
static inline void computeTransformRT(const Vec3d& p1, const Vec3d& n1, Matx33d& R, Vec3d& t)
{
  // dot product with x axis
  double angle = acos(n1[0]);

  // cross product with x axis
  Vec3d axis(0, n1[2], -n1[1]);

  // we try to project on the ground plane but it's already parallel
  if (n1[1] == 0 && n1[2] == 0)
  {
    axis[1] = 1;
    axis[2] = 0;
  }
  else
  {
    TNormalize3(axis);
  }

  aaToR(axis, angle, R);
  t = -R * p1;
}

/**
 *  \brief Flip a normal to the viewing direction
 *
 *  \param [in] point Scene point
 *  \param [in] vp view direction
 *  \param [in] n normal
 */
static inline void flipNormalViewpoint(const Vec3f& point, const Vec3f& vp, Vec3f& n)
{
  float cos_theta;

  // See if we need to flip any plane normals
  Vec3f diff = vp - point;

  // Dot product between the (viewpoint - point) and the plane normal
  cos_theta = diff.dot(n);

  // Flip the plane normal
  if (cos_theta < 0)
  {
    n *= -1;
  }
}

/**
 *  \brief Convert a rotation matrix to axis angle representation
 *
 *  \param [in] R Rotation matrix
 *  \param [out] axis Axis vector
 *  \param [out] angle Angle in radians
 */
static inline void dcmToAA(Matx33d& R, Vec3d& axis, double *angle)
{
  Mat(Vec3d(R(2, 1) - R(2, 1),
            R(0, 2) - R(2, 0),
            R(1, 0) - R(0, 1))).copyTo(axis);
  TNormalize3(axis);
  *angle = acos(0.5 * (cv::trace(R) - 1.0));
}

/**
 *  \brief Convert axis angle representation to rotation matrix
 *
 *  \param [in] axis Axis Vector
 *  \param [in] angle Angle (In radians)
 *  \param [out] R 3x3 Rotation matrix
 */
static inline void aaToDCM(const Vec3d& axis, double angle, Matx33d& R)
{
  uint i, j;
  Matx33d n = Matx33d::all(0);

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      if (i != j)
        n(i, j) = (((i + 1) % 3 == j) ? -1 : 1) * axis[3 - i - j];
  Mat(Matx33d::eye() + sin(angle) * n + cos(angle) * n * n).copyTo(R);
}

/**
 *  \brief Convert a discrete cosine matrix to quaternion
 *
 *  \param [in] R Rotation Matrix
 *  \param [in] q Quaternion
 */
static inline void dcmToQuat(Matx33d& R, Vec4d& q)
{
  double tr = cv::trace(R);
  Vec3d v(R(0, 0), R(1, 1), R(2, 2));
  int idx = tr > 0.0 ? 3 : (int)(std::max_element(v.val, v.val + 3) - v.val);
  double norm4 = q[(idx + 1) % 4] = 1.0 + (tr > 0.0 ? tr : 2 * R(idx, idx) - tr);
  int i, prev, next, step = idx % 2 ? 1 : -1, curr = 3;
  for (i = 0; i < 3; i++)
  {
    curr = (curr + step) % 4;
    next = (curr + 1) % 3, prev = (curr + 2) % 3;
    q[(idx + i + 2) % 4] = R(next, prev) + (tr > 0.0 || idx == curr ? -1 : 1) * R(prev, next);
  }
  q *= 0.5 / sqrt(norm4);
}

/**
 *  \brief Convert quaternion to a discrete cosine matrix
 *
 *  \param [in] q Quaternion (w is at first element)
 *  \param [in] R Rotation Matrix
 *
 */
static inline void quatToDCM(Vec4d& q, Matx33d& R)
{
  Vec4d sq = q.mul(q);

  double tmp1, tmp2;

  R(0, 0) = sq[0] + sq[1] - sq[2] - sq[3]; // since norm(q) = 1
  R(1, 1) = sq[0] - sq[1] + sq[2] - sq[3];
  R(2, 2) = sq[0] - sq[1] - sq[2] + sq[3];

  tmp1 = q[1] * q[2];
  tmp2 = q[3] * q[0];

  R(0, 1) = 2.0 * (tmp1 + tmp2);
  R(1, 0) = 2.0 * (tmp1 - tmp2);

  tmp1 = q[1] * q[3];
  tmp2 = q[2] * q[0];

  R(0, 2) = 2.0 * (tmp1 - tmp2);
  R(2, 0) = 2.0 * (tmp1 + tmp2);

  tmp1 = q[2] * q[3];
  tmp2 = q[1] * q[0];

  R(1, 2) = 2.0 * (tmp1 + tmp2);
  R(2, 1) = 2.0 * (tmp1 - tmp2);
}

} // namespace ppf_match_3d

} // namespace cv

#endif
