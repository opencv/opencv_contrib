// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include <cmath>
#include <limits>
#include <string>
#include "ceres/internal/eigen.h"
#include "ceres/internal/port.h"
#include "ceres/jet.h"
#include "ceres/rotation.h"
#include "ceres/stringprintf.h"
#include "ceres/test_util.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

const double kPi = 3.14159265358979323846;
const double kHalfSqrt2 = 0.707106781186547524401;

double RandDouble() {
  double r = rand();
  return r / RAND_MAX;
}

// A tolerance value for floating-point comparisons.
static double const kTolerance = numeric_limits<double>::epsilon() * 10;

// Looser tolerance used for numerically unstable conversions.
static double const kLooseTolerance = 1e-9;

// Use as:
// double quaternion[4];
// EXPECT_THAT(quaternion, IsNormalizedQuaternion());
MATCHER(IsNormalizedQuaternion, "") {
  if (arg == NULL) {
    *result_listener << "Null quaternion";
    return false;
  }

  double norm2 = arg[0] * arg[0] + arg[1] * arg[1] +
      arg[2] * arg[2] + arg[3] * arg[3];
  if (fabs(norm2 - 1.0) > kTolerance) {
    *result_listener << "squared norm is " << norm2;
    return false;
  }

  return true;
}

// Use as:
// double expected_quaternion[4];
// double actual_quaternion[4];
// EXPECT_THAT(actual_quaternion, IsNearQuaternion(expected_quaternion));
MATCHER_P(IsNearQuaternion, expected, "") {
  if (arg == NULL) {
    *result_listener << "Null quaternion";
    return false;
  }

  // Quaternions are equivalent upto a sign change. So we will compare
  // both signs before declaring failure.
  bool near = true;
  for (int i = 0; i < 4; i++) {
    if (fabs(arg[i] - expected[i]) > kTolerance) {
      near = false;
      break;
    }
  }

  if (near) {
    return true;
  }

  near = true;
  for (int i = 0; i < 4; i++) {
    if (fabs(arg[i] + expected[i]) > kTolerance) {
      near = false;
      break;
    }
  }

  if (near) {
    return true;
  }

  *result_listener << "expected : "
                   << expected[0] << " "
                   << expected[1] << " "
                   << expected[2] << " "
                   << expected[3] << " "
                   << "actual : "
                   << arg[0] << " "
                   << arg[1] << " "
                   << arg[2] << " "
                   << arg[3];
  return false;
}

// Use as:
// double expected_axis_angle[4];
// double actual_axis_angle[4];
// EXPECT_THAT(actual_axis_angle, IsNearAngleAxis(expected_axis_angle));
MATCHER_P(IsNearAngleAxis, expected, "") {
  if (arg == NULL) {
    *result_listener << "Null axis/angle";
    return false;
  }

  for (int i = 0; i < 3; i++) {
    if (fabs(arg[i] - expected[i]) > kTolerance) {
      *result_listener << "component " << i << " should be " << expected[i];
      return false;
    }
  }

  return true;
}

// Use as:
// double matrix[9];
// EXPECT_THAT(matrix, IsOrthonormal());
MATCHER(IsOrthonormal, "") {
  if (arg == NULL) {
    *result_listener << "Null matrix";
    return false;
  }

  for (int c1 = 0; c1 < 3; c1++) {
    for (int c2 = 0; c2 < 3; c2++) {
      double v = 0;
      for (int i = 0; i < 3; i++) {
        v += arg[i + 3 * c1] * arg[i + 3 * c2];
      }
      double expected = (c1 == c2) ? 1 : 0;
      if (fabs(expected - v) > kTolerance) {
        *result_listener << "Columns " << c1 << " and " << c2
                         << " should have dot product " << expected
                         << " but have " << v;
        return false;
      }
    }
  }

  return true;
}

// Use as:
// double matrix1[9];
// double matrix2[9];
// EXPECT_THAT(matrix1, IsNear3x3Matrix(matrix2));
MATCHER_P(IsNear3x3Matrix, expected, "") {
  if (arg == NULL) {
    *result_listener << "Null matrix";
    return false;
  }

  for (int i = 0; i < 9; i++) {
    if (fabs(arg[i] - expected[i]) > kTolerance) {
      *result_listener << "component " << i << " should be " << expected[i];
      return false;
    }
  }

  return true;
}

// Transforms a zero axis/angle to a quaternion.
TEST(Rotation, ZeroAngleAxisToQuaternion) {
  double axis_angle[3] = { 0, 0, 0 };
  double quaternion[4];
  double expected[4] = { 1, 0, 0, 0 };
  AngleAxisToQuaternion(axis_angle, quaternion);
  EXPECT_THAT(quaternion, IsNormalizedQuaternion());
  EXPECT_THAT(quaternion, IsNearQuaternion(expected));
}

// Test that exact conversion works for small angles.
TEST(Rotation, SmallAngleAxisToQuaternion) {
  // Small, finite value to test.
  double theta = 1.0e-2;
  double axis_angle[3] = { theta, 0, 0 };
  double quaternion[4];
  double expected[4] = { cos(theta/2), sin(theta/2.0), 0, 0 };
  AngleAxisToQuaternion(axis_angle, quaternion);
  EXPECT_THAT(quaternion, IsNormalizedQuaternion());
  EXPECT_THAT(quaternion, IsNearQuaternion(expected));
}

// Test that approximate conversion works for very small angles.
TEST(Rotation, TinyAngleAxisToQuaternion) {
  // Very small value that could potentially cause underflow.
  double theta = pow(numeric_limits<double>::min(), 0.75);
  double axis_angle[3] = { theta, 0, 0 };
  double quaternion[4];
  double expected[4] = { cos(theta/2), sin(theta/2.0), 0, 0 };
  AngleAxisToQuaternion(axis_angle, quaternion);
  EXPECT_THAT(quaternion, IsNormalizedQuaternion());
  EXPECT_THAT(quaternion, IsNearQuaternion(expected));
}

// Transforms a rotation by pi/2 around X to a quaternion.
TEST(Rotation, XRotationToQuaternion) {
  double axis_angle[3] = { kPi / 2, 0, 0 };
  double quaternion[4];
  double expected[4] = { kHalfSqrt2, kHalfSqrt2, 0, 0 };
  AngleAxisToQuaternion(axis_angle, quaternion);
  EXPECT_THAT(quaternion, IsNormalizedQuaternion());
  EXPECT_THAT(quaternion, IsNearQuaternion(expected));
}

// Transforms a unit quaternion to an axis angle.
TEST(Rotation, UnitQuaternionToAngleAxis) {
  double quaternion[4] = { 1, 0, 0, 0 };
  double axis_angle[3];
  double expected[3] = { 0, 0, 0 };
  QuaternionToAngleAxis(quaternion, axis_angle);
  EXPECT_THAT(axis_angle, IsNearAngleAxis(expected));
}

// Transforms a quaternion that rotates by pi about the Y axis to an axis angle.
TEST(Rotation, YRotationQuaternionToAngleAxis) {
  double quaternion[4] = { 0, 0, 1, 0 };
  double axis_angle[3];
  double expected[3] = { 0, kPi, 0 };
  QuaternionToAngleAxis(quaternion, axis_angle);
  EXPECT_THAT(axis_angle, IsNearAngleAxis(expected));
}

// Transforms a quaternion that rotates by pi/3 about the Z axis to an axis
// angle.
TEST(Rotation, ZRotationQuaternionToAngleAxis) {
  double quaternion[4] = { sqrt(3) / 2, 0, 0, 0.5 };
  double axis_angle[3];
  double expected[3] = { 0, 0, kPi / 3 };
  QuaternionToAngleAxis(quaternion, axis_angle);
  EXPECT_THAT(axis_angle, IsNearAngleAxis(expected));
}

// Test that exact conversion works for small angles.
TEST(Rotation, SmallQuaternionToAngleAxis) {
  // Small, finite value to test.
  double theta = 1.0e-2;
  double quaternion[4] = { cos(theta/2), sin(theta/2.0), 0, 0 };
  double axis_angle[3];
  double expected[3] = { theta, 0, 0 };
  QuaternionToAngleAxis(quaternion, axis_angle);
  EXPECT_THAT(axis_angle, IsNearAngleAxis(expected));
}

// Test that approximate conversion works for very small angles.
TEST(Rotation, TinyQuaternionToAngleAxis) {
  // Very small value that could potentially cause underflow.
  double theta = pow(numeric_limits<double>::min(), 0.75);
  double quaternion[4] = { cos(theta/2), sin(theta/2.0), 0, 0 };
  double axis_angle[3];
  double expected[3] = { theta, 0, 0 };
  QuaternionToAngleAxis(quaternion, axis_angle);
  EXPECT_THAT(axis_angle, IsNearAngleAxis(expected));
}

TEST(Rotation, QuaternionToAngleAxisAngleIsLessThanPi) {
  double quaternion[4];
  double angle_axis[3];

  const double half_theta = 0.75 * kPi;

  quaternion[0] = cos(half_theta);
  quaternion[1] = 1.0 * sin(half_theta);
  quaternion[2] = 0.0;
  quaternion[3] = 0.0;
  QuaternionToAngleAxis(quaternion, angle_axis);
  const double angle = sqrt(angle_axis[0] * angle_axis[0] +
                            angle_axis[1] * angle_axis[1] +
                            angle_axis[2] * angle_axis[2]);
  EXPECT_LE(angle, kPi);
}

static const int kNumTrials = 10000;

// Takes a bunch of random axis/angle values, converts them to quaternions,
// and back again.
TEST(Rotation, AngleAxisToQuaterionAndBack) {
  srand(5);
  for (int i = 0; i < kNumTrials; i++) {
    double axis_angle[3];
    // Make an axis by choosing three random numbers in [-1, 1) and
    // normalizing.
    double norm = 0;
    for (int i = 0; i < 3; i++) {
      axis_angle[i] = RandDouble() * 2 - 1;
      norm += axis_angle[i] * axis_angle[i];
    }
    norm = sqrt(norm);

    // Angle in [-pi, pi).
    double theta = kPi * 2 * RandDouble() - kPi;
    for (int i = 0; i < 3; i++) {
      axis_angle[i] = axis_angle[i] * theta / norm;
    }

    double quaternion[4];
    double round_trip[3];
    // We use ASSERTs here because if there's one failure, there are
    // probably many and spewing a million failures doesn't make anyone's
    // day.
    AngleAxisToQuaternion(axis_angle, quaternion);
    ASSERT_THAT(quaternion, IsNormalizedQuaternion());
    QuaternionToAngleAxis(quaternion, round_trip);
    ASSERT_THAT(round_trip, IsNearAngleAxis(axis_angle));
  }
}

// Takes a bunch of random quaternions, converts them to axis/angle,
// and back again.
TEST(Rotation, QuaterionToAngleAxisAndBack) {
  srand(5);
  for (int i = 0; i < kNumTrials; i++) {
    double quaternion[4];
    // Choose four random numbers in [-1, 1) and normalize.
    double norm = 0;
    for (int i = 0; i < 4; i++) {
      quaternion[i] = RandDouble() * 2 - 1;
      norm += quaternion[i] * quaternion[i];
    }
    norm = sqrt(norm);

    for (int i = 0; i < 4; i++) {
      quaternion[i] = quaternion[i] / norm;
    }

    double axis_angle[3];
    double round_trip[4];
    QuaternionToAngleAxis(quaternion, axis_angle);
    AngleAxisToQuaternion(axis_angle, round_trip);
    ASSERT_THAT(round_trip, IsNormalizedQuaternion());
    ASSERT_THAT(round_trip, IsNearQuaternion(quaternion));
  }
}

// Transforms a zero axis/angle to a rotation matrix.
TEST(Rotation, ZeroAngleAxisToRotationMatrix) {
  double axis_angle[3] = { 0, 0, 0 };
  double matrix[9];
  double expected[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  AngleAxisToRotationMatrix(axis_angle, matrix);
  EXPECT_THAT(matrix, IsOrthonormal());
  EXPECT_THAT(matrix, IsNear3x3Matrix(expected));
}

TEST(Rotation, NearZeroAngleAxisToRotationMatrix) {
  double axis_angle[3] = { 1e-24, 2e-24, 3e-24 };
  double matrix[9];
  double expected[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  AngleAxisToRotationMatrix(axis_angle, matrix);
  EXPECT_THAT(matrix, IsOrthonormal());
  EXPECT_THAT(matrix, IsNear3x3Matrix(expected));
}

// Transforms a rotation by pi/2 around X to a rotation matrix and back.
TEST(Rotation, XRotationToRotationMatrix) {
  double axis_angle[3] = { kPi / 2, 0, 0 };
  double matrix[9];
  // The rotation matrices are stored column-major.
  double expected[9] = { 1, 0, 0, 0, 0, 1, 0, -1, 0 };
  AngleAxisToRotationMatrix(axis_angle, matrix);
  EXPECT_THAT(matrix, IsOrthonormal());
  EXPECT_THAT(matrix, IsNear3x3Matrix(expected));
  double round_trip[3];
  RotationMatrixToAngleAxis(matrix, round_trip);
  EXPECT_THAT(round_trip, IsNearAngleAxis(axis_angle));
}

// Transforms an axis angle that rotates by pi about the Y axis to a
// rotation matrix and back.
TEST(Rotation, YRotationToRotationMatrix) {
  double axis_angle[3] = { 0, kPi, 0 };
  double matrix[9];
  double expected[9] = { -1, 0, 0, 0, 1, 0, 0, 0, -1 };
  AngleAxisToRotationMatrix(axis_angle, matrix);
  EXPECT_THAT(matrix, IsOrthonormal());
  EXPECT_THAT(matrix, IsNear3x3Matrix(expected));

  double round_trip[3];
  RotationMatrixToAngleAxis(matrix, round_trip);
  EXPECT_THAT(round_trip, IsNearAngleAxis(axis_angle));
}

TEST(Rotation, NearPiAngleAxisRoundTrip) {
  double in_axis_angle[3];
  double matrix[9];
  double out_axis_angle[3];

  srand(5);
  for (int i = 0; i < kNumTrials; i++) {
    // Make an axis by choosing three random numbers in [-1, 1) and
    // normalizing.
    double norm = 0;
    for (int i = 0; i < 3; i++) {
      in_axis_angle[i] = RandDouble() * 2 - 1;
      norm += in_axis_angle[i] * in_axis_angle[i];
    }
    norm = sqrt(norm);

    // Angle in [pi - kMaxSmallAngle, pi).
    const double kMaxSmallAngle = 1e-2;
    double theta = kPi - kMaxSmallAngle * RandDouble();

    for (int i = 0; i < 3; i++) {
      in_axis_angle[i] *= (theta / norm);
    }
    AngleAxisToRotationMatrix(in_axis_angle, matrix);
    RotationMatrixToAngleAxis(matrix, out_axis_angle);

    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR(out_axis_angle[i], in_axis_angle[i], kLooseTolerance);
    }
  }
}

TEST(Rotation, AtPiAngleAxisRoundTrip) {
  // A rotation of kPi about the X axis;
  static const double kMatrix[3][3] = {
    {1.0,  0.0,  0.0},
    {0.0,  -1.0,  0.0},
    {0.0,  0.0,  -1.0}
  };

  double in_matrix[9];
  // Fill it from kMatrix in col-major order.
  for (int j = 0, k = 0; j < 3; ++j) {
     for (int i = 0; i < 3; ++i, ++k) {
       in_matrix[k] = kMatrix[i][j];
     }
  }

  const double expected_axis_angle[3] = { kPi, 0, 0 };

  double out_matrix[9];
  double axis_angle[3];
  RotationMatrixToAngleAxis(in_matrix, axis_angle);
  AngleAxisToRotationMatrix(axis_angle, out_matrix);

  LOG(INFO) << "AngleAxis = " << axis_angle[0] << " " << axis_angle[1]
            << " " << axis_angle[2];
  LOG(INFO) << "Expected AngleAxis = " << kPi << " 0 0";
  double out_rowmajor[3][3];
  for (int j = 0, k = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i, ++k) {
      out_rowmajor[i][j] = out_matrix[k];
    }
  }
  LOG(INFO) << "Rotation:";
  LOG(INFO) << "EXPECTED      |        ACTUAL";
  for (int i = 0; i < 3; ++i) {
    string line;
    for (int j = 0; j < 3; ++j) {
      StringAppendF(&line, "%g ", kMatrix[i][j]);
    }
    line += "         |        ";
    for (int j = 0; j < 3; ++j) {
      StringAppendF(&line, "%g ", out_rowmajor[i][j]);
    }
    LOG(INFO) << line;
  }

  EXPECT_THAT(axis_angle, IsNearAngleAxis(expected_axis_angle));
  EXPECT_THAT(out_matrix, IsNear3x3Matrix(in_matrix));
}

// Transforms an axis angle that rotates by pi/3 about the Z axis to a
// rotation matrix.
TEST(Rotation, ZRotationToRotationMatrix) {
  double axis_angle[3] =  { 0, 0, kPi / 3 };
  double matrix[9];
  // This is laid-out row-major on the screen but is actually stored
  // column-major.
  double expected[9] = { 0.5, sqrt(3) / 2, 0,   // Column 1
                         -sqrt(3) / 2, 0.5, 0,  // Column 2
                         0, 0, 1 };             // Column 3
  AngleAxisToRotationMatrix(axis_angle, matrix);
  EXPECT_THAT(matrix, IsOrthonormal());
  EXPECT_THAT(matrix, IsNear3x3Matrix(expected));
  double round_trip[3];
  RotationMatrixToAngleAxis(matrix, round_trip);
  EXPECT_THAT(round_trip, IsNearAngleAxis(axis_angle));
}

// Takes a bunch of random axis/angle values, converts them to rotation
// matrices, and back again.
TEST(Rotation, AngleAxisToRotationMatrixAndBack) {
  srand(5);
  for (int i = 0; i < kNumTrials; i++) {
    double axis_angle[3];
    // Make an axis by choosing three random numbers in [-1, 1) and
    // normalizing.
    double norm = 0;
    for (int i = 0; i < 3; i++) {
      axis_angle[i] = RandDouble() * 2 - 1;
      norm += axis_angle[i] * axis_angle[i];
    }
    norm = sqrt(norm);

    // Angle in [-pi, pi).
    double theta = kPi * 2 * RandDouble() - kPi;
    for (int i = 0; i < 3; i++) {
      axis_angle[i] = axis_angle[i] * theta / norm;
    }

    double matrix[9];
    double round_trip[3];
    AngleAxisToRotationMatrix(axis_angle, matrix);
    ASSERT_THAT(matrix, IsOrthonormal());
    RotationMatrixToAngleAxis(matrix, round_trip);

    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR(round_trip[i], axis_angle[i], kLooseTolerance);
    }
  }
}

// Transposes a 3x3 matrix.
static void Transpose3x3(double m[9]) {
  std::swap(m[1], m[3]);
  std::swap(m[2], m[6]);
  std::swap(m[5], m[7]);
}

// Convert Euler angles from radians to degrees.
static void ToDegrees(double ea[3]) {
  for (int i = 0; i < 3; ++i)
    ea[i] *= 180.0 / kPi;
}

// Compare the 3x3 rotation matrices produced by the axis-angle
// rotation 'aa' and the Euler angle rotation 'ea' (in radians).
static void CompareEulerToAngleAxis(double aa[3], double ea[3]) {
  double aa_matrix[9];
  AngleAxisToRotationMatrix(aa, aa_matrix);
  Transpose3x3(aa_matrix);  // Column to row major order.

  double ea_matrix[9];
  ToDegrees(ea);  // Radians to degrees.
  const int kRowStride = 3;
  EulerAnglesToRotationMatrix(ea, kRowStride, ea_matrix);

  EXPECT_THAT(aa_matrix, IsOrthonormal());
  EXPECT_THAT(ea_matrix, IsOrthonormal());
  EXPECT_THAT(ea_matrix, IsNear3x3Matrix(aa_matrix));
}

// Test with rotation axis along the x/y/z axes.
// Also test zero rotation.
TEST(EulerAnglesToRotationMatrix, OnAxis) {
  int n_tests = 0;
  for (double x = -1.0; x <= 1.0; x += 1.0) {
    for (double y = -1.0; y <= 1.0; y += 1.0) {
      for (double z = -1.0; z <= 1.0; z += 1.0) {
        if ((x != 0) + (y != 0) + (z != 0) > 1)
          continue;
        double axis_angle[3] = {x, y, z};
        double euler_angles[3] = {x, y, z};
        CompareEulerToAngleAxis(axis_angle, euler_angles);
        ++n_tests;
      }
    }
  }
  CHECK_EQ(7, n_tests);
}

// Test that a random rotation produces an orthonormal rotation
// matrix.
TEST(EulerAnglesToRotationMatrix, IsOrthonormal) {
  srand(5);
  for (int trial = 0; trial < kNumTrials; ++trial) {
    double ea[3];
    for (int i = 0; i < 3; ++i)
      ea[i] = 360.0 * (RandDouble() * 2.0 - 1.0);
    double ea_matrix[9];
    ToDegrees(ea);  // Radians to degrees.
    EulerAnglesToRotationMatrix(ea, 3, ea_matrix);
    EXPECT_THAT(ea_matrix, IsOrthonormal());
  }
}

// Tests using Jets for specific behavior involving auto differentiation
// near singularity points.

typedef Jet<double, 3> J3;
typedef Jet<double, 4> J4;

J3 MakeJ3(double a, double v0, double v1, double v2) {
  J3 j;
  j.a = a;
  j.v[0] = v0;
  j.v[1] = v1;
  j.v[2] = v2;
  return j;
}

J4 MakeJ4(double a, double v0, double v1, double v2, double v3) {
  J4 j;
  j.a = a;
  j.v[0] = v0;
  j.v[1] = v1;
  j.v[2] = v2;
  j.v[3] = v3;
  return j;
}


bool IsClose(double x, double y) {
  EXPECT_FALSE(IsNaN(x));
  EXPECT_FALSE(IsNaN(y));
  double absdiff = fabs(x - y);
  if (x == 0 || y == 0) {
    return absdiff <= kTolerance;
  }
  double reldiff = absdiff / max(fabs(x), fabs(y));
  return reldiff <= kTolerance;
}

template <int N>
bool IsClose(const Jet<double, N> &x, const Jet<double, N> &y) {
  if (IsClose(x.a, y.a)) {
    for (int i = 0; i < N; i++) {
      if (!IsClose(x.v[i], y.v[i])) {
        return false;
      }
    }
  }
  return true;
}

template <int M, int N>
void ExpectJetArraysClose(const Jet<double, N> *x, const Jet<double, N> *y) {
  for (int i = 0; i < M; i++) {
    if (!IsClose(x[i], y[i])) {
      LOG(ERROR) << "Jet " << i << "/" << M << " not equal";
      LOG(ERROR) << "x[" << i << "]: " << x[i];
      LOG(ERROR) << "y[" << i << "]: " << y[i];
      Jet<double, N> d, zero;
      d.a = y[i].a - x[i].a;
      for (int j = 0; j < N; j++) {
        d.v[j] = y[i].v[j] - x[i].v[j];
      }
      LOG(ERROR) << "diff: " << d;
      EXPECT_TRUE(IsClose(x[i], y[i]));
    }
  }
}

// Log-10 of a value well below machine precision.
static const int kSmallTinyCutoff =
    static_cast<int>(2 * log(numeric_limits<double>::epsilon())/log(10.0));

// Log-10 of a value just below values representable by double.
static const int kTinyZeroLimit   =
    static_cast<int>(1 + log(numeric_limits<double>::min())/log(10.0));

// Test that exact conversion works for small angles when jets are used.
TEST(Rotation, SmallAngleAxisToQuaternionForJets) {
  // Examine small x rotations that are still large enough
  // to be well within the range represented by doubles.
  for (int i = -2; i >= kSmallTinyCutoff; i--) {
    double theta = pow(10.0, i);
    J3 axis_angle[3] = { J3(theta, 0), J3(0, 1), J3(0, 2) };
    J3 quaternion[4];
    J3 expected[4] = {
        MakeJ3(cos(theta/2), -sin(theta/2)/2, 0, 0),
        MakeJ3(sin(theta/2), cos(theta/2)/2, 0, 0),
        MakeJ3(0, 0, sin(theta/2)/theta, 0),
        MakeJ3(0, 0, 0, sin(theta/2)/theta),
    };
    AngleAxisToQuaternion(axis_angle, quaternion);
    ExpectJetArraysClose<4, 3>(quaternion, expected);
  }
}


// Test that conversion works for very small angles when jets are used.
TEST(Rotation, TinyAngleAxisToQuaternionForJets) {
  // Examine tiny x rotations that extend all the way to where
  // underflow occurs.
  for (int i = kSmallTinyCutoff; i >= kTinyZeroLimit; i--) {
    double theta = pow(10.0, i);
    J3 axis_angle[3] = { J3(theta, 0), J3(0, 1), J3(0, 2) };
    J3 quaternion[4];
    // To avoid loss of precision in the test itself,
    // a finite expansion is used here, which will
    // be exact up to machine precision for the test values used.
    J3 expected[4] = {
        MakeJ3(1.0, 0, 0, 0),
        MakeJ3(0, 0.5, 0, 0),
        MakeJ3(0, 0, 0.5, 0),
        MakeJ3(0, 0, 0, 0.5),
    };
    AngleAxisToQuaternion(axis_angle, quaternion);
    ExpectJetArraysClose<4, 3>(quaternion, expected);
  }
}

// Test that derivatives are correct for zero rotation.
TEST(Rotation, ZeroAngleAxisToQuaternionForJets) {
  J3 axis_angle[3] = { J3(0, 0), J3(0, 1), J3(0, 2) };
  J3 quaternion[4];
  J3 expected[4] = {
      MakeJ3(1.0, 0, 0, 0),
      MakeJ3(0, 0.5, 0, 0),
      MakeJ3(0, 0, 0.5, 0),
      MakeJ3(0, 0, 0, 0.5),
  };
  AngleAxisToQuaternion(axis_angle, quaternion);
  ExpectJetArraysClose<4, 3>(quaternion, expected);
}

// Test that exact conversion works for small angles.
TEST(Rotation, SmallQuaternionToAngleAxisForJets) {
  // Examine small x rotations that are still large enough
  // to be well within the range represented by doubles.
  for (int i = -2; i >= kSmallTinyCutoff; i--) {
    double theta = pow(10.0, i);
    double s = sin(theta);
    double c = cos(theta);
    J4 quaternion[4] = { J4(c, 0), J4(s, 1), J4(0, 2), J4(0, 3) };
    J4 axis_angle[3];
    J4 expected[3] = {
        MakeJ4(s, -2*theta, 2*theta*c, 0, 0),
        MakeJ4(0, 0, 0, 2*theta/s, 0),
        MakeJ4(0, 0, 0, 0, 2*theta/s),
    };
    QuaternionToAngleAxis(quaternion, axis_angle);
    ExpectJetArraysClose<3, 4>(axis_angle, expected);
  }
}

// Test that conversion works for very small angles.
TEST(Rotation, TinyQuaternionToAngleAxisForJets) {
  // Examine tiny x rotations that extend all the way to where
  // underflow occurs.
  for (int i = kSmallTinyCutoff; i >= kTinyZeroLimit; i--) {
    double theta = pow(10.0, i);
    double s = sin(theta);
    double c = cos(theta);
    J4 quaternion[4] = { J4(c, 0), J4(s, 1), J4(0, 2), J4(0, 3) };
    J4 axis_angle[3];
    // To avoid loss of precision in the test itself,
    // a finite expansion is used here, which will
    // be exact up to machine precision for the test values used.
    J4 expected[3] = {
        MakeJ4(theta, -2*theta, 2.0, 0, 0),
        MakeJ4(0, 0, 0, 2.0, 0),
        MakeJ4(0, 0, 0, 0, 2.0),
    };
    QuaternionToAngleAxis(quaternion, axis_angle);
    ExpectJetArraysClose<3, 4>(axis_angle, expected);
  }
}

// Test that conversion works for no rotation.
TEST(Rotation, ZeroQuaternionToAngleAxisForJets) {
  J4 quaternion[4] = { J4(1, 0), J4(0, 1), J4(0, 2), J4(0, 3) };
  J4 axis_angle[3];
  J4 expected[3] = {
      MakeJ4(0, 0, 2.0, 0, 0),
      MakeJ4(0, 0, 0, 2.0, 0),
      MakeJ4(0, 0, 0, 0, 2.0),
  };
  QuaternionToAngleAxis(quaternion, axis_angle);
  ExpectJetArraysClose<3, 4>(axis_angle, expected);
}

TEST(Quaternion, RotatePointGivesSameAnswerAsRotationByMatrixCanned) {
  // Canned data generated in octave.
  double const q[4] = {
    +0.1956830471754074,
    -0.0150618562474847,
    +0.7634572982788086,
    -0.3019454777240753,
  };
  double const Q[3][3] = {  // Scaled rotation matrix.
    { -0.6355194033477252,  0.0951730541682254,  0.3078870197911186 },
    { -0.1411693904792992,  0.5297609702153905, -0.4551502574482019 },
    { -0.2896955822708862, -0.4669396571547050, -0.4536309793389248 },
  };
  double const R[3][3] = {  // With unit rows and columns.
    { -0.8918859164053080,  0.1335655625725649,  0.4320876677394745 },
    { -0.1981166751680096,  0.7434648665444399, -0.6387564287225856 },
    { -0.4065578619806013, -0.6553016349046693, -0.6366242786393164 },
  };

  // Compute R from q and compare to known answer.
  double Rq[3][3];
  QuaternionToScaledRotation<double>(q, Rq[0]);
  ExpectArraysClose(9, Q[0], Rq[0], kTolerance);

  // Now do the same but compute R with normalization.
  QuaternionToRotation<double>(q, Rq[0]);
  ExpectArraysClose(9, R[0], Rq[0], kTolerance);
}


TEST(Quaternion, RotatePointGivesSameAnswerAsRotationByMatrix) {
  // Rotation defined by a unit quaternion.
  double const q[4] = {
    0.2318160216097109,
    -0.0178430356832060,
    0.9044300776717159,
    -0.3576998641394597,
  };
  double const p[3] = {
    +0.11,
    -13.15,
    1.17,
  };

  double R[3 * 3];
  QuaternionToRotation(q, R);

  double result1[3];
  UnitQuaternionRotatePoint(q, p, result1);

  double result2[3];
  VectorRef(result2, 3) = ConstMatrixRef(R, 3, 3)* ConstVectorRef(p, 3);
  ExpectArraysClose(3, result1, result2, kTolerance);
}


// Verify that (a * b) * c == a * (b * c).
TEST(Quaternion, MultiplicationIsAssociative) {
  double a[4];
  double b[4];
  double c[4];
  for (int i = 0; i < 4; ++i) {
    a[i] = 2 * RandDouble() - 1;
    b[i] = 2 * RandDouble() - 1;
    c[i] = 2 * RandDouble() - 1;
  }

  double ab[4];
  double ab_c[4];
  QuaternionProduct(a, b, ab);
  QuaternionProduct(ab, c, ab_c);

  double bc[4];
  double a_bc[4];
  QuaternionProduct(b, c, bc);
  QuaternionProduct(a, bc, a_bc);

  ASSERT_NEAR(ab_c[0], a_bc[0], kTolerance);
  ASSERT_NEAR(ab_c[1], a_bc[1], kTolerance);
  ASSERT_NEAR(ab_c[2], a_bc[2], kTolerance);
  ASSERT_NEAR(ab_c[3], a_bc[3], kTolerance);
}


TEST(AngleAxis, RotatePointGivesSameAnswerAsRotationMatrix) {
  double angle_axis[3];
  double R[9];
  double p[3];
  double angle_axis_rotated_p[3];
  double rotation_matrix_rotated_p[3];

  for (int i = 0; i < 10000; ++i) {
    double theta = (2.0 * i * 0.0011 - 1.0) * kPi;
    for (int j = 0; j < 50; ++j) {
      double norm2 = 0.0;
      for (int k = 0; k < 3; ++k) {
        angle_axis[k] = 2.0 * RandDouble() - 1.0;
        p[k] = 2.0 * RandDouble() - 1.0;
        norm2 = angle_axis[k] * angle_axis[k];
      }

      const double inv_norm = theta / sqrt(norm2);
      for (int k = 0; k < 3; ++k) {
        angle_axis[k] *= inv_norm;
      }

      AngleAxisToRotationMatrix(angle_axis, R);
      rotation_matrix_rotated_p[0] = R[0] * p[0] + R[3] * p[1] + R[6] * p[2];
      rotation_matrix_rotated_p[1] = R[1] * p[0] + R[4] * p[1] + R[7] * p[2];
      rotation_matrix_rotated_p[2] = R[2] * p[0] + R[5] * p[1] + R[8] * p[2];

      AngleAxisRotatePoint(angle_axis, p, angle_axis_rotated_p);
      for (int k = 0; k < 3; ++k) {
        EXPECT_NEAR(rotation_matrix_rotated_p[k],
                    angle_axis_rotated_p[k],
                    kTolerance) << "p: " << p[0]
                                << " " << p[1]
                                << " " << p[2]
                                << " angle_axis: " << angle_axis[0]
                                << " " << angle_axis[1]
                                << " " << angle_axis[2];
      }
    }
  }
}

TEST(AngleAxis, NearZeroRotatePointGivesSameAnswerAsRotationMatrix) {
  double angle_axis[3];
  double R[9];
  double p[3];
  double angle_axis_rotated_p[3];
  double rotation_matrix_rotated_p[3];

  for (int i = 0; i < 10000; ++i) {
    double norm2 = 0.0;
    for (int k = 0; k < 3; ++k) {
      angle_axis[k] = 2.0 * RandDouble() - 1.0;
      p[k] = 2.0 * RandDouble() - 1.0;
      norm2 = angle_axis[k] * angle_axis[k];
    }

    double theta = (2.0 * i * 0.0001  - 1.0) * 1e-16;
    const double inv_norm = theta / sqrt(norm2);
    for (int k = 0; k < 3; ++k) {
      angle_axis[k] *= inv_norm;
    }

    AngleAxisToRotationMatrix(angle_axis, R);
    rotation_matrix_rotated_p[0] = R[0] * p[0] + R[3] * p[1] + R[6] * p[2];
    rotation_matrix_rotated_p[1] = R[1] * p[0] + R[4] * p[1] + R[7] * p[2];
    rotation_matrix_rotated_p[2] = R[2] * p[0] + R[5] * p[1] + R[8] * p[2];

    AngleAxisRotatePoint(angle_axis, p, angle_axis_rotated_p);
    for (int k = 0; k < 3; ++k) {
      EXPECT_NEAR(rotation_matrix_rotated_p[k],
                  angle_axis_rotated_p[k],
                  kTolerance) << "p: " << p[0]
                              << " " << p[1]
                              << " " << p[2]
                              << " angle_axis: " << angle_axis[0]
                              << " " << angle_axis[1]
                              << " " << angle_axis[2];
    }
  }
}

TEST(MatrixAdapter, RowMajor3x3ReturnTypeAndAccessIsCorrect) {
  double array[9] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
  const float const_array[9] =
      { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
  MatrixAdapter<double, 3, 1> A = RowMajorAdapter3x3(array);
  MatrixAdapter<const float, 3, 1> B = RowMajorAdapter3x3(const_array);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // The values are integers from 1 to 9, so equality tests are appropriate
      // even for float and double values.
      EXPECT_EQ(A(i, j), array[3*i+j]);
      EXPECT_EQ(B(i, j), const_array[3*i+j]);
    }
  }
}

TEST(MatrixAdapter, ColumnMajor3x3ReturnTypeAndAccessIsCorrect) {
  double array[9] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
  const float const_array[9] =
      { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
  MatrixAdapter<double, 1, 3> A = ColumnMajorAdapter3x3(array);
  MatrixAdapter<const float, 1, 3> B = ColumnMajorAdapter3x3(const_array);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // The values are integers from 1 to 9, so equality tests are
      // appropriate even for float and double values.
      EXPECT_EQ(A(i, j), array[3*j+i]);
      EXPECT_EQ(B(i, j), const_array[3*j+i]);
    }
  }
}

TEST(MatrixAdapter, RowMajor2x4IsCorrect) {
  const int expected[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
  int array[8];
  MatrixAdapter<int, 4, 1> M(array);
  M(0, 0) = 1; M(0, 1) = 2; M(0, 2) = 3; M(0, 3) = 4;
  M(1, 0) = 5; M(1, 1) = 6; M(1, 2) = 7; M(1, 3) = 8;
  for (int k = 0; k < 8; ++k) {
    EXPECT_EQ(array[k], expected[k]);
  }
}

TEST(MatrixAdapter, ColumnMajor2x4IsCorrect) {
  const int expected[8] = { 1, 5, 2, 6, 3, 7, 4, 8 };
  int array[8];
  MatrixAdapter<int, 1, 2> M(array);
  M(0, 0) = 1; M(0, 1) = 2; M(0, 2) = 3; M(0, 3) = 4;
  M(1, 0) = 5; M(1, 1) = 6; M(1, 2) = 7; M(1, 3) = 8;
  for (int k = 0; k < 8; ++k) {
    EXPECT_EQ(array[k], expected[k]);
  }
}

}  // namespace internal
}  // namespace ceres
