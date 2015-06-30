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
// Author: keir@google.com (Keir Mierle)

#include "ceres/jet.h"

#include <algorithm>
#include <cmath>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "ceres/fpclassify.h"
#include "ceres/stringprintf.h"
#include "ceres/test_util.h"

#define VL VLOG(1)

namespace ceres {
namespace internal {

const double kE = 2.71828182845904523536;

typedef Jet<double, 2> J;

// Convenient shorthand for making a jet.
J MakeJet(double a, double v0, double v1) {
  J z;
  z.a = a;
  z.v[0] = v0;
  z.v[1] = v1;
  return z;
}

// On a 32-bit optimized build, the mismatch is about 1.4e-14.
double const kTolerance = 1e-13;

void ExpectJetsClose(const J &x, const J &y) {
  ExpectClose(x.a, y.a, kTolerance);
  ExpectClose(x.v[0], y.v[0], kTolerance);
  ExpectClose(x.v[1], y.v[1], kTolerance);
}

TEST(Jet, Jet) {
  // Pick arbitrary values for x and y.
  J x = MakeJet(2.3, -2.7, 1e-3);
  J y = MakeJet(1.7,  0.5, 1e+2);

  VL << "x = " << x;
  VL << "y = " << y;

  { // Check that log(exp(x)) == x.
    J z = exp(x);
    J w = log(z);
    VL << "z = " << z;
    VL << "w = " << w;
    ExpectJetsClose(w, x);
  }

  { // Check that (x * y) / x == y.
    J z = x * y;
    J w = z / x;
    VL << "z = " << z;
    VL << "w = " << w;
    ExpectJetsClose(w, y);
  }

  { // Check that sqrt(x * x) == x.
    J z = x * x;
    J w = sqrt(z);
    VL << "z = " << z;
    VL << "w = " << w;
    ExpectJetsClose(w, x);
  }

  { // Check that sqrt(y) * sqrt(y) == y.
    J z = sqrt(y);
    J w = z * z;
    VL << "z = " << z;
    VL << "w = " << w;
    ExpectJetsClose(w, y);
  }

  { // Check that cos(2*x) = cos(x)^2 - sin(x)^2
    J z = cos(J(2.0) * x);
    J w = cos(x)*cos(x) - sin(x)*sin(x);
    VL << "z = " << z;
    VL << "w = " << w;
    ExpectJetsClose(w, z);
  }

  { // Check that sin(2*x) = 2*cos(x)*sin(x)
    J z = sin(J(2.0) * x);
    J w = J(2.0)*cos(x)*sin(x);
    VL << "z = " << z;
    VL << "w = " << w;
    ExpectJetsClose(w, z);
  }

  { // Check that cos(x)*cos(x) + sin(x)*sin(x) = 1
    J z = cos(x) * cos(x);
    J w = sin(x) * sin(x);
    VL << "z = " << z;
    VL << "w = " << w;
    ExpectJetsClose(z + w, J(1.0));
  }

  { // Check that atan2(r*sin(t), r*cos(t)) = t.
    J t = MakeJet(0.7, -0.3, +1.5);
    J r = MakeJet(2.3, 0.13, -2.4);
    VL << "t = " << t;
    VL << "r = " << r;

    J u = atan2(r * sin(t), r * cos(t));
    VL << "u = " << u;

    ExpectJetsClose(u, t);
  }

  { // Check that pow(x, 1) == x.
    VL << "x = " << x;

    J u = pow(x, 1.);
    VL << "u = " << u;

    ExpectJetsClose(x, u);
  }

  { // Check that pow(x, 1) == x.
    J y = MakeJet(1, 0.0, 0.0);
    VL << "x = " << x;
    VL << "y = " << y;

    J u = pow(x, y);
    VL << "u = " << u;

    ExpectJetsClose(x, u);
  }

  { // Check that pow(e, log(x)) == x.
    J logx = log(x);

    VL << "x = " << x;
    VL << "y = " << y;

    J u = pow(kE, logx);
    VL << "u = " << u;

    ExpectJetsClose(x, u);
  }

  { // Check that pow(e, log(x)) == x.
    J logx = log(x);
    J e = MakeJet(kE, 0., 0.);
    VL << "x = " << x;
    VL << "log(x) = " << logx;

    J u = pow(e, logx);
    VL << "u = " << u;

    ExpectJetsClose(x, u);
  }

  { // Check that pow(e, log(x)) == x.
    J logx = log(x);
    J e = MakeJet(kE, 0., 0.);
    VL << "x = " << x;
    VL << "logx = " << logx;

    J u = pow(e, logx);
    VL << "u = " << u;

    ExpectJetsClose(x, u);
  }

  { // Check that pow(x,y) = exp(y*log(x)).
    J logx = log(x);
    J e = MakeJet(kE, 0., 0.);
    VL << "x = " << x;
    VL << "logx = " << logx;

    J u = pow(e, y*logx);
    J v = pow(x, y);
    VL << "u = " << u;
    VL << "v = " << v;

    ExpectJetsClose(v, u);
  }

  { // Check that 1 + x == x + 1.
    J a = x + 1.0;
    J b = 1.0 + x;

    ExpectJetsClose(a, b);
  }

  { // Check that 1 - x == -(x - 1).
    J a = 1.0 - x;
    J b = -(x - 1.0);

    ExpectJetsClose(a, b);
  }

  { // Check that x/s == x*s.
    J a = x / 5.0;
    J b = x * 5.0;

    ExpectJetsClose(5.0 * a, b / 5.0);
  }

  { // Check that x / y == 1 / (y / x).
    J a = x / y;
    J b = 1.0 / (y / x);
    VL << "a = " << a;
    VL << "b = " << b;

    ExpectJetsClose(a, b);
  }

  { // Check that abs(-x * x) == sqrt(x * x).
    ExpectJetsClose(abs(-x), sqrt(x * x));
  }

  { // Check that cos(acos(x)) == x.
    J a = MakeJet(0.1, -2.7, 1e-3);
    ExpectJetsClose(cos(acos(a)), a);
    ExpectJetsClose(acos(cos(a)), a);

    J b = MakeJet(0.6,  0.5, 1e+2);
    ExpectJetsClose(cos(acos(b)), b);
    ExpectJetsClose(acos(cos(b)), b);
  }

  { // Check that sin(asin(x)) == x.
    J a = MakeJet(0.1, -2.7, 1e-3);
    ExpectJetsClose(sin(asin(a)), a);
    ExpectJetsClose(asin(sin(a)), a);

    J b = MakeJet(0.4,  0.5, 1e+2);
    ExpectJetsClose(sin(asin(b)), b);
    ExpectJetsClose(asin(sin(b)), b);
  }
}

TEST(Jet, JetsInEigenMatrices) {
  J x = MakeJet(2.3, -2.7, 1e-3);
  J y = MakeJet(1.7,  0.5, 1e+2);
  J z = MakeJet(5.3, -4.7, 1e-3);
  J w = MakeJet(9.7,  1.5, 10.1);

  Eigen::Matrix<J, 2, 2> M;
  Eigen::Matrix<J, 2, 1> v, r1, r2;

  M << x, y, z, w;
  v << x, z;

  // Check that M * v == (v^T * M^T)^T
  r1 = M * v;
  r2 = (v.transpose() * M.transpose()).transpose();

  ExpectJetsClose(r1(0), r2(0));
  ExpectJetsClose(r1(1), r2(1));
}

TEST(JetTraitsTest, ClassificationMixed) {
  Jet<double, 3> a(5.5, 0);
  a.v[0] = std::numeric_limits<double>::quiet_NaN();
  a.v[1] = std::numeric_limits<double>::infinity();
  a.v[2] = -std::numeric_limits<double>::infinity();
  EXPECT_FALSE(IsFinite(a));
  EXPECT_FALSE(IsNormal(a));
  EXPECT_TRUE(IsInfinite(a));
  EXPECT_TRUE(IsNaN(a));
}

TEST(JetTraitsTest, ClassificationNaN) {
  Jet<double, 3> a(5.5, 0);
  a.v[0] = std::numeric_limits<double>::quiet_NaN();
  a.v[1] = 0.0;
  a.v[2] = 0.0;
  EXPECT_FALSE(IsFinite(a));
  EXPECT_FALSE(IsNormal(a));
  EXPECT_FALSE(IsInfinite(a));
  EXPECT_TRUE(IsNaN(a));
}

TEST(JetTraitsTest, ClassificationInf) {
  Jet<double, 3> a(5.5, 0);
  a.v[0] = std::numeric_limits<double>::infinity();
  a.v[1] = 0.0;
  a.v[2] = 0.0;
  EXPECT_FALSE(IsFinite(a));
  EXPECT_FALSE(IsNormal(a));
  EXPECT_TRUE(IsInfinite(a));
  EXPECT_FALSE(IsNaN(a));
}

TEST(JetTraitsTest, ClassificationFinite) {
  Jet<double, 3> a(5.5, 0);
  a.v[0] = 100.0;
  a.v[1] = 1.0;
  a.v[2] = 3.14159;
  EXPECT_TRUE(IsFinite(a));
  EXPECT_TRUE(IsNormal(a));
  EXPECT_FALSE(IsInfinite(a));
  EXPECT_FALSE(IsNaN(a));
}

}  // namespace internal
}  // namespace ceres
