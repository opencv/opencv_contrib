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

#include "ceres/internal/autodiff.h"

#include "gtest/gtest.h"
#include "ceres/random.h"

namespace ceres {
namespace internal {

template <typename T> inline
T &RowMajorAccess(T *base, int rows, int cols, int i, int j) {
  return base[cols * i + j];
}

// Do (symmetric) finite differencing using the given function object 'b' of
// type 'B' and scalar type 'T' with step size 'del'.
//
// The type B should have a signature
//
//   bool operator()(T const *, T *) const;
//
// which maps a vector of parameters to a vector of outputs.
template <typename B, typename T, int M, int N> inline
bool SymmetricDiff(const B& b,
                   const T par[N],
                   T del,           // step size.
                   T fun[M],
                   T jac[M * N]) {  // row-major.
  if (!b(par, fun)) {
    return false;
  }

  // Temporary parameter vector.
  T tmp_par[N];
  for (int j = 0; j < N; ++j) {
    tmp_par[j] = par[j];
  }

  // For each dimension, we do one forward step and one backward step in
  // parameter space, and store the output vector vectors in these vectors.
  T fwd_fun[M];
  T bwd_fun[M];

  for (int j = 0; j < N; ++j) {
    // Forward step.
    tmp_par[j] = par[j] + del;
    if (!b(tmp_par, fwd_fun)) {
      return false;
    }

    // Backward step.
    tmp_par[j] = par[j] - del;
    if (!b(tmp_par, bwd_fun)) {
      return false;
    }

    // Symmetric differencing:
    //   f'(a) = (f(a + h) - f(a - h)) / (2 h)
    for (int i = 0; i < M; ++i) {
      RowMajorAccess(jac, M, N, i, j) =
          (fwd_fun[i] - bwd_fun[i]) / (T(2) * del);
    }

    // Restore our temporary vector.
    tmp_par[j] = par[j];
  }

  return true;
}

template <typename A> inline
void QuaternionToScaledRotation(A const q[4], A R[3 * 3]) {
  // Make convenient names for elements of q.
  A a = q[0];
  A b = q[1];
  A c = q[2];
  A d = q[3];
  // This is not to eliminate common sub-expression, but to
  // make the lines shorter so that they fit in 80 columns!
  A aa = a*a;
  A ab = a*b;
  A ac = a*c;
  A ad = a*d;
  A bb = b*b;
  A bc = b*c;
  A bd = b*d;
  A cc = c*c;
  A cd = c*d;
  A dd = d*d;
#define R(i, j) RowMajorAccess(R, 3, 3, (i), (j))
  R(0, 0) =  aa+bb-cc-dd; R(0, 1) = A(2)*(bc-ad); R(0, 2) = A(2)*(ac+bd);  // NOLINT
  R(1, 0) = A(2)*(ad+bc); R(1, 1) =  aa-bb+cc-dd; R(1, 2) = A(2)*(cd-ab);  // NOLINT
  R(2, 0) = A(2)*(bd-ac); R(2, 1) = A(2)*(ab+cd); R(2, 2) =  aa-bb-cc+dd;  // NOLINT
#undef R
}

// A structure for projecting a 3x4 camera matrix and a
// homogeneous 3D point, to a 2D inhomogeneous point.
struct Projective {
  // Function that takes P and X as separate vectors:
  //   P, X -> x
  template <typename A>
  bool operator()(A const P[12], A const X[4], A x[2]) const {
    A PX[3];
    for (int i = 0; i < 3; ++i) {
      PX[i] = RowMajorAccess(P, 3, 4, i, 0) * X[0] +
              RowMajorAccess(P, 3, 4, i, 1) * X[1] +
              RowMajorAccess(P, 3, 4, i, 2) * X[2] +
              RowMajorAccess(P, 3, 4, i, 3) * X[3];
    }
    if (PX[2] != 0.0) {
      x[0] = PX[0] / PX[2];
      x[1] = PX[1] / PX[2];
      return true;
    }
    return false;
  }

  // Version that takes P and X packed in one vector:
  //
  //   (P, X) -> x
  //
  template <typename A>
  bool operator()(A const P_X[12 + 4], A x[2]) const {
    return operator()(P_X + 0, P_X + 12, x);
  }
};

// Test projective camera model projector.
TEST(AutoDiff, ProjectiveCameraModel) {
  srand(5);
  double const tol = 1e-10;  // floating-point tolerance.
  double const del = 1e-4;   // finite-difference step.
  double const err = 1e-6;   // finite-difference tolerance.

  Projective b;

  // Make random P and X, in a single vector.
  double PX[12 + 4];
  for (int i = 0; i < 12 + 4; ++i) {
    PX[i] = RandDouble();
  }

  // Handy names for the P and X parts.
  double *P = PX + 0;
  double *X = PX + 12;

  // Apply the mapping, to get image point b_x.
  double b_x[2];
  b(P, X, b_x);

  // Use finite differencing to estimate the Jacobian.
  double fd_x[2];
  double fd_J[2 * (12 + 4)];
  ASSERT_TRUE((SymmetricDiff<Projective, double, 2, 12 + 4>(b, PX, del,
                                                            fd_x, fd_J)));

  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(fd_x[i], b_x[i]);
  }

  // Use automatic differentiation to compute the Jacobian.
  double ad_x1[2];
  double J_PX[2 * (12 + 4)];
  {
    double *parameters[] = { PX };
    double *jacobians[] = { J_PX };
    ASSERT_TRUE((AutoDiff<Projective, double, 12 + 4>::Differentiate(
        b, parameters, 2, ad_x1, jacobians)));

    for (int i = 0; i < 2; ++i) {
      ASSERT_NEAR(ad_x1[i], b_x[i], tol);
    }
  }

  // Use automatic differentiation (again), with two arguments.
  {
    double ad_x2[2];
    double J_P[2 * 12];
    double J_X[2 * 4];
    double *parameters[] = { P, X };
    double *jacobians[] = { J_P, J_X };
    ASSERT_TRUE((AutoDiff<Projective, double, 12, 4>::Differentiate(
        b, parameters, 2, ad_x2, jacobians)));

    for (int i = 0; i < 2; ++i) {
      ASSERT_NEAR(ad_x2[i], b_x[i], tol);
    }

    // Now compare the jacobians we got.
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 12 + 4; ++j) {
        ASSERT_NEAR(J_PX[(12 + 4) * i + j], fd_J[(12 + 4) * i + j], err);
      }

      for (int j = 0; j < 12; ++j) {
        ASSERT_NEAR(J_PX[(12 + 4) * i + j], J_P[12 * i + j], tol);
      }
      for (int j = 0; j < 4; ++j) {
        ASSERT_NEAR(J_PX[(12 + 4) * i + 12 + j], J_X[4 * i + j], tol);
      }
    }
  }
}

// Object to implement the projection by a calibrated camera.
struct Metric {
  // The mapping is
  //
  //   q, c, X -> x = dehomg(R(q) (X - c))
  //
  // where q is a quaternion and c is the center of projection.
  //
  // This function takes three input vectors.
  template <typename A>
  bool operator()(A const q[4], A const c[3], A const X[3], A x[2]) const {
    A R[3 * 3];
    QuaternionToScaledRotation(q, R);

    // Convert the quaternion mapping all the way to projective matrix.
    A P[3 * 4];

    // Set P(:, 1:3) = R
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        RowMajorAccess(P, 3, 4, i, j) = RowMajorAccess(R, 3, 3, i, j);
      }
    }

    // Set P(:, 4) = - R c
    for (int i = 0; i < 3; ++i) {
      RowMajorAccess(P, 3, 4, i, 3) =
        - (RowMajorAccess(R, 3, 3, i, 0) * c[0] +
           RowMajorAccess(R, 3, 3, i, 1) * c[1] +
           RowMajorAccess(R, 3, 3, i, 2) * c[2]);
    }

    A X1[4] = { X[0], X[1], X[2], A(1) };
    Projective p;
    return p(P, X1, x);
  }

  // A version that takes a single vector.
  template <typename A>
  bool operator()(A const q_c_X[4 + 3 + 3], A x[2]) const {
    return operator()(q_c_X, q_c_X + 4, q_c_X + 4 + 3, x);
  }
};

// This test is similar in structure to the previous one.
TEST(AutoDiff, Metric) {
  srand(5);
  double const tol = 1e-10;  // floating-point tolerance.
  double const del = 1e-4;   // finite-difference step.
  double const err = 1e-5;   // finite-difference tolerance.

  Metric b;

  // Make random parameter vector.
  double qcX[4 + 3 + 3];
  for (int i = 0; i < 4 + 3 + 3; ++i)
    qcX[i] = RandDouble();

  // Handy names.
  double *q = qcX;
  double *c = qcX + 4;
  double *X = qcX + 4 + 3;

  // Compute projection, b_x.
  double b_x[2];
  ASSERT_TRUE(b(q, c, X, b_x));

  // Finite differencing estimate of Jacobian.
  double fd_x[2];
  double fd_J[2 * (4 + 3 + 3)];
  ASSERT_TRUE((SymmetricDiff<Metric, double, 2, 4 + 3 + 3>(b, qcX, del,
                                                           fd_x, fd_J)));

  for (int i = 0; i < 2; ++i) {
    ASSERT_NEAR(fd_x[i], b_x[i], tol);
  }

  // Automatic differentiation.
  double ad_x[2];
  double J_q[2 * 4];
  double J_c[2 * 3];
  double J_X[2 * 3];
  double *parameters[] = { q, c, X };
  double *jacobians[] = { J_q, J_c, J_X };
  ASSERT_TRUE((AutoDiff<Metric, double, 4, 3, 3>::Differentiate(
      b, parameters, 2, ad_x, jacobians)));

  for (int i = 0; i < 2; ++i) {
    ASSERT_NEAR(ad_x[i], b_x[i], tol);
  }

  // Compare the pieces.
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      ASSERT_NEAR(J_q[4 * i + j], fd_J[(4 + 3 + 3) * i + j], err);
    }
    for (int j = 0; j < 3; ++j) {
      ASSERT_NEAR(J_c[3 * i + j], fd_J[(4 + 3 + 3) * i + j + 4], err);
    }
    for (int j = 0; j < 3; ++j) {
      ASSERT_NEAR(J_X[3 * i + j], fd_J[(4 + 3 + 3) * i + j + 4 + 3], err);
    }
  }
}

struct VaryingResidualFunctor {
  template <typename T>
  bool operator()(const T x[2], T* y) const {
    for (int i = 0; i < num_residuals; ++i) {
      y[i] = T(i) * x[0] * x[1] * x[1];
    }
    return true;
  }

  int num_residuals;
};

TEST(AutoDiff, VaryingNumberOfResidualsForOneCostFunctorType) {
  double x[2] = { 1.0, 5.5 };
  double *parameters[] = { x };
  const int kMaxResiduals = 10;
  double J_x[2 * kMaxResiduals];
  double residuals[kMaxResiduals];
  double *jacobians[] = { J_x };

  // Use a single functor, but tweak it to produce different numbers of
  // residuals.
  VaryingResidualFunctor functor;

  for (int num_residuals = 1; num_residuals < kMaxResiduals; ++num_residuals) {
    // Tweak the number of residuals to produce.
    functor.num_residuals = num_residuals;

    // Run autodiff with the new number of residuals.
    ASSERT_TRUE((AutoDiff<VaryingResidualFunctor, double, 2>::Differentiate(
        functor, parameters, num_residuals, residuals, jacobians)));

    const double kTolerance = 1e-14;
    for (int i = 0; i < num_residuals; ++i) {
      EXPECT_NEAR(J_x[2 * i + 0], i * x[1] * x[1], kTolerance) << "i: " << i;
      EXPECT_NEAR(J_x[2 * i + 1], 2 * i * x[0] * x[1], kTolerance)
          << "i: " << i;
    }
  }
}

struct Residual1Param {
  template <typename T>
  bool operator()(const T* x0, T* y) const {
    y[0] = *x0;
    return true;
  }
};

struct Residual2Param {
  template <typename T>
  bool operator()(const T* x0, const T* x1, T* y) const {
    y[0] = *x0 + pow(*x1, 2);
    return true;
  }
};

struct Residual3Param {
  template <typename T>
  bool operator()(const T* x0, const T* x1, const T* x2, T* y) const {
    y[0] = *x0 + pow(*x1, 2) + pow(*x2, 3);
    return true;
  }
};

struct Residual4Param {
  template <typename T>
  bool operator()(const T* x0,
                  const T* x1,
                  const T* x2,
                  const T* x3,
                  T* y) const {
    y[0] = *x0 + pow(*x1, 2) + pow(*x2, 3) + pow(*x3, 4);
    return true;
  }
};

struct Residual5Param {
  template <typename T>
  bool operator()(const T* x0,
                  const T* x1,
                  const T* x2,
                  const T* x3,
                  const T* x4,
                  T* y) const {
    y[0] = *x0 + pow(*x1, 2) + pow(*x2, 3) + pow(*x3, 4) + pow(*x4, 5);
    return true;
  }
};

struct Residual6Param {
  template <typename T>
  bool operator()(const T* x0,
                  const T* x1,
                  const T* x2,
                  const T* x3,
                  const T* x4,
                  const T* x5,
                  T* y) const {
    y[0] = *x0 + pow(*x1, 2) + pow(*x2, 3) + pow(*x3, 4) + pow(*x4, 5) +
        pow(*x5, 6);
    return true;
  }
};

struct Residual7Param {
  template <typename T>
  bool operator()(const T* x0,
                  const T* x1,
                  const T* x2,
                  const T* x3,
                  const T* x4,
                  const T* x5,
                  const T* x6,
                  T* y) const {
    y[0] = *x0 + pow(*x1, 2) + pow(*x2, 3) + pow(*x3, 4) + pow(*x4, 5) +
        pow(*x5, 6) + pow(*x6, 7);
    return true;
  }
};

struct Residual8Param {
  template <typename T>
  bool operator()(const T* x0,
                  const T* x1,
                  const T* x2,
                  const T* x3,
                  const T* x4,
                  const T* x5,
                  const T* x6,
                  const T* x7,
                  T* y) const {
    y[0] = *x0 + pow(*x1, 2) + pow(*x2, 3) + pow(*x3, 4) + pow(*x4, 5) +
        pow(*x5, 6) + pow(*x6, 7) + pow(*x7, 8);
    return true;
  }
};

struct Residual9Param {
  template <typename T>
  bool operator()(const T* x0,
                  const T* x1,
                  const T* x2,
                  const T* x3,
                  const T* x4,
                  const T* x5,
                  const T* x6,
                  const T* x7,
                  const T* x8,
                  T* y) const {
    y[0] = *x0 + pow(*x1, 2) + pow(*x2, 3) + pow(*x3, 4) + pow(*x4, 5) +
        pow(*x5, 6) + pow(*x6, 7) + pow(*x7, 8) + pow(*x8, 9);
    return true;
  }
};

struct Residual10Param {
  template <typename T>
  bool operator()(const T* x0,
                  const T* x1,
                  const T* x2,
                  const T* x3,
                  const T* x4,
                  const T* x5,
                  const T* x6,
                  const T* x7,
                  const T* x8,
                  const T* x9,
                  T* y) const {
    y[0] = *x0 + pow(*x1, 2) + pow(*x2, 3) + pow(*x3, 4) + pow(*x4, 5) +
        pow(*x5, 6) + pow(*x6, 7) + pow(*x7, 8) + pow(*x8, 9) + pow(*x9, 10);
    return true;
  }
};

TEST(AutoDiff, VariadicAutoDiff) {
  double x[10];
  double residual = 0;
  double* parameters[10];
  double jacobian_values[10];
  double* jacobians[10];

  for (int i = 0; i < 10; ++i) {
    x[i] = 2.0;
    parameters[i] = x + i;
    jacobians[i] = jacobian_values + i;
  }

  {
    Residual1Param functor;
    int num_variables = 1;
    EXPECT_TRUE((AutoDiff<Residual1Param, double, 1>::Differentiate(
                     functor, parameters, 1, &residual, jacobians)));
    EXPECT_EQ(residual, pow(2, num_variables + 1) - 2);
    for (int i = 0; i < num_variables; ++i) {
      EXPECT_EQ(jacobian_values[i], (i + 1) * pow(2, i));
    }
  }

  {
    Residual2Param functor;
    int num_variables = 2;
    EXPECT_TRUE((AutoDiff<Residual2Param, double, 1, 1>::Differentiate(
                     functor, parameters, 1, &residual, jacobians)));
    EXPECT_EQ(residual, pow(2, num_variables + 1) - 2);
    for (int i = 0; i < num_variables; ++i) {
      EXPECT_EQ(jacobian_values[i], (i + 1) * pow(2, i));
    }
  }

  {
    Residual3Param functor;
    int num_variables = 3;
    EXPECT_TRUE((AutoDiff<Residual3Param, double, 1, 1, 1>::Differentiate(
                     functor, parameters, 1, &residual, jacobians)));
    EXPECT_EQ(residual, pow(2, num_variables + 1) - 2);
    for (int i = 0; i < num_variables; ++i) {
      EXPECT_EQ(jacobian_values[i], (i + 1) * pow(2, i));
    }
  }

  {
    Residual4Param functor;
    int num_variables = 4;
    EXPECT_TRUE((AutoDiff<Residual4Param, double, 1, 1, 1, 1>::Differentiate(
                     functor, parameters, 1, &residual, jacobians)));
    EXPECT_EQ(residual, pow(2, num_variables + 1) - 2);
    for (int i = 0; i < num_variables; ++i) {
      EXPECT_EQ(jacobian_values[i], (i + 1) * pow(2, i));
    }
  }

  {
    Residual5Param functor;
    int num_variables = 5;
    EXPECT_TRUE((AutoDiff<Residual5Param, double, 1, 1, 1, 1, 1>::Differentiate(
                     functor, parameters, 1, &residual, jacobians)));
    EXPECT_EQ(residual, pow(2, num_variables + 1) - 2);
    for (int i = 0; i < num_variables; ++i) {
      EXPECT_EQ(jacobian_values[i], (i + 1) * pow(2, i));
    }
  }

  {
    Residual6Param functor;
    int num_variables = 6;
    EXPECT_TRUE((AutoDiff<Residual6Param,
                 double,
                 1, 1, 1, 1, 1, 1>::Differentiate(
                     functor, parameters, 1, &residual, jacobians)));
    EXPECT_EQ(residual, pow(2, num_variables + 1) - 2);
    for (int i = 0; i < num_variables; ++i) {
      EXPECT_EQ(jacobian_values[i], (i + 1) * pow(2, i));
    }
  }

  {
    Residual7Param functor;
    int num_variables = 7;
    EXPECT_TRUE((AutoDiff<Residual7Param,
                 double,
                 1, 1, 1, 1, 1, 1, 1>::Differentiate(
                     functor, parameters, 1, &residual, jacobians)));
    EXPECT_EQ(residual, pow(2, num_variables + 1) - 2);
    for (int i = 0; i < num_variables; ++i) {
      EXPECT_EQ(jacobian_values[i], (i + 1) * pow(2, i));
    }
  }

  {
    Residual8Param functor;
    int num_variables = 8;
    EXPECT_TRUE((AutoDiff<
                 Residual8Param,
                 double, 1, 1, 1, 1, 1, 1, 1, 1>::Differentiate(
                     functor, parameters, 1, &residual, jacobians)));
    EXPECT_EQ(residual, pow(2, num_variables + 1) - 2);
    for (int i = 0; i < num_variables; ++i) {
      EXPECT_EQ(jacobian_values[i], (i + 1) * pow(2, i));
    }
  }

  {
    Residual9Param functor;
    int num_variables = 9;
    EXPECT_TRUE((AutoDiff<
                 Residual9Param,
                 double,
                 1, 1, 1, 1, 1, 1, 1, 1, 1>::Differentiate(
                     functor, parameters, 1, &residual, jacobians)));
    EXPECT_EQ(residual, pow(2, num_variables + 1) - 2);
    for (int i = 0; i < num_variables; ++i) {
      EXPECT_EQ(jacobian_values[i], (i + 1) * pow(2, i));
    }
  }

  {
    Residual10Param functor;
    int num_variables = 10;
    EXPECT_TRUE((AutoDiff<
                 Residual10Param,
                 double,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>::Differentiate(
                     functor, parameters, 1, &residual, jacobians)));
    EXPECT_EQ(residual, pow(2, num_variables + 1) - 2);
    for (int i = 0; i < num_variables; ++i) {
      EXPECT_EQ(jacobian_values[i], (i + 1) * pow(2, i));
    }
  }
}

// This is fragile test that triggers the alignment bug on
// i686-apple-darwin10-llvm-g++-4.2 (GCC) 4.2.1. It is quite possible,
// that other combinations of operating system + compiler will
// re-arrange the operations in this test.
//
// But this is the best (and only) way we know of to trigger this
// problem for now. A more robust solution that guarantees the
// alignment of Eigen types used for automatic differentiation would
// be nice.
TEST(AutoDiff, AlignedAllocationTest) {
  // This int is needed to allocate 16 bits on the stack, so that the
  // next allocation is not aligned by default.
  char y = 0;

  // This is needed to prevent the compiler from optimizing y out of
  // this function.
  y += 1;

  typedef Jet<double, 2> JetT;
  FixedArray<JetT, (256 * 7) / sizeof(JetT)> x(3);

  // Need this to makes sure that x does not get optimized out.
  x[0] = x[0] + JetT(1.0);
}

}  // namespace internal
}  // namespace ceres
