// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2012 Google Inc. All rights reserved.
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
// Author: moll.markus@arcor.de (Markus Moll)
//         sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/polynomial.h"

#include <limits>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include "gtest/gtest.h"
#include "ceres/test_util.h"

namespace ceres {
namespace internal {
namespace {

// For IEEE-754 doubles, machine precision is about 2e-16.
const double kEpsilon = 1e-13;
const double kEpsilonLoose = 1e-9;

// Return the constant polynomial p(x) = 1.23.
Vector ConstantPolynomial(double value) {
  Vector poly(1);
  poly(0) = value;
  return poly;
}

// Return the polynomial p(x) = poly(x) * (x - root).
Vector AddRealRoot(const Vector& poly, double root) {
  Vector poly2(poly.size() + 1);
  poly2.setZero();
  poly2.head(poly.size()) += poly;
  poly2.tail(poly.size()) -= root * poly;
  return poly2;
}

// Return the polynomial
// p(x) = poly(x) * (x - real - imag*i) * (x - real + imag*i).
Vector AddComplexRootPair(const Vector& poly, double real, double imag) {
  Vector poly2(poly.size() + 2);
  poly2.setZero();
  // Multiply poly by x^2 - 2real + abs(real,imag)^2
  poly2.head(poly.size()) += poly;
  poly2.segment(1, poly.size()) -= 2 * real * poly;
  poly2.tail(poly.size()) += (real*real + imag*imag) * poly;
  return poly2;
}

// Sort the entries in a vector.
// Needed because the roots are not returned in sorted order.
Vector SortVector(const Vector& in) {
  Vector out(in);
  std::sort(out.data(), out.data() + out.size());
  return out;
}

// Run a test with the polynomial defined by the N real roots in roots_real.
// If use_real is false, NULL is passed as the real argument to
// FindPolynomialRoots. If use_imaginary is false, NULL is passed as the
// imaginary argument to FindPolynomialRoots.
template<int N>
void RunPolynomialTestRealRoots(const double (&real_roots)[N],
                                bool use_real,
                                bool use_imaginary,
                                double epsilon) {
  Vector real;
  Vector imaginary;
  Vector poly = ConstantPolynomial(1.23);
  for (int i = 0; i < N; ++i) {
    poly = AddRealRoot(poly, real_roots[i]);
  }
  Vector* const real_ptr = use_real ? &real : NULL;
  Vector* const imaginary_ptr = use_imaginary ? &imaginary : NULL;
  bool success = FindPolynomialRoots(poly, real_ptr, imaginary_ptr);

  EXPECT_EQ(success, true);
  if (use_real) {
    EXPECT_EQ(real.size(), N);
    real = SortVector(real);
    ExpectArraysClose(N, real.data(), real_roots, epsilon);
  }
  if (use_imaginary) {
    EXPECT_EQ(imaginary.size(), N);
    const Vector zeros = Vector::Zero(N);
    ExpectArraysClose(N, imaginary.data(), zeros.data(), epsilon);
  }
}
}  // namespace

TEST(Polynomial, InvalidPolynomialOfZeroLengthIsRejected) {
  // Vector poly(0) is an ambiguous constructor call, so
  // use the constructor with explicit column count.
  Vector poly(0, 1);
  Vector real;
  Vector imag;
  bool success = FindPolynomialRoots(poly, &real, &imag);

  EXPECT_EQ(success, false);
}

TEST(Polynomial, ConstantPolynomialReturnsNoRoots) {
  Vector poly = ConstantPolynomial(1.23);
  Vector real;
  Vector imag;
  bool success = FindPolynomialRoots(poly, &real, &imag);

  EXPECT_EQ(success, true);
  EXPECT_EQ(real.size(), 0);
  EXPECT_EQ(imag.size(), 0);
}

TEST(Polynomial, LinearPolynomialWithPositiveRootWorks) {
  const double roots[1] = { 42.42 };
  RunPolynomialTestRealRoots(roots, true, true, kEpsilon);
}

TEST(Polynomial, LinearPolynomialWithNegativeRootWorks) {
  const double roots[1] = { -42.42 };
  RunPolynomialTestRealRoots(roots, true, true, kEpsilon);
}

TEST(Polynomial, QuadraticPolynomialWithPositiveRootsWorks) {
  const double roots[2] = { 1.0, 42.42 };
  RunPolynomialTestRealRoots(roots, true, true, kEpsilon);
}

TEST(Polynomial, QuadraticPolynomialWithOneNegativeRootWorks) {
  const double roots[2] = { -42.42, 1.0 };
  RunPolynomialTestRealRoots(roots, true, true, kEpsilon);
}

TEST(Polynomial, QuadraticPolynomialWithTwoNegativeRootsWorks) {
  const double roots[2] = { -42.42, -1.0 };
  RunPolynomialTestRealRoots(roots, true, true, kEpsilon);
}

TEST(Polynomial, QuadraticPolynomialWithCloseRootsWorks) {
  const double roots[2] = { 42.42, 42.43 };
  RunPolynomialTestRealRoots(roots, true, false, kEpsilonLoose);
}

TEST(Polynomial, QuadraticPolynomialWithComplexRootsWorks) {
  Vector real;
  Vector imag;

  Vector poly = ConstantPolynomial(1.23);
  poly = AddComplexRootPair(poly, 42.42, 4.2);
  bool success = FindPolynomialRoots(poly, &real, &imag);

  EXPECT_EQ(success, true);
  EXPECT_EQ(real.size(), 2);
  EXPECT_EQ(imag.size(), 2);
  ExpectClose(real(0), 42.42, kEpsilon);
  ExpectClose(real(1), 42.42, kEpsilon);
  ExpectClose(std::abs(imag(0)), 4.2, kEpsilon);
  ExpectClose(std::abs(imag(1)), 4.2, kEpsilon);
  ExpectClose(std::abs(imag(0) + imag(1)), 0.0, kEpsilon);
}

TEST(Polynomial, QuarticPolynomialWorks) {
  const double roots[4] = { 1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5 };
  RunPolynomialTestRealRoots(roots, true, true, kEpsilon);
}

TEST(Polynomial, QuarticPolynomialWithTwoClustersOfCloseRootsWorks) {
  const double roots[4] = { 1.23e-1, 2.46e-1, 1.23e+5, 2.46e+5 };
  RunPolynomialTestRealRoots(roots, true, true, kEpsilonLoose);
}

TEST(Polynomial, QuarticPolynomialWithTwoZeroRootsWorks) {
  const double roots[4] = { -42.42, 0.0, 0.0, 42.42 };
  RunPolynomialTestRealRoots(roots, true, true, kEpsilonLoose);
}

TEST(Polynomial, QuarticMonomialWorks) {
  const double roots[4] = { 0.0, 0.0, 0.0, 0.0 };
  RunPolynomialTestRealRoots(roots, true, true, kEpsilon);
}

TEST(Polynomial, NullPointerAsImaginaryPartWorks) {
  const double roots[4] = { 1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5 };
  RunPolynomialTestRealRoots(roots, true, false, kEpsilon);
}

TEST(Polynomial, NullPointerAsRealPartWorks) {
  const double roots[4] = { 1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5 };
  RunPolynomialTestRealRoots(roots, false, true, kEpsilon);
}

TEST(Polynomial, BothOutputArgumentsNullWorks) {
  const double roots[4] = { 1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5 };
  RunPolynomialTestRealRoots(roots, false, false, kEpsilon);
}

TEST(Polynomial, DifferentiateConstantPolynomial) {
  // p(x) = 1;
  Vector polynomial(1);
  polynomial(0) = 1.0;
  const Vector derivative = DifferentiatePolynomial(polynomial);
  EXPECT_EQ(derivative.rows(), 1);
  EXPECT_EQ(derivative(0), 0);
}

TEST(Polynomial, DifferentiateQuadraticPolynomial) {
  // p(x) = x^2 + 2x + 3;
  Vector polynomial(3);
  polynomial(0) = 1.0;
  polynomial(1) = 2.0;
  polynomial(2) = 3.0;

  const Vector derivative = DifferentiatePolynomial(polynomial);
  EXPECT_EQ(derivative.rows(), 2);
  EXPECT_EQ(derivative(0), 2.0);
  EXPECT_EQ(derivative(1), 2.0);
}

TEST(Polynomial, MinimizeConstantPolynomial) {
  // p(x) = 1;
  Vector polynomial(1);
  polynomial(0) = 1.0;

  double optimal_x = 0.0;
  double optimal_value = 0.0;
  double min_x = 0.0;
  double max_x = 1.0;
  MinimizePolynomial(polynomial, min_x, max_x, &optimal_x, &optimal_value);

  EXPECT_EQ(optimal_value, 1.0);
  EXPECT_LE(optimal_x, max_x);
  EXPECT_GE(optimal_x, min_x);
}

TEST(Polynomial, MinimizeLinearPolynomial) {
  // p(x) = x - 2
  Vector polynomial(2);

  polynomial(0) = 1.0;
  polynomial(1) = 2.0;

  double optimal_x = 0.0;
  double optimal_value = 0.0;
  double min_x = 0.0;
  double max_x = 1.0;
  MinimizePolynomial(polynomial, min_x, max_x, &optimal_x, &optimal_value);

  EXPECT_EQ(optimal_x, 0.0);
  EXPECT_EQ(optimal_value, 2.0);
}


TEST(Polynomial, MinimizeQuadraticPolynomial) {
  // p(x) = x^2 - 3 x + 2
  // min_x = 3/2
  // min_value = -1/4;
  Vector polynomial(3);
  polynomial(0) = 1.0;
  polynomial(1) = -3.0;
  polynomial(2) = 2.0;

  double optimal_x = 0.0;
  double optimal_value = 0.0;
  double min_x = -2.0;
  double max_x = 2.0;
  MinimizePolynomial(polynomial, min_x, max_x, &optimal_x, &optimal_value);
  EXPECT_EQ(optimal_x, 3.0/2.0);
  EXPECT_EQ(optimal_value, -1.0/4.0);

  min_x = -2.0;
  max_x = 1.0;
  MinimizePolynomial(polynomial, min_x, max_x, &optimal_x, &optimal_value);
  EXPECT_EQ(optimal_x, 1.0);
  EXPECT_EQ(optimal_value, 0.0);

  min_x = 2.0;
  max_x = 3.0;
  MinimizePolynomial(polynomial, min_x, max_x, &optimal_x, &optimal_value);
  EXPECT_EQ(optimal_x, 2.0);
  EXPECT_EQ(optimal_value, 0.0);
}

TEST(Polymomial, ConstantInterpolatingPolynomial) {
  // p(x) = 1.0
  Vector true_polynomial(1);
  true_polynomial << 1.0;

  vector<FunctionSample> samples;
  FunctionSample sample;
  sample.x = 1.0;
  sample.value = 1.0;
  sample.value_is_valid = true;
  samples.push_back(sample);

  const Vector polynomial = FindInterpolatingPolynomial(samples);
  EXPECT_NEAR((true_polynomial - polynomial).norm(), 0.0, 1e-15);
}

TEST(Polynomial, LinearInterpolatingPolynomial) {
  // p(x) = 2x - 1
  Vector true_polynomial(2);
  true_polynomial << 2.0, -1.0;

  vector<FunctionSample> samples;
  FunctionSample sample;
  sample.x = 1.0;
  sample.value = 1.0;
  sample.value_is_valid = true;
  sample.gradient = 2.0;
  sample.gradient_is_valid = true;
  samples.push_back(sample);

  const Vector polynomial = FindInterpolatingPolynomial(samples);
  EXPECT_NEAR((true_polynomial - polynomial).norm(), 0.0, 1e-15);
}

TEST(Polynomial, QuadraticInterpolatingPolynomial) {
  // p(x) = 2x^2 + 3x + 2
  Vector true_polynomial(3);
  true_polynomial << 2.0, 3.0, 2.0;

  vector<FunctionSample> samples;
  {
    FunctionSample sample;
    sample.x = 1.0;
    sample.value = 7.0;
    sample.value_is_valid = true;
    sample.gradient = 7.0;
    sample.gradient_is_valid = true;
    samples.push_back(sample);
  }

  {
    FunctionSample sample;
    sample.x = -3.0;
    sample.value = 11.0;
    sample.value_is_valid = true;
    samples.push_back(sample);
  }

  Vector polynomial = FindInterpolatingPolynomial(samples);
  EXPECT_NEAR((true_polynomial - polynomial).norm(), 0.0, 1e-15);
}

TEST(Polynomial, DeficientCubicInterpolatingPolynomial) {
  // p(x) = 2x^2 + 3x + 2
  Vector true_polynomial(4);
  true_polynomial << 0.0, 2.0, 3.0, 2.0;

  vector<FunctionSample> samples;
  {
    FunctionSample sample;
    sample.x = 1.0;
    sample.value = 7.0;
    sample.value_is_valid = true;
    sample.gradient = 7.0;
    sample.gradient_is_valid = true;
    samples.push_back(sample);
  }

  {
    FunctionSample sample;
    sample.x = -3.0;
    sample.value = 11.0;
    sample.value_is_valid = true;
    sample.gradient = -9;
    sample.gradient_is_valid = true;
    samples.push_back(sample);
  }

  const Vector polynomial = FindInterpolatingPolynomial(samples);
  EXPECT_NEAR((true_polynomial - polynomial).norm(), 0.0, 1e-14);
}


TEST(Polynomial, CubicInterpolatingPolynomialFromValues) {
  // p(x) = x^3 + 2x^2 + 3x + 2
  Vector true_polynomial(4);
  true_polynomial << 1.0, 2.0, 3.0, 2.0;

  vector<FunctionSample> samples;
  {
    FunctionSample sample;
    sample.x = 1.0;
    sample.value = EvaluatePolynomial(true_polynomial, sample.x);
    sample.value_is_valid = true;
    samples.push_back(sample);
  }

  {
    FunctionSample sample;
    sample.x = -3.0;
    sample.value = EvaluatePolynomial(true_polynomial, sample.x);
    sample.value_is_valid = true;
    samples.push_back(sample);
  }

  {
    FunctionSample sample;
    sample.x = 2.0;
    sample.value = EvaluatePolynomial(true_polynomial, sample.x);
    sample.value_is_valid = true;
    samples.push_back(sample);
  }

  {
    FunctionSample sample;
    sample.x = 0.0;
    sample.value = EvaluatePolynomial(true_polynomial, sample.x);
    sample.value_is_valid = true;
    samples.push_back(sample);
  }

  const Vector polynomial = FindInterpolatingPolynomial(samples);
  EXPECT_NEAR((true_polynomial - polynomial).norm(), 0.0, 1e-14);
}

TEST(Polynomial, CubicInterpolatingPolynomialFromValuesAndOneGradient) {
  // p(x) = x^3 + 2x^2 + 3x + 2
  Vector true_polynomial(4);
  true_polynomial << 1.0, 2.0, 3.0, 2.0;
  Vector true_gradient_polynomial = DifferentiatePolynomial(true_polynomial);

  vector<FunctionSample> samples;
  {
    FunctionSample sample;
    sample.x = 1.0;
    sample.value = EvaluatePolynomial(true_polynomial, sample.x);
    sample.value_is_valid = true;
    samples.push_back(sample);
  }

  {
    FunctionSample sample;
    sample.x = -3.0;
    sample.value = EvaluatePolynomial(true_polynomial, sample.x);
    sample.value_is_valid = true;
    samples.push_back(sample);
  }

  {
    FunctionSample sample;
    sample.x = 2.0;
    sample.value = EvaluatePolynomial(true_polynomial, sample.x);
    sample.value_is_valid = true;
    sample.gradient = EvaluatePolynomial(true_gradient_polynomial, sample.x);
    sample.gradient_is_valid = true;
    samples.push_back(sample);
  }

  const Vector polynomial = FindInterpolatingPolynomial(samples);
  EXPECT_NEAR((true_polynomial - polynomial).norm(), 0.0, 1e-14);
}

TEST(Polynomial, CubicInterpolatingPolynomialFromValuesAndGradients) {
  // p(x) = x^3 + 2x^2 + 3x + 2
  Vector true_polynomial(4);
  true_polynomial << 1.0, 2.0, 3.0, 2.0;
  Vector true_gradient_polynomial = DifferentiatePolynomial(true_polynomial);

  vector<FunctionSample> samples;
  {
    FunctionSample sample;
    sample.x = -3.0;
    sample.value = EvaluatePolynomial(true_polynomial, sample.x);
    sample.value_is_valid = true;
    sample.gradient = EvaluatePolynomial(true_gradient_polynomial, sample.x);
    sample.gradient_is_valid = true;
    samples.push_back(sample);
  }

  {
    FunctionSample sample;
    sample.x = 2.0;
    sample.value = EvaluatePolynomial(true_polynomial, sample.x);
    sample.value_is_valid = true;
    sample.gradient = EvaluatePolynomial(true_gradient_polynomial, sample.x);
    sample.gradient_is_valid = true;
    samples.push_back(sample);
  }

  const Vector polynomial = FindInterpolatingPolynomial(samples);
  EXPECT_NEAR((true_polynomial - polynomial).norm(), 0.0, 1e-14);
}

}  // namespace internal
}  // namespace ceres
