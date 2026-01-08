// Copyright (c) 2007, 2008 libmv authors.
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

#include "libmv/numeric/numeric.h"
#include "libmv/multiview/projection.h"
#include "libmv/multiview/twoviewtriangulation.h"
#include "libmv/multiview/projection.h"


namespace libmv {

void TwoViewTriangulationByPlanes(const Vec3 &x1, const Vec3 &x2,
                                  const Mat34 &P, const Mat3 &E,
                                  Vec4 *X) {
  Vec3 a = E.transpose() * x2;
  Vec3 a0 = a;
  a0(2) = 0;
  Vec3 b = x1;
  b = b.cross(a0);

  Vec3 c = E * x1;
  c(2) = 0;
  c = c.cross(x2);
  Vec4 C = P.transpose() * c;

  Vec3 d = a.cross(b);
  Vec3 q = d * C[3];
  (*X)[0] = q[0];
  (*X)[1] = q[1];
  (*X)[2] = q[2];
  (*X)[3] = -(d[0] * C[0] + d[1] * C[1] + d[2] * C[2]);
}

void TwoViewTriangulationByPlanes(const Vec2 &x1, const Vec2 &x2,
                                  const Mat34 &P,const Mat3 &E,
                                  Vec3 *X) {
  Vec3 x1_homogenious = EuclideanToHomogeneous(x1);
  Vec3 x2_homogenious = EuclideanToHomogeneous(x2);
  Vec4 X_homogenious;
  TwoViewTriangulationByPlanes(x1_homogenious,
                               x2_homogenious,
                               P, E, &X_homogenious);
  (*X) = HomogeneousToEuclidean(X_homogenious);
}

void TwoViewTriangulationIdeal(const Vec3 &x1, const Vec3 &x2,
                               const Mat34 &P, const Mat3 &E,
                               Vec4 *X){
  Vec3 c = E * x1;
  c(2) = 0;
  c = c.cross(x2);
  Vec4 C = P.transpose() * c;

  Vec3 q = x1 * C[3];
  (*X)[0] = q[0];
  (*X)[1] = q[1];
  (*X)[2] = q[2];
  (*X)[3] = -(x1[0] * C[0] + x1[1] * C[1] + x1[2] * C[2]);
}

void TwoViewTriangulationIdeal(const Vec2 &x1, const Vec2 &x2,
                               const Mat34 &P, const Mat3 &E,
                               Vec3 *X) {
  Vec3 x1_homogenious = EuclideanToHomogeneous(x1);
  Vec3 x2_homogenious = EuclideanToHomogeneous(x2);
  Vec4 X_homogenious;
  TwoViewTriangulationIdeal(x1_homogenious,
                            x2_homogenious,
                            P, E, &X_homogenious);
  (*X) = HomogeneousToEuclidean(X_homogenious);
}

}  // namespace libmv
