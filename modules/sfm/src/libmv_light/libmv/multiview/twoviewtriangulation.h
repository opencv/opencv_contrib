// Copyright (c) 2010 libmv authors.
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
//
// Compute a 3D position of a point from several images of it. In particular,
// compute the projective point X in R^4 such that x = PX.
//
// Algorithm is the standard DLT; for derivation see appendix of Keir's thesis.

#ifndef LIBMV_TWOVIEW_NVIEWTRIANGULATION_H
#define LIBMV_TWOVIEW_NVIEWTRIANGULATION_H

#include "libmv/numeric/numeric.h"

namespace libmv {

/**
* Two view triangulation for cameras in canonical form,
* where the reference camera is in the form [I|0] and P is in
* the form [R|t]. The algorithm minimizes the re-projection error
* in the first image only, i.e. the error in the second image is 0
* while the point in the first image is the point lying on the
* epipolar line that is closest to x1.
*
* \param x1 The normalized image point in the first camera
*          (inv(K1)*x1_image)
* \param x2 The normalized image point in the second camera
*           (inv(K2)*x2_image)
* \param P  The second camera matrix in the form [R|t]
* \param E  The essential matrix between the two cameras
* \param X  The 3D homogeneous point
*
* This is the algorithm described in Appendix A in:
* "An efficient solution to the five-point relative pose problem",
* by D. Nist\'er, IEEE PAMI, vol. 26
*/
void TwoViewTriangulationByPlanes(const Vec3 &x1, const Vec3 &x2,
                                  const Mat34 &P,const Mat3 &E, Vec4 *X);
void TwoViewTriangulationByPlanes(const Vec2 &x1, const Vec2 &x2,
                                  const Mat34 &P,const Mat3 &E, Vec3 *X);

/**
* The same algorithm as above generalized for ideal points,
* e.i. where x1*E*x2' = 0. This will not work if the points are
* not ideal. In the case of measured image points it is best to
* either use the TwoViewTriangulationByPlanes function or correct
* the points so that they lay on the corresponding epipolar lines.
*
* \param x1 The normalized image point in the first camera
*          (inv(K1)*x1_image)
* \param x2 The normalized image point in the second camera
*           (inv(K2)*x2_image)
* \param P  The second camera matrix in the form [R|t]
* \param E  The essential matrix between the two cameras
* \param X  The 3D homogeneous point
*/
void TwoViewTriangulationIdeal(const Vec3 &x1, const Vec3 &x2,
                                const Mat34 &P, const Mat3 &E,
                                Vec4 *X);
void TwoViewTriangulationIdeal(const Vec2 &x1, const Vec2 &x2,
                                const Mat34 &P, const Mat3 &E,
                                Vec3 *X);

}  // namespace libmv

#endif  // LIBMV_MULTIVIEW_RESECTION_H
