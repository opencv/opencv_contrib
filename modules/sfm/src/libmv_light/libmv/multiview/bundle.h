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


#ifndef LIBMV_MULTIVIEW_BUNDLE_H_
#define LIBMV_MULTIVIEW_BUNDLE_H_

#include "libmv/base/vector.h"
#include "libmv/numeric/numeric.h"

namespace libmv {

enum eLibmvBundleType
{
  eBUNDLE_METRIC = 0,
  eBUNDLE_FOCAL_LENGTH = 1,      // f
  eBUNDLE_FOCAL_LENGTH_PP = 2,   // f, cx, cy
  eBUNDLE_RADIAL = 3,            // f, cx, cy, k1, k2
  eBUNDLE_RADIAL_TANGENTIAL = 4  // f, cx, cy, k1, k2, p1, p2
};



/**
 * \brief Euclidean bundle adjustment given a full observation.
 *
 * \param[in]     x  The observed projections. x[i].col(j) is the projection of
 *                   point j in camera i.
 * \param[in/out] Ks The calibration matrices.  For now it assume a single
 *                   calibration matrix, so all Ks[i] should be the same.
 * \param[in/out] Rs The rotation matrices.
 * \param[in/out] ts The translation vectors.
 * \param[in/out] X  The point structure.
 * \param[in]     type  The type of bundle (instrinsic parameters)
 *
 * All the points are assumed to be observed in all images.
 * We use the convention x = Ks * (Rs * X + ts).
 */
void EuclideanBAFull(const vector<Mat2X> &x,
                     vector<Mat3> *Ks,
                     vector<Mat3> *Rs,
                     vector<Vec3> *ts,
                     Mat3X *X,
                     eLibmvBundleType type = eBUNDLE_FOCAL_LENGTH);

/**
 * \brief Euclidean bundle adjustment given a full observation.
 *
 * \param[in]     x  The observed projections. Each element of the vector
 *                   contains a pair of a projections matrix and a vector of
 *                   corresponding point structure ids
 *                   x[i].col(j) is the projection
 *                   of point x_ids[i][j] in camera i.
 * \param[in]     x_ids  Indexes of points structures corresponding to the
 *                       projections x
 * \param[in/out] Ks The calibration matrices.  For now it assume a single
 *                   calibration matrix, so all Ks[i] should be the same.
 * \param[in/out] Rs The rotation matrices.
 * \param[in/out] ts The translation vectors.
 * \param[in/out] X  The point structure.
 * \param[in]     type  The type of bundle (instrinsic parameters)
 * \return        rms The root mean square error
 *
 * All the points are assumed to be observed in all images.
 * We use the convention x = Ks * (Rs * X + ts).
 */
double EuclideanBA(const vector<Mat2X> &x,
                   const vector<Vecu> &x_ids,
                   vector<Mat3> *Ks,
                   vector<Mat3> *Rs,
                   vector<Vec3> *ts,
                   Mat3X *X,
                   eLibmvBundleType type = eBUNDLE_FOCAL_LENGTH);

} // namespace libmv

#endif  // LIBMV_MULTIVIEW_BUNDLE_H_
