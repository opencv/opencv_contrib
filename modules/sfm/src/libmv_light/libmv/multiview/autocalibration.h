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
//

#ifndef LIBMV_MULTIVIEW_AUTOCALIBRATION_H_
#define LIBMV_MULTIVIEW_AUTOCALIBRATION_H_

#include <algorithm>

#include <Eigen/QR>

#include "libmv/base/vector.h"
#include "libmv/logging/logging.h"
#include "libmv/numeric/numeric.h"

namespace libmv {

void K_From_AbsoluteConic(const Mat3 &W, Mat3 *K);

/** \brief Compute a metric reconstruction from a projective one by computing
 *         the dual absolute quadric using linear constraints.
 *
 * We follow the linear approach proposed by Pollefeys in section 3.4 of [1]
 *
 * [1] M. Pollefeys, L. Van Gool, M. Vergauwen, F. Verbiest, K. Cornelis,
 *     J. Tops, R. Koch, "Visual modeling with a hand-held camera",
 *     International Journal of Computer Vision 59(3), 207-232, 2004.
 */
class AutoCalibrationLinear {
 public:
  /** \brief Add a projection to be used for autocalibration.
   *
   *  \param P The projection matrix.
   *  \param width  The width of the image plane.
   *  \param height The height of the image plane.
   *
   *  The width and height parameters are used to normalize the projection
   *  matrix for improving numerical stability.  The don't need to be exact.
   */
  int AddProjection(const Mat34 &P, double width, double height);

  /** \brief Computes the metric updating transformation.
   *
   *  \return The homography, H, that transforms the space into a metric space.
   *          If {P, X} is a projective reconstruction, then {P H, H^{-1} X} is
   *          a metric reconstruction.  Note that this follows the notation of
   *          HZ section 19.1 page 459, and not the notation of Pollefeys'
   *          paper [1].
   */
  Mat4 MetricTransformation();

 private:
  /** \brief Add constraints on the absolute quadric based assumptions on the
   *         parameters of one camera.
   *
   *  \param P The projection matrix of the camera in projective coordinates.
   */
  void AddProjectionConstraints(const Mat34 &P);

  /** \brief Computes the constraint associated to elements of the DIAC.
   *
   *  \param P The projection used to project the absolute quadric.
   *  \param i Row of the DIAC.
   *  \param j Column of the DIAC.
   *  \return The coeficients of the element i, j of the dual image of the
   *          absolute conic when written as a linear combination of the
   *          elements of the absolute quadric.  There are 10 coeficients since
   *          the absolute quadric is represented by 10 numbers.
   */
  static Vec wc(const Mat34 &P, int i, int j);

  static Mat4 AbsoluteQuadricMatFromVec(const Vec &q);

  static void NormalizeProjection(const Mat34 &P,
                                  double width,
                                  double height,
                                  Mat34 *P_new);

  static void DenormalizeProjection(const Mat34 &P,
                                    double width,
                                    double height,
                                    Mat34 *P_new);

 private:
  vector<Mat34> projections_; // The *normalized* projection matrices.
  vector<double> widths_;
  vector<double> heights_;
  vector<Vec> constraints_;  // Linear constraints on q.
};

} // namespace libmv

#endif  // LIBMV_MULTIVIEW_AUTOCALIBRATION_H_
