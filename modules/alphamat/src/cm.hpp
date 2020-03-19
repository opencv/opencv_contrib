// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_ALPHAMAT_CM_H__
#define __OPENCV_ALPHAMAT_CM_H__

namespace cv { namespace alphamat {

using namespace Eigen;
using namespace nanoflann;

void cm(Mat& image, Mat& tmap, SparseMatrix<double>& Wcm, SparseMatrix<double>& Dcm);

}}

#endif
