// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// #ifndef local_info
// #define local_info

namespace cv{
  namespace alphamat{

using namespace Eigen;

void local_info(Mat& img, Mat& tmap, SparseMatrix<double>& Wl, SparseMatrix<double>& Dl);
}}

// #endif