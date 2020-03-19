// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_INTRAU_H__
#define __OPENCV_INTRAU_H__

namespace cv{
  namespace alphamat{

using namespace Eigen;
using namespace nanoflann;

typedef std::vector<std::vector<double>> my_vector_of_vectors_t;

double l1norm(std::vector<double>& x, std::vector<double>& y);

int findColMajorInd(int rowMajorInd, int nRows, int nCols);

void UU(Mat& image, Mat& tmap, SparseMatrix<double>& Wuu, SparseMatrix<double>& Duu);

  }
}

#endif
