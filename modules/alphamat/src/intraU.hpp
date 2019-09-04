// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_INTRAU_H__
#define __OPENCV_INTRAU_H__

namespace cv{
  namespace alphamat{

using namespace Eigen;
using namespace nanoflann;

void generateFVectorIntraU(my_vector_of_vectors_t &samples, Mat &img, Mat& tmap);

void kdtree_intraU(Mat &img, Mat& tmap, my_vector_of_vectors_t& indm, my_vector_of_set_t& inds, my_vector_of_vectors_t& samples);

double l1norm(std::vector<double>& x, std::vector<double>& y);

void intraU(my_vector_of_vectors_t& indm, my_vector_of_set_t& inds, my_vector_of_vectors_t& samples, SparseMatrix<double>& Wuu, SparseMatrix<double>& Duu);

void UU(Mat& image, Mat& tmap, SparseMatrix<double>& Wuu, SparseMatrix<double>& Duu);

}
}

#endif