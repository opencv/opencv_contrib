// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_CM_H__
#define __OPENCV_CM_H__

namespace cv{
  namespace alphamat{

using namespace Eigen;
using namespace nanoflann;

typedef std::vector<std::vector<double>> my_vector_of_vectors_t;
// typedef vector<set<int, greater<int>>> my_vector_of_set_t;

void generateFVectorCM(my_vector_of_vectors_t &samples, Mat &img);

void kdtree_CM(Mat &img, my_vector_of_vectors_t& indm, my_vector_of_vectors_t& samples, std::unordered_set<int>& unk);

void lle(my_vector_of_vectors_t& indm, my_vector_of_vectors_t& samples, float eps, std::unordered_set<int>& unk
             , SparseMatrix<double>& Wcm, SparseMatrix<double>& Dcm);

void cm(Mat& image, Mat& tmap, SparseMatrix<double>& Wcm, SparseMatrix<double>& Dcm);

}}

#endif