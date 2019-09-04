// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_KTOU_H__
#define __OPENCV_KTOU_H__

namespace cv{
  namespace alphamat{

using namespace Eigen;
using namespace nanoflann;

typedef std::vector<std::vector<double>> my_vector_of_vectors_t;
typedef std::vector<std::set<int, std::greater<int>>> my_vector_of_set_t;

void generateFVectorKtoU(my_vector_of_vectors_t& fv_unk, my_vector_of_vectors_t& fv_fg, my_vector_of_vectors_t& fv_bg,
             Mat &img, Mat &tmap);

void kdtree_KtoU(Mat &img, Mat &tmap, my_vector_of_vectors_t& indm, my_vector_of_vectors_t& fv_unk,
         my_vector_of_vectors_t& fv_fg, my_vector_of_vectors_t& fv_bg);

SparseMatrix<double> lle_KtoU(my_vector_of_vectors_t& indm, my_vector_of_vectors_t& fv_unk,
  my_vector_of_vectors_t& fv_fg, my_vector_of_vectors_t& fv_bg, double eps, Mat& wf);

SparseMatrix<double> KtoU(Mat& image, Mat& tmap, Mat& wf);

}
}

#endif
