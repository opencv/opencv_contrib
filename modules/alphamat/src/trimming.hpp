// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_TRIMMING_H__
#define __OPENCV_TRIMMING_H__


namespace cv{
  namespace alphamat{

using namespace nanoflann;

double l2norm(int x1, int y1, int x2, int y2);

void generateMean(Mat &img, Mat &tmap, my_vector_of_pair &map);

void findNearestNbr(my_vector_of_vectors_t& indm);

double Bhattacharya(Mat mean1, Mat mean2, Mat cov1, Mat cov2);

void trimming(Mat &img, Mat &tmap, Mat &new_tmap, bool post);

}}

#endif