// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef _GEO_INTERPOLATION_HPP_
#define _GEO_INTERPOLATION_HPP_

namespace cv {
namespace optflow {

typedef Vec<float, 8> Vec8f;
Mat getGraph(const Mat & image, float edge_length);
Mat sgeo_dist(const Mat& gra, int y, int x, float max, Mat &prev);
Mat sgeo_dist(const Mat& gra, const std::vector<Point2f> & points, float max, Mat &prev);
Mat interpolate_irregular_nw(const Mat &in, const Mat &mask, const Mat &color_img, float max_d, float bandwidth, float pixeldistance);
Mat interpolate_irregular_nn(
    const std::vector<Point2f> & prevPoints,
    const std::vector<Point2f> & nextPoints,
    const std::vector<uchar> & status,
    const Mat &color_img,
    float pixeldistance);
Mat interpolate_irregular_knn(
    const std::vector<Point2f> & _prevPoints,
    const std::vector<Point2f> & _nextPoints,
    const std::vector<uchar> & status,
    const Mat &color_img,
    int k,
    float pixeldistance);

Mat interpolate_irregular_nn_raster(const std::vector<Point2f> & prevPoints,
    const std::vector<Point2f> & nextPoints,
    const std::vector<uchar> & status,
    const Mat & i1);

}} // namespace
#endif
