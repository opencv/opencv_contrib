// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_TRACKER_CSRT_UTILS
#define OPENCV_TRACKER_CSRT_UTILS

#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/hal/interface.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

namespace cv
{

inline int modul(int a, int b)
{
    // function calculates the module of two numbers and it takes into account also negative numbers
    return ((a % b) + b) % b;
}

inline double kernel_epan(double x)
{
    return (x <= 1) ? (2.0/3.14)*(1-x) : 0;
}

Mat circshift(Mat matrix, int dx, int dy);
Mat gaussian_shaped_labels(const float sigma, const int w, const int h);
std::vector<Mat> fourier_transform_features(const std::vector<Mat> &M, Ptr<hal::DFT2D> fwd_plan = Ptr<hal::DFT2D>());
Mat divide_complex_matrices(const Mat &A, const Mat &B);

// Computes res = A / (B + Scalar(b_add, 0)) for CV_32FC2 matrices with zero scratch allocations.
// Matches stock CSRT's `divide_complex_matrices(A, B + b_add)` where OpenCV's `Mat + float`
// constructs `Scalar(b_add)` and adds `b_add` only to the real channel `[0]`.
inline void divide_complex_add_scalar(const Mat &A, const Mat &B, float b_add, Mat &res)
{
    CV_DbgAssert(A.type() == CV_32FC2 && B.type() == CV_32FC2 && A.size() == B.size());
    res.create(A.size(), CV_32FC2);
    for (int y = 0; y < A.rows; ++y)
    {
        const Vec2f* arow = A.ptr<Vec2f>(y);
        const Vec2f* brow = B.ptr<Vec2f>(y);
        Vec2f* rrow = res.ptr<Vec2f>(y);
        for (int x = 0; x < A.cols; ++x)
        {
            const float a = arow[x][0], b = arow[x][1];
            const float c = brow[x][0] + b_add, d = brow[x][1];
            const float den = c * c + d * d;
            rrow[x] = Vec2f((a * c + b * d) / den, (b * c - a * d) / den);
        }
    }
}

void computeHOG32D(const Mat &imageM, Mat &featM, const int sbin, const int pad_x, const int pad_y,
                   Mat &histM, Mat &normM);
Mat get_subwindow(const Mat &image, const Point2f center,
        const int w, const int h,Rect *valid_pixels = NULL);

float subpixel_peak(const Mat &response, const std::string &s, const Point2f &p);
double get_max(const Mat &m);
double get_min(const Mat &m);

Mat get_hann_win(Size sz);
Mat get_kaiser_win(Size sz, float alpha);
Mat get_chebyshev_win(Size sz, float attenuation);

std::vector<Mat> get_features_rgb(const Mat &patch, const Size &output_size);
std::vector<Mat> get_features_hog(const Mat &im, const int bin_size);
std::vector<Mat> get_features_cn(const Mat &im, const Size &output_size);

Mat bgr2hsv(const Mat &img);

} //cv namespace

#endif
