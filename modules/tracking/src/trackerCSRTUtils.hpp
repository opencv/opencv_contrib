/*///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#ifndef OPENCV_TRACKER_CSRT_UTILS
#define OPENCV_TRACKER_CSRT_UTILS

#include "precomp.hpp"

#include <fstream>
#include <iostream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>

namespace cv
{

Mat readcsv(std::string filename);
void save_matrix(std::string filename, cv::Mat m);
template < class T >
std::ostream& print_dvec( const std::vector<T>& v)
{
	if (!v.empty()) {
        std::cout << '[';
		std::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, ", "));
        std::cout << "\b\b]" << std::endl;
	}
	return std::cout;
}

template < class T >
void save_vector(std::string name, const std::vector<T> &v)
{
    if(v.size() == 0)
        return;
    std::ofstream myfile;
	char buffer[500];
    sprintf(buffer, "/home/amuhic/Desktop/mat/v_%s", name.c_str());
    myfile.open(buffer);
    myfile << v[0];
    for(int i=1; i< v.size(); ++i) {
        myfile << ", " << v[i];
    }
    std::cout << "vector saved: " << name << std::endl;
    myfile.close();
}

#define	 pvar(obj)				std::cout << "; " << #obj << " > " << obj << "\t call@: " << __func__ << std::endl;
#define SAVE_MATRIX(M) \
	do { \
		save_matrix( #M , M); \
	} while(false)

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
std::vector<Mat> fourier_transform_features(const std::vector<Mat> &M);
Mat divide_complex_matrices(const Mat &_A, const Mat &_B);
Mat get_subwindow( const Mat &image, const Point2f center,
        const int w, const int h,Rect *valid_pixels = NULL);

float subpixel_peak(const Mat &response, const std::string &s, const Point2f &p);
double get_max(const Mat &m);
double get_min(const Mat &m);

Mat get_hann_win(Size sz);
Mat get_kaiser_win(Size sz, double alpha);
Mat get_chebyshev_win(Size sz, double attenuation);

std::vector<Mat> get_features_rgb(const Mat &patch, const Size &output_size);
std::vector<Mat> get_features_hog( const Mat &im, const int bin_size);
std::vector<Mat> get_features_cn(const Mat &im, const Size &output_size);

Mat bgr2hsv(const Mat &img);

} //cv namespace

#endif
