// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_INTENSITY_TRANSFORM_H
#define OPENCV_INTENSITY_TRANSFORM_H

#include "opencv2/core.hpp"

/**
 * @defgroup intensity_transform The module brings implementations of intensity transformation algorithms to adjust image contrast.
 *
 * Namespace for all functions is `cv::intensity_transform`.
 *
 * ### Supported Algorithms
 * - Autoscaling
 * - Log Transformations
 * - Power-Law (Gamma) Transformations
 * - Contrast Stretching
 * - BIMEF, A Bio-Inspired Multi-Exposure Fusion Framework for Low-light Image Enhancement @cite ying2017bio @cite ying2017new
 *
 * References from following book and websites:
 * - Digital Image Processing 4th Edition Chapter 3 [Rafael C. Gonzalez, Richard E. Woods] @cite Gonzalez2018
 * - http://www.cs.uregina.ca/Links/class-info/425/Lab3/ @cite lcs435lab
 * - https://theailearner.com/2019/01/30/contrast-stretching/ @cite theailearner
*/

namespace cv {
namespace intensity_transform {

//! @addtogroup intensity_transform
//! @{

/**
 * @brief Given an input bgr or grayscale image and constant c, apply log transformation to the image
 * on domain [0, 255] and return the resulting image.
 *
 * @param input input bgr or grayscale image.
 * @param output resulting image of log transformations.
*/
CV_EXPORTS_W void logTransform(const Mat input, Mat& output);

/**
 * @brief Given an input bgr or grayscale image and constant gamma, apply power-law transformation,
 * a.k.a. gamma correction to the image on domain [0, 255] and return the resulting image.
 *
 * @param input input bgr or grayscale image.
 * @param output resulting image of gamma corrections.
 * @param gamma constant in c*r^gamma where r is pixel value.
*/
CV_EXPORTS_W void gammaCorrection(const Mat input, Mat& output, const float gamma);

/**
 * @brief Given an input bgr or grayscale image, apply autoscaling on domain [0, 255] to increase
 * the contrast of the input image and return the resulting image.
 *
 * @param input input bgr or grayscale image.
 * @param output resulting image of autoscaling.
*/
CV_EXPORTS_W void autoscaling(const Mat input, Mat& output);

/**
 * @brief Given an input bgr or grayscale image, apply linear contrast stretching on domain [0, 255]
 * and return the resulting image.
 *
 * @param input input bgr or grayscale image.
 * @param output resulting image of contrast stretching.
 * @param r1 x coordinate of first point (r1, s1) in the transformation function.
 * @param s1 y coordinate of first point (r1, s1) in the transformation function.
 * @param r2 x coordinate of second point (r2, s2) in the transformation function.
 * @param s2 y coordinate of second point (r2, s2) in the transformation function.
*/
CV_EXPORTS_W void contrastStretching(const Mat input, Mat& output, const int r1, const int s1, const int r2, const int s2);

/**
 * @brief Given an input color image, enhance low-light images using the BIMEF method (@cite ying2017bio @cite ying2017new).
 *
 * @param input input color image.
 * @param output resulting image.
 * @param mu enhancement ratio.
 * @param a a-parameter in the Camera Response Function (CRF).
 * @param b b-parameter in the Camera Response Function (CRF).
 *
 * @warning This is a C++ implementation of the [original MATLAB algorithm](https://github.com/baidut/BIMEF).
 * Compared to the original code, this implementation is a little bit slower and does not provide the same results.
 * In particular, quality of the image enhancement is degraded for the bright areas in certain conditions.
*/
CV_EXPORTS_W void BIMEF(InputArray input, OutputArray output, float mu=0.5f, float a=-0.3293f, float b=1.1258f);

/**
 * @brief Given an input color image, enhance low-light images using the BIMEF method (@cite ying2017bio @cite ying2017new).
 *
 * This is an overloaded function with the exposure ratio given as parameter.
 *
 * @param input input color image.
 * @param output resulting image.
 * @param k exposure ratio.
 * @param mu enhancement ratio.
 * @param a a-parameter in the Camera Response Function (CRF).
 * @param b b-parameter in the Camera Response Function (CRF).
 *
 * @warning This is a C++ implementation of the [original MATLAB algorithm](https://github.com/baidut/BIMEF).
 * Compared to the original code, this implementation is a little bit slower and does not provide the same results.
 * In particular, quality of the image enhancement is degraded for the bright areas in certain conditions.
*/
CV_EXPORTS_AS(BIMEF2) void BIMEF(InputArray input, OutputArray output, float k, float mu, float a, float b);

//! @}

}} // cv::intensity_transform::

#endif
