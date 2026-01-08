// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_RADON_TRANSFORM_HPP__
#define __OPENCV_RADON_TRANSFORM_HPP__

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace cv { namespace ximgproc {
/**
* @brief   Calculate Radon Transform of an image.
* @param   src         The source (input) image.
* @param   dst         The destination image, result of transformation.
* @param   theta       Angle resolution of the transform in degrees.
* @param   start_angle Start angle of the transform in degrees.
* @param   end_angle   End angle of the transform in degrees.
* @param   crop        Crop the source image into a circle.
* @param   norm        Normalize the output Mat to grayscale and convert type to CV_8U
*
* This function calculates the Radon Transform of a given image in any range.
* See https://engineering.purdue.edu/~malcolm/pct/CTI_Ch03.pdf for detail.
* If the input type is CV_8U, the output will be CV_32S.
* If the input type is CV_32F or CV_64F, the output will be CV_64F
* The output size will be num_of_integral x src_diagonal_length.
* If crop is selected, the input image will be crop into square then circle,
* and output size will be num_of_integral x min_edge.
*
*/
CV_EXPORTS_W void RadonTransform(InputArray src,
                                      OutputArray dst,
                                      double theta = 1,
                                      double start_angle = 0,
                                      double end_angle = 180,
                                      bool crop = false,
                                      bool norm = false);
} }

#endif
