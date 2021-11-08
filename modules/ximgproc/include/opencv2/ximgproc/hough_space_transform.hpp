// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_HOUGH_SPACE_TRANSFORM_HPP__
#define __OPENCV_HOUGH_SPACE_TRANSFORM_HPP__

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace cv { namespace ximgproc {
/**
* @brief   Calculate Hough Space of an image.
* @param   src         The source (input) image.
* @param   dst         The destination image, result of transformation.
* @param   theta       Angle resolution of the transform in degrees. 
* @param   start_angle Start angle of the transform in degrees.
* @param   end_angle   End angle of the transform in degrees.
* @param   crop        Crop the source image into a circle. 
* @param   norm        Normalize the output Mat to grayscale and convert type to CV_8U
*
* This function calculates the Hough Space of a given image in any range.
* Hough Transform is a discrete implementation of Radon Transform.
* Input image is required to be type CV_8U. A square image is recommended. 
* If crop is selected, the input image must be square.
* The output is a Mat of size num_of_integral x src.cols
* i.e. [(end_angle - start_angle) / theta] x src.cols
* with type CV_32SC1 by default.
* 
*/
CV_EXPORTS_W void HoughSpaceTransform(InputArray src, 
                                      OutputArray dst,
                                      double theta = 1,
                                      double start_angle = 0,
                                      double end_angle = 180,
                                      bool crop = false,
                                      bool norm = false);
} }

#endif
