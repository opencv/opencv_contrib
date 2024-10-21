// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_RADON_TRANSFORM_HPP__
#define __OPENCV_RADON_TRANSFORM_HPP__

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace cv { namespace ximgproc {
/**
 * @brief Computes the Radon transform of a given 2D image.
 *
 * The Radon transform is often used in image processing, particularly in applications
 * like computed tomography, to analyze the structure of an image by projecting it along
 * different angles. This function calculates the Radon Transform over a specified range
 * of angles.
 *
 * The output type will vary depending on the input type:
 * - If the input type is CV_8U, the output will be CV_32S.
 * - If the input type is CV_32F or CV_64F, the output will be CV_64F.
 *
 * The size of the output matrix depends on whether cropping is applied:
 * - Without cropping, the output size will be `num_of_integral x src_diagonal_length`.
 * - With cropping (circular), the output size will be `num_of_integral x min_edge`,
 *   where `min_edge` is the smaller dimension of the cropped square.
 *
 * See https://engineering.purdue.edu/~malcolm/pct/CTI_Ch03.pdf for more details.
 *
 * @param src The input image on which the Radon transform is to be applied.
 *            Must be a 2D single-channel array (e.g., grayscale image).
 * @param dst The output array that will hold the result of the Radon transform.
 *            The type of the output will depend on the input image type.
 * @param theta The angle increment in degrees for the projection (resolution of the transform).
 *              Default is 1 degree.
 * @param start_angle The starting angle for the Radon transform in degrees.
 *                    Default is 0 degrees.
 * @param end_angle The ending angle for the Radon transform in degrees.
 *                  Default is 180 degrees. The difference between end_angle and start_angle must
 *                  be positive when multiplied by theta.
 * @param crop A flag indicating whether to crop the input image to a square or circular shape
 *             before the transformation. If enabled, the image is first cropped to a square
 *             (smallest dimension) and then transformed into a circle.
 * @param norm A flag indicating whether to normalize the output image to the range [0, 255] after
 *             computation and convert the type to `CV_8U`. If normalization is not enabled,
 *             the output will retain its original data range.
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
