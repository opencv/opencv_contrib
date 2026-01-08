// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/** Information Flow algorithm implementaton for alphamatting */

#ifndef _OPENCV_ALPHAMAT_HPP_
#define _OPENCV_ALPHAMAT_HPP_

#include <opencv2/core.hpp>

/**
 * @defgroup alphamat Alpha Matting
 * Alpha matting is used to extract a foreground object with soft boundaries from a background image.
 *
 * This module is dedicated to computing alpha matte of objects in images from a given input image and a greyscale trimap image that contains information about the foreground, background and unknown pixels. The unknown pixels are assumed to be a combination of foreground and background pixels. The algorithm uses a combination of multiple carefully defined pixels affinities to estimate the opacity of the foreground pixels in the unkown region.
 *
 * The implementation is based on @cite aksoy2017designing.
 *
 * This module was developed by Muskaan Kularia and Sunita Nayak as a project
 * for Google Summer of Code 2019 (GSoC 19).
 *
 */

namespace cv { namespace alphamat {
//! @addtogroup alphamat
//! @{

/**
 * @brief Compute alpha matte of an object in an image
 * @param image Input RGB image
 * @param tmap Input greyscale trimap image
 * @param result Output alpha matte image
 *
 * The function infoFlow performs alpha matting on a RGB image using a greyscale trimap image, and outputs a greyscale alpha matte image. The output alpha matte can be used to softly extract the foreground object from a background image. Examples can be found in the samples directory.
 *
 */
CV_EXPORTS_W void infoFlow(InputArray image, InputArray tmap, OutputArray result);

//! @}
}}  // namespace

#endif
