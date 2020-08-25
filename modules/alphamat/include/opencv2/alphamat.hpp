// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/** Information Flow algorithm implementaton for alphamatting */

#ifndef _OPENCV_ALPHAMAT_HPP_
#define _OPENCV_ALPHAMAT_HPP_

/**
 * @defgroup alphamat Alpha Matting
 * This module is dedicated to compute alpha matting of images, given the input image and an input trimap.
 * The samples directory includes easy examples of how to use the module.
 */

namespace cv { namespace alphamat {
//! @addtogroup alphamat
//! @{

/**
 * The implementation is based on Designing Effective Inter-Pixel Information Flow for Natural Image Matting by Yağız Aksoy, Tunç Ozan Aydın and Marc Pollefeys, CVPR 2019.
 *
 * This module has been originally developed by Muskaan Kularia and Sunita Nayak as a project
 * for Google Summer of Code 2019 (GSoC 19).
 *
 */
CV_EXPORTS_W void infoFlow(InputArray image, InputArray tmap, OutputArray result);

//! @}
}}  // namespace

#endif
