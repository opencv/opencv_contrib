/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_FASTCV_FAST10_HPP
#define OPENCV_FASTCV_FAST10_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

//! @addtogroup fastcv
//! @{

/**
 * @brief Extracts FAST10 corners and scores from the image based on the mask.
 *        The mask specifies pixels to be ignored by the detector
 *        designed for corner detection on Qualcomm's processors, provides enhanced speed.
 *
 * @param src 8-bit grayscale image
 * @param mask Optional mask indicating which pixels should be omited from corner dection.
               Its size should be k times image width and height, where k = 1/2, 1/4 , 1/8 , 1, 2, 4 and 8
               For more details see documentation to `fcvCornerFast9InMaskScoreu8` function in FastCV
 * @param coords Output array of CV_32S containing interleave x, y positions of detected corners
 * @param scores Optional output array containing the scores of the detected corners.
                 The score is the highest threshold that can still validate the detected corner.
                 A higher score value indicates a stronger corner feature.
                 For example, a corner of score 108 is stronger than a corner of score 50
 * @param barrier FAST threshold. The threshold is used to compare difference between intensity value
                  of the central pixel and pixels on a circle surrounding this pixel
 * @param border Number for pixels to ignore from top,bottom,right,left of the image. Defaults to 4 if it's below 4
 * @param nmsEnabled Enable non-maximum suppresion to prune weak key points
 */
CV_EXPORTS_W void FAST10(InputArray src, InputArray mask, OutputArray coords, OutputArray scores, int barrier, int border, bool nmsEnabled);

//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_FAST10_HPP
