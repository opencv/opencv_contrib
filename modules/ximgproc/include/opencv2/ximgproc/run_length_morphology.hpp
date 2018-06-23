/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

#ifndef __OPENCV_RUN_LENGTH_MORPHOLOGY_HPP__
#define __OPENCV_RUN_LENGTH_MORPHOLOGY_HPP__
#ifdef __cplusplus

#include <opencv2/core.hpp>

namespace cv {
namespace ximgproc {
namespace rl {


//! @addtogroup ximgproc_run_length_morphology
//! @{

/**
* @brief   Applies a fixed-level threshold to each array element.
*
*
* @param   src         input array (single-channel).
* @param   rlDest      resulting run length encoded image.
* @param   thresh      threshold value.
* @param   type        thresholding type (only THRESH_BINARY and THRESH_BINARY_INV are supported)
*
*/
CV_EXPORTS void threshold(InputArray src, OutputArray rlDest, double thresh, int type);


/**
* @brief   Dilates an run-length encoded binary image by using a specific structuring element.
*
*
* @param   rlSrc       input image
* @param   rlDest      result
* @param   rlKernel    kernel
* @param   anchor      position of the anchor within the element; default value (0, 0)
*                      is usually the element center.
*
*/
CV_EXPORTS void dilate(InputArray rlSrc, OutputArray rlDest, InputArray rlKernel, Point anchor = Point(0, 0));

/**
* @brief   Erodes an run-length encoded binary image by using a specific structuring element.
*
*
* @param   rlSrc       input image
* @param   rlDest      result
* @param   rlKernel    kernel
* @param   bBoundaryOn indicates whether pixel outside the image boundary are assumed to be on
            (True: works in the same way as the default of cv::erode, False: is a little faster)
* @param   anchor      position of the anchor within the element; default value (0, 0)
*                      is usually the element center.
*
*/
CV_EXPORTS void erode(InputArray rlSrc, OutputArray rlDest, InputArray rlKernel,
    bool bBoundaryOn = true, Point anchor = Point(0, 0));

/**
* @brief   Returns a run length encoded structuring element of the specified size and shape.
*
*
* @param   shape	Element shape that can be one of cv::MorphShapes
* @param   ksize	Size of the structuring element.
*
*/
CV_EXPORTS cv::Mat getStructuringElement(int shape, Size ksize);

/**
* @brief   Paint run length encoded binary image into an image.
*
*
* @param   image       image to paint into (currently only single channel images).
* @param   rlSrc       run length encoded image
* @param   value      all foreground pixel of the binary image are set to this value
*
*/
CV_EXPORTS void paint(InputOutputArray image, InputArray rlSrc, const cv::Scalar& value);

/**
* @brief   Check whether a custom made structuring element can be used with run length morphological operations.
*          (It must consist of a continuous array of single runs per row)
*
* @param   rlStructuringElement   mask to be tested
*/
CV_EXPORTS bool isRLMorphologyPossible(InputArray rlStructuringElement);

/**
* @brief   Creates a run-length encoded image from a vector of runs (column begin, column end, row)
*
* @param   runs   vector of runs
* @param   res    result
* @param   size   image size (to be used if an "on" boundary should be used in erosion, using the default
*                  means that the size is computed from the extension of the input)
*/
CV_EXPORTS void createRLEImage(std::vector<cv::Point3i>& runs, OutputArray res, Size size = Size(0, 0));

/**
* @brief   Applies a morphological operation to an run-length encoded binary image.
*
*
* @param   rlSrc       input image
* @param   rlDest      result
* @param   op          all operations supported by cv::morphologyEx (except cv::MORPH_HITMISS)
* @param   rlKernel    kernel
* @param   bBoundaryOnForErosion indicates whether pixel outside the image boundary are assumed
*          to be on for erosion operations (True: works in the same way as the default of cv::erode,
*          False: is a little faster)
* @param   anchor      position of the anchor within the element; default value (0, 0) is usually the element center.
*
*/
CV_EXPORTS void morphologyEx(InputArray rlSrc, OutputArray rlDest, int op, InputArray rlKernel,
    bool bBoundaryOnForErosion = true, Point anchor = Point(0,0));

}
}
}
#endif
#endif
