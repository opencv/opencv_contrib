/*M///////////////////////////////////////////////////////////////////////////////////////
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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_SIMPLE_COLOR_BALANCE_HPP__
#define __OPENCV_SIMPLE_COLOR_BALANCE_HPP__

/** @file
@date Jun 26, 2014
@author Yury Gitman
*/

#include <opencv2/core.hpp>

namespace cv
{
namespace xphoto
{

//! @addtogroup xphoto
//! @{

    //! various white balance algorithms
    enum WhitebalanceTypes
    {
        /** perform smart histogram adjustments (ignoring 4% pixels with minimal and maximal
        values) for each channel */
        WHITE_BALANCE_SIMPLE = 0,
        WHITE_BALANCE_GRAYWORLD = 1
    };

    /** @brief The function implements different algorithm of automatic white balance,

    i.e. it tries to map image's white color to perceptual white (this can be violated due to
    specific illumination or camera settings).

    @param src
    @param dst
    @param algorithmType see xphoto::WhitebalanceTypes
    @param inputMin minimum value in the input image
    @param inputMax maximum value in the input image
    @param outputMin minimum value in the output image
    @param outputMax maximum value in the output image
    @sa cvtColor, equalizeHist
     */
    CV_EXPORTS_W void balanceWhite(const Mat &src, Mat &dst, const int algorithmType,
        const float inputMin  = 0.0f, const float inputMax  = 255.0f,
        const float outputMin = 0.0f, const float outputMax = 255.0f);

    /** @brief Implements a simple grayworld white balance algorithm.

    The function autowbGrayworld scales the values of pixels based on a
    gray-world assumption which states that the average of all channels
    should result in a gray image.

    This function adds a modification which thresholds pixels based on their
    saturation value and only uses pixels below the provided threshold in
    finding average pixel values.

    Saturation is calculated using the following for a 3-channel RGB image per
    pixel I and is in the range [0, 1]:

    \f[ \texttt{Saturation} [I] = \frac{\textrm{max}(R,G,B) - \textrm{min}(R,G,B)
    }{\textrm{max}(R,G,B)} \f]

    A threshold of 1 means that all pixels are used to white-balance, while a
    threshold of 0 means no pixels are used. Lower thresholds are useful in
    white-balancing saturated images.

    Currently only works on images of type @ref CV_8UC3.

    @param src Input array.
    @param dst Output array of the same size and type as src.
    @param thresh Maximum saturation for a pixel to be included in the
        gray-world assumption.

    @sa balanceWhite
     */
    CV_EXPORTS_W void autowbGrayworld(InputArray src, OutputArray dst,
        float thresh = 0.5f);

//! @}

}
}

#endif // __OPENCV_SIMPLE_COLOR_BALANCE_HPP__
