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

    Currently only works on images of type @ref CV_8UC3 and @ref CV_16UC3.

    @param src Input array.
    @param dst Output array of the same size and type as src.
    @param thresh Maximum saturation for a pixel to be included in the
        gray-world assumption.

    @sa balanceWhite
     */
    CV_EXPORTS_W void autowbGrayworld(InputArray src, OutputArray dst,
        float thresh = 0.5f);

    /** @brief Implements a more sophisticated learning-based automatic color balance algorithm.

    As autowbGrayworld, this function works by applying different gains to the input
    image channels, but their computation is a bit more involved compared to the
    simple grayworld assumption. More details about the algorithm can be found in
    @cite Cheng2015 .

    To mask out saturated pixels this function uses only pixels that satisfy the
    following condition:

    \f[ \frac{\textrm{max}(R,G,B)}{\texttt{range_max_val}} < \texttt{saturation_thresh} \f]

    Currently supports images of type @ref CV_8UC3 and @ref CV_16UC3.

    @param src Input three-channel image in the BGR color space.
    @param dst Output image of the same size and type as src.
    @param range_max_val Maximum possible value of the input image (e.g. 255 for 8 bit images, 4095 for 12 bit images)
    @param saturation_thresh Threshold that is used to determine saturated pixels
    @param hist_bin_num Defines the size of one dimension of a three-dimensional RGB histogram that is used internally by
    the algorithm. It often makes sense to increase the number of bins for images with higher bit depth (e.g. 256 bins
    for a 12 bit image)

    @sa autowbGrayworld
    */
    CV_EXPORTS_W void autowbLearningBased(InputArray src, OutputArray dst, int range_max_val = 255,
                                          float saturation_thresh = 0.98f, int hist_bin_num = 64);

    /** @brief Implements the feature extraction part of the learning-based color balance algorithm.

    In accordance with @cite Cheng2015 , computes the following features for the input image:
    1. Chromaticity of an average (R,G,B) tuple
    2. Chromaticity of the brightest (R,G,B) tuple (while ignoring saturated pixels)
    3. Chromaticity of the dominant (R,G,B) tuple (the one that has the highest value in the RGB histogram)
    4. Mode of the chromaticity pallete, that is constructed by taking 300 most common colors according to
       the RGB histogram and projecting them on the chromaticity plane. Mode is the most high-density point
       of the pallete, which is computed by a straightforward fixed-bandwidth kernel density estimator with
       a Epanechnikov kernel function.

    @param src Input three-channel image in the BGR color space.
    @param dst An array of four (r,g) chromaticity tuples corresponding to the features listed above.
    @param range_max_val Maximum possible value of the input image (e.g. 255 for 8 bit images, 4095 for 12 bit images)
    @param saturation_thresh Threshold that is used to determine saturated pixels
    @param hist_bin_num Defines the size of one dimension of a three-dimensional RGB histogram that is used internally by
    the algorithm. It often makes sense to increase the number of bins for images with higher bit depth (e.g. 256 bins
    for a 12 bit image)

    @sa autowbLearningBased
    */
    CV_EXPORTS_W void extractSimpleFeatures(InputArray src, OutputArray dst, int range_max_val = 255,
                                            float saturation_thresh = 0.98f, int hist_bin_num = 64);

    /** @brief Implements an efficient fixed-point approximation for applying channel gains.

    @param src Input three-channel image in the BGR color space (either CV_8UC3 or CV_16UC3)
    @param dst Output image of the same size and type as src.
    @param gainB gain for the B channel
    @param gainG gain for the G channel
    @param gainR gain for the R channel

    @sa autowbGrayworld, autowbLearningBased
    */
    CV_EXPORTS_W void applyChannelGains(InputArray src, OutputArray dst, float gainB, float gainG, float gainR);
    //! @}

}
}

#endif // __OPENCV_SIMPLE_COLOR_BALANCE_HPP__
