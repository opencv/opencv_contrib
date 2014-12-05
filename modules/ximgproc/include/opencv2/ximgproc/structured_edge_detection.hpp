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

#ifndef __OPENCV_STRUCTURED_EDGE_DETECTION_HPP__
#define __OPENCV_STRUCTURED_EDGE_DETECTION_HPP__
#ifdef __cplusplus

/** @file
@date Jun 17, 2014
@author Yury Gitman
 */

#include <opencv2/core.hpp>

namespace cv
{
namespace ximgproc
{

//! @addtogroup ximgproc_edge
//! @{

/*!
  Helper class for training part of [P. Dollar and C. L. Zitnick. Structured Forests for Fast Edge Detection, 2013].
 */
class CV_EXPORTS_W RFFeatureGetter : public Algorithm
{
public:

    /*!
     * This functions extracts feature channels from src.
     * Than StructureEdgeDetection uses this feature space
     * to detect edges.
     *
     * \param src : source image to extract features
     * \param features : output n-channel floating point feature matrix.
     *
     * \param gnrmRad : __rf.options.gradientNormalizationRadius
     * \param gsmthRad : __rf.options.gradientSmoothingRadius
     * \param shrink : __rf.options.shrinkNumber
     * \param outNum : __rf.options.numberOfOutputChannels
     * \param gradNum : __rf.options.numberOfGradientOrientations
     */
    CV_WRAP virtual void getFeatures(const Mat &src, Mat &features,
                                     const int gnrmRad,
                                     const int gsmthRad,
                                     const int shrink,
                                     const int outNum,
                                     const int gradNum) const = 0;
};

CV_EXPORTS_W Ptr<RFFeatureGetter> createRFFeatureGetter();



/** @brief Class implementing edge detection algorithm from @cite Dollar2013 :
 */
class CV_EXPORTS_W StructuredEdgeDetection : public Algorithm
{
public:

    /** @brief The function detects edges in src and draw them to dst.

    The algorithm underlies this function is much more robust to texture presence, than common
    approaches, e.g. Sobel
    @param src source image (RGB, float, in [0;1]) to detect edges
    @param dst destination image (grayscale, float, in [0;1]) where edges are drawn
    @sa Sobel, Canny
     */
    CV_WRAP virtual void detectEdges(const Mat &src, CV_OUT Mat &dst) const = 0;
};

/*!
* The only constructor
*
* \param model : name of the file where the model is stored
* \param howToGetFeatures : optional object inheriting from RFFeatureGetter.
*                           You need it only if you would like to train your
*                           own forest, pass NULL otherwise
*/
CV_EXPORTS_W Ptr<StructuredEdgeDetection> createStructuredEdgeDetection(const String &model,
    Ptr<const RFFeatureGetter> howToGetFeatures = Ptr<RFFeatureGetter>());

//! @}

}
}
#endif
#endif /* __OPENCV_STRUCTURED_EDGE_DETECTION_HPP__ */
