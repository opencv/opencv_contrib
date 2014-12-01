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
// Copyright (C) 2008, Willow Garage Inc., all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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
#ifndef __OPENCV_PEILIN_HPP__
#define __OPENCV_PEILIN_HPP__

#include <opencv2/core.hpp>

namespace cv
{
    //! @addtogroup ximgproc_filters
    //! @{

    /**
    * @brief   Calculates an affine transformation that normalize given image using Pei&Lin Normalization.
    *
    * Assume given image :math:`I=T(\bar{I})` where :math:`\bar{I}` is a normalized image and :math:`T` is is an affine transformation distorting this image by translation, rotation, scaling and skew.
    * The function returns an affine transformation matrix corresponding to the transformation :math:`T^{-1}` described in [PeiLin95].
    * For more details about this implementation, please see
    * [PeiLin95] Soo-Chang Pei and Chao-Nan Lin. Image normalization for pattern recognition. Image and Vision Computing, Vol. 13, N.10, pp. 711-723, 1995.
    *
    * @param I Given transformed image.
    */
    CV_EXPORTS Matx23d PeiLinNormalization ( InputArray I );
    /** @overload */
    CV_EXPORTS_W void PeiLinNormalization ( InputArray I, OutputArray T );
}

#endif
