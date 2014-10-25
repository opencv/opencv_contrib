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

/*
* simple_color_balance.hpp
*
*  Created on: Jun 26, 2014
*      Author: Yury Gitman
*/

#include <opencv2/core.hpp>

/*! \namespace cv
Namespace where all the C++ OpenCV functionality resides
*/
namespace cv
{
namespace xphoto
{
    //! various white balance algorithms
    enum
    {
        WHITE_BALANCE_SIMPLE = 0,
        WHITE_BALANCE_GRAYWORLD = 1
    };

    /*! This function implements different white balance algorithms
    *  \param src : source image
    *  \param dst : destination image
    *  \param algorithmType : type of the algorithm to use
    *  \param inputMin : minimum input value
    *  \param inputMax : maximum output value
    *  \param outputMin : minimum input value
    *  \param outputMax : maximum output value
    */
    CV_EXPORTS_W void balanceWhite(const Mat &src, Mat &dst, const int algorithmType,
        const float inputMin  = 0.0f, const float inputMax  = 255.0f,
        const float outputMin = 0.0f, const float outputMax = 255.0f);
}
}

#endif // __OPENCV_SIMPLE_COLOR_BALANCE_HPP__