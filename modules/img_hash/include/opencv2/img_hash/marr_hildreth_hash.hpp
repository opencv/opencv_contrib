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
// Copyright (C) 2015, University of Ostrava, Institute for Research and Applications of Fuzzy Modeling,
// Pavel Vlasanek, all rights reserved. Third party copyrights are property of their respective owners.
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

#ifndef __OPENCV_MARR_HILDRETH_HASH_HPP__
#define __OPENCV_MARR_HILDRETH_HASH_HPP__

#include "opencv2/core.hpp"
#include "img_hash_base.hpp"

namespace cv
{

namespace img_hash
{
//! @addtogroup marr_hash
//! @{

/** @brief Computes average hash value of the input image
    @param inputArr Input CV_8UC3, CV_8UC1 array.
    @param outputArr Hash value of input, it will contain 16 hex
    decimal number, return type is CV_8U
    @param alpha int scale factor for marr wavelet (default=2).
    @param scale int level of scale factor (default = 1)
*/
CV_EXPORTS_W void marrHildrethHash(cv::InputArray inputArr,
                                   cv::OutputArray outputArr,
                                   float alpha = 2.0f, float scale = 1.0f);

class CV_EXPORTS_W MarrHildrethHash : public ImgHashBase
{
public:

    /** @brief Constructor
        @param alpha int scale factor for marr wavelet (default=2).
        @param scale int level of scale factor (default = 1)
    */
    CV_WRAP explicit MarrHildrethHash(float alpha = 2.0f, float scale = 1.0f);

    CV_WRAP ~MarrHildrethHash();

    /** @brief Computes marr hildreth operator based hash of the input image
        @param inputArr Input CV_8UC3, CV_8UC1 array.
        @param outputArr hash of the image, store 72 uchar hash value
    */
    CV_WRAP virtual void compute(cv::InputArray inputArr,
                                 cv::OutputArray outputArr);

    /** @brief Compare the hash value between inOne and inTwo
    @param hashOne Hash value one
    @param hashTwo Hash value two
    @return value indicate similarity of two hash, smaller mean the
    hash values are more similar to each other
    */
    CV_WRAP virtual double compare(cv::InputArray hashOne,
                                   cv::InputArray hashTwo) const;

    /**
     * @brief self explain
     */
    CV_WRAP float getAlpha() const;

    /**
     * @brief self explain
     */
    CV_WRAP float getScale() const;

    /** @brief Set Mh kernel parameters
        @param alpha int scale factor for marr wavelet (default=2).
        @param scale int level of scale factor (default = 1)
    */
    CV_WRAP void setKernelParam(float alpha, float scale);

    /**
        @param alpha int scale factor for marr wavelet (default=2).
        @param scale int level of scale factor (default = 1)
    */
    CV_WRAP static Ptr<MarrHildrethHash> create(float alpha = 2.0f, float scale = 1.0f);

private:
    float alphaVal;
    cv::Mat blocks;
    cv::Mat blurImg;
    cv::Mat equalizeImg;
    cv::Mat freImg; //frequency response image
    cv::Mat grayImg;
    cv::Mat mhKernel;
    cv::Mat resizeImg;
    float scaleVal;
};

//! @}
}
}

#endif // __OPENCV_MARR_HILDRETH_HASH_HPP__
