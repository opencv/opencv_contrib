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

#ifndef __OPENCV_AVERAGE_HASH_HPP__
#define __OPENCV_AVERAGE_HASH_HPP__

#include "opencv2/core.hpp"
#include "img_hash_base.hpp"

namespace cv
{

namespace ihash
{
//! @addtogroup ihash
//! @{

/** @brief Computes average hash value of the input image
    @param input Input CV_8UC3, CV_8UC1 array.
    @param hash Hash value of input, it will contain 16 hex
    decimal number, return type is CV_8U
     */
CV_EXPORTS void averageHash(cv::Mat const &input, cv::Mat &hash);

class AverageHash : public ImgHashBase
{
public:
  /** @brief Computes average hash of the input image
      @param input input image want to compute hash value
      @param hash hash of the image
  */
  virtual void compute(cv::Mat const &input, cv::Mat &hash);

  CV_EXPORTS static Ptr<AverageHash> create();

private:
  cv::Mat bitsImg;
  cv::Mat grayImg;
  cv::Mat resizeImg;
};

//! @}
}
}

#endif // __OPENCV_FUZZY_F0_MATH_H__
