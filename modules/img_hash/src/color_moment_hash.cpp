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

#include "precomp.hpp"

#include <iostream>

namespace cv{

namespace img_hash{

ColorMomentHash::~ColorMomentHash()
{

}

void ColorMomentHash::compute(cv::InputArray inputArr,
                              cv::OutputArray outputArr)
{
  cv::Mat const input = inputArr.getMat();
  CV_Assert(input.type() == CV_8UC3);

  cv::resize(input, resizeImg_, cv::Size(512,512), 0, 0,
             INTER_CUBIC);
  cv::GaussianBlur(resizeImg_, blurImg_, cv::Size(3,3), 0, 0);

  cv::cvtColor(blurImg_, colorSpace_, CV_BGR2HSV);
  cv::split(colorSpace_, channels_);
  outputArr.create(1, 42, CV_64F);
  cv::Mat hash = outputArr.getMat();
  hash.setTo(0);
  computeMoments(hash.ptr<double>(0));

  cv::cvtColor(blurImg_, colorSpace_, CV_BGR2YCrCb);
  cv::split(colorSpace_, channels_);
  computeMoments(hash.ptr<double>(0) + 21);
}

double ColorMomentHash::compare(cv::InputArray hashOne,
                                cv::InputArray hashTwo) const
{
  return norm(hashOne, hashTwo, NORM_L2) * 10000;
}

Ptr<ColorMomentHash> ColorMomentHash::create()
{
  return makePtr<ColorMomentHash>();
}

void ColorMomentHash::computeMoments(double *inout)
{
  for(size_t i = 0; i != channels_.size(); ++i)
  {
    cv::HuMoments(cv::moments(channels_[i]), inout);
    inout += 7;
  }
}

void colorMomentHash(cv::InputArray inputArr,
                     cv::OutputArray outputArr)
{
  ColorMomentHash().compute(inputArr, outputArr);
}

}

}
