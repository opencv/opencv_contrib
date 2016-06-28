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

#ifndef __OPENCV_RADIAL_VARIANCE_HASH_HPP__
#define __OPENCV_RADIAL_VARIANCE_HASH_HPP__

#include "opencv2/core.hpp"
#include "img_hash_base.hpp"

namespace cv
{

namespace img_hash
{
//! @addtogroup radial_var_hash
//! @{

/** @brief Computes radial variance hash of the input image
    @param inputArr Input CV_8UC3, CV_8UC1 array.
    @param outputArr Hash value of input
    @param sigma Gaussian kernel standard deviation
    @param numOfAngleLine The number of angles to consider
     */
CV_EXPORTS_W void radialVarianceHash(cv::InputArray inputArr,
                                     cv::OutputArray outputArr,
                                     double sigma = 1,
                                     int numOfAngleLine = 180);

class CV_EXPORTS_W RadialVarianceHash : public ImgHashBase
{
  //This friend class is design for unit test, please do not
  //use it under normal case
  friend class RadialVarHashTester;
public:
  /** @brief Constructor
      @param sigma Gaussian kernel standard deviation
      @param numOfAngleLine The number of angles to consider
   */
  CV_WRAP explicit RadialVarianceHash(double sigma = 1,
                                      int numOfAngleLine = 180);
  CV_WRAP ~RadialVarianceHash();

  /** @brief Computes average hash of the input image
      @param inputArr input image want to compute hash value
      @param outputArr hash of the image, contain 40 uchar value
  */
  CV_WRAP virtual void compute(cv::InputArray inputArr,
                               cv::OutputArray outputArr);

  /** @brief Compare the hash value between inOne and inTwo
  @param hashOne Hash value one
  @param hashTwo Hash value two
  @return cross correlation of two hash, the closer the value to 1,
  the more similar the hash values. We could assume the threshold is 0.9
  by default
  */
  CV_WRAP virtual double compare(cv::InputArray hashOne,
                                 cv::InputArray hashTwo) const;

  CV_WRAP static Ptr<RadialVarianceHash> create(double sigma = 1,
                                                int numOfAngleLine = 180);

  CV_WRAP int getNumOfAngleLine() const;
  CV_WRAP double getSigma() const;

  CV_WRAP void setNumOfAngleLine(int value);
  CV_WRAP void setSigma(double value);

private:
  void afterHalfProjections(cv::Mat const &input, int D,
                            int xOff, int yOff);
  CV_WRAP void findFeatureVector();
  void firstHalfProjections(cv::Mat const &input, int D,
                            int xOff, int yOff);
  CV_WRAP void hashCalculate(cv::Mat &hash);
  CV_WRAP void radialProjections(cv::Mat const &input);

  cv::Mat blurImg_;
  std::vector<double> features_;
  cv::Mat grayImg_;
  int numOfAngelLine_;
  cv::Mat pixPerLine_;
  cv::Mat projections_;
  double sigma_;
};

//! @}
}
}

#endif // __OPENCV_RADIAL_VARIANCE_HASH_HPP__
