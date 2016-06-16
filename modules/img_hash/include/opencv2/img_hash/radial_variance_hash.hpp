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
    @param input Input CV_8UC3, CV_8UC1 array.
    @param hash Hash value of input
     */
CV_EXPORTS void radialVarianceHash(cv::Mat const &input, cv::Mat &hash);

class RadialVarianceHash : public ImgHashBase
{
  //This friend class is design for unit test, please do not
  //use it under normal case
  friend class RadialVarHashTester;
public:
  /** @brief Constructor
      @param sigma Gaussian kernel standard deviation
      @param gamma Gamma correction on the input image
      @param numOfAngleLine The number of angles to consider
   */
  CV_EXPORTS explicit RadialVarianceHash(double sigma = 1,
                                         float gamma = 1.0f,
                                         int numOfAngleLine = 180);
  CV_EXPORTS ~RadialVarianceHash();

  /** @brief Computes average hash of the input image
      @param input input image want to compute hash value
      @param hash hash of the image, contain 40 uchar value
  */
  virtual void compute(cv::Mat const &input, cv::Mat &hash);

  /** @brief Compare the hash value between inOne and inTwo
  @param hashOne Hash value one
  @param hashTwo Hash value two
  @return cross correlation of two hash, the closer the value to 1,
  the more similar the hash values. We could assume the threshold is 0.9
  by default
  */
  CV_EXPORTS virtual double compare(cv::Mat const &hashOne, cv::Mat const &hashTwo) const;

  CV_EXPORTS static Ptr<RadialVarianceHash> create();    

private:
  void afterHalfProjections(cv::Mat const &input, int D,
                            int xOff, int yOff);
  CV_EXPORTS void findFeatureVector();
  void firstHalfProjections(cv::Mat const &input, int D,
                            int xOff, int yOff);
  CV_EXPORTS void hashCalculate(cv::Mat &hash);
  CV_EXPORTS void radialProjections(cv::Mat const &input);

  float gamma_;
  cv::Mat blurImg_;
  std::vector<double> features_;
  cv::Mat gammaImg_;
  cv::Mat grayImg_;
  cv::Mat normalizeImg_;
  int numOfAngelLine_;
  cv::Mat pixPerLine_;
  cv::Mat projections_;
  double sigma_;  
};

//! @}
}
}

#endif // __OPENCV_RADIAL_VARIANCE_HASH_HPP__
