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

#ifndef __OPENCV_BLOCK_MEAN_HASH_HPP__
#define __OPENCV_BLOCK_MEAN_HASH_HPP__

#include "opencv2/core.hpp"

namespace cv
{

namespace img_hash
{
//! @addtogroup block_mean_hash
//! @{

/** @brief Computes block mean hash of the input image
    @param inputArr Input CV_8UC3, CV_8UC1 array.
    @param outputArr Hash value of input, it will contain 16 hex
    decimal number, return type is CV_8U
    @param mode By now only support mode 0 and mode 1.
    mode 0 : use fewer block and generate 16*16/8 uchar hash value
    mode 1 : use block blocks(step sizes/2), generate 31*31/8 + 1 uchar hash value
     */
CV_EXPORTS_W void blockMeanHash(cv::InputArray inputArr,
                                cv::OutputArray outputArr,
                                size_t mode = 0);

class CV_EXPORTS_W BlockMeanHash : public ImgHashBase
{
  //This friend class is design for unit test only, please do not
  //use it under normal case
  friend class BlockMeanHashTester;
public:
   /** @brief Constructor
        @param mode By now only support mode 0 and mode 1.
        mode 0 : use fewer block and generate 16*16/8 uchar hash value
        mode 1 : use block blocks(step sizes/2), generate 31*31/8 + 1 uchar hash value
    */
    CV_WRAP explicit BlockMeanHash(size_t mode = 0);
    CV_WRAP ~BlockMeanHash();

    /** @brief Computes Block mean hash of the input image
        @param inputArr input CV_8UC3, CV_8UC1 array
        @param outputArr hash of the image
      */
    CV_WRAP virtual void compute(cv::InputArray inputArr,
                                 cv::OutputArray outputArr);

    /** @brief Compare the hash value between inOne and inTwo
      @param hashOne Hash value one
      @param hashTwo Hash value two
      @return zero means the images are likely very similar;
      5 means a few things maybe different; 10 or more means
      they maybe are very different image
      */
    CV_WRAP virtual double compare(cv::InputArray hashOne,
                                   cv::InputArray hashTwo) const;

    /** @brief Create BLockMeanHash object
        @param mode By now only support mode 0 and mode 1.
        mode 0 : use fewer block and generate 16*16/8 uchar hash value
        mode 1 : use block blocks(step sizes/2), generate 31*31/8 + 1 uchar hash value
    */
    CV_WRAP static Ptr<BlockMeanHash> create(size_t mode = 0);

    CV_WRAP void setMode(size_t mode);

private:
    void createHash(cv::Mat &hash);
    void findMean(int pixRowStep, int pixColStep);

    cv::Mat grayImg_;
    std::vector<double> mean_;
    size_t mode_;
    cv::Mat resizeImg_;
};

//! @}
}
}

#endif // __OPENCV_PHASH_HPP__
