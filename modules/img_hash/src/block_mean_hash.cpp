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

#include <bitset>
#include <iostream>

namespace cv{

namespace img_hash{

namespace{

enum
{
    imgWidth = 256,
    imgHeight = 256,
    blockWidth = 16,
    blockHeigth = 16,
    blockPerCol = imgHeight / blockHeigth,
    blockPerRow = imgWidth / blockWidth,
    rowSize = imgHeight - blockHeigth,
    colSize = imgWidth - blockWidth
};

}

BlockMeanHash::BlockMeanHash(size_t mode)
{
    setMode(mode);
}

BlockMeanHash::~BlockMeanHash()
{

}

void BlockMeanHash::compute(const Mat &input, Mat &hash)
{
    CV_Assert(input.type() == CV_8UC3 ||
              input.type() == CV_8U);

    cv::resize(input, resizeImg_, cv::Size(imgWidth,imgHeight));
    if(input.type() == CV_8UC3)
    {
        cv::cvtColor(resizeImg_, grayImg_, CV_BGR2GRAY);
    }
    else
    {
        grayImg_ = resizeImg_;
    }

    int pixColStep = blockWidth;
    int pixRowStep = blockHeigth;
    int numOfBlocks = 0;
    int hashSize = 0;
    switch(mode_)
    {
    case 0:
    {
        numOfBlocks = blockPerCol * blockPerRow;
        hashSize = numOfBlocks/8;
        break;
    }
    case 1:
    {
        pixColStep /= 2;
        pixRowStep /= 2;
        numOfBlocks = (blockPerCol*2-1) * (blockPerRow*2-1);
        hashSize = numOfBlocks/8 + 1;
        break;
    }
    case 2:
    {
       numOfBlocks = blockPerCol * blockPerRow * 24;
       hashSize =  (numOfBlocks/8) * 24;
       break;
    }
    default:
        break;
    }

    mean_.resize(numOfBlocks);
    hash.create(1, hashSize, CV_8U);
    findMean(hash, pixRowStep, pixColStep);
    //hash.create(1, hashSize, CV_8U);
    //createHash(hash);
}

double BlockMeanHash::compare(cv::Mat const &hashOne, cv::Mat const &hashTwo) const
{
    return norm(hashOne, hashTwo, NORM_HAMMING);
}

Ptr<BlockMeanHash> BlockMeanHash::create(size_t mode)
{
    return makePtr<BlockMeanHash>(mode);
}

void BlockMeanHash::setMode(size_t mode)
{
    CV_Assert(mode == 0 || mode == 1 || mode == 2);
    mode_ = mode;
}

void BlockMeanHash::createHash(Mat &hash)
{
    double const median = cv::mean(grayImg_)[0];
    uchar *hashPtr = hash.ptr<uchar>(0);
    std::bitset<8> bits = 0;
    for(size_t i = 0; i < mean_.size(); ++i)
    {
        size_t const residual = i%8;
        bits[residual] = mean_[i] < median ? 0 : 1;
        if(residual == 7)
        {
            *hashPtr = static_cast<uchar>(bits.to_ulong());
            ++hashPtr;
        }else if(i == mean_.size() - 1)
        {
            *hashPtr = bits[residual];
        }
    }   
}

uchar *BlockMeanHash::createHash(uchar *hashPtr, double median,
                                 int beg, int end)
{
    std::bitset<8> bits = 0;
    for(int i = beg; i < end; ++i)
    {
        size_t const residual = i%8;
        bits[residual] = mean_[i] < median ? 0 : 1;
        if(residual == 7)
        {
            *hashPtr = static_cast<uchar>(bits.to_ulong());
            ++hashPtr;
        }
    }

    return hashPtr;
}

void BlockMeanHash::findMean(cv::Mat &hash, int pixRowStep, int pixColStep)
{
    int const maxEngle = mode_ < 2 ? 15 : 360;
    int blockIdx = 0;
    uchar *hashPtr = hash.ptr<uchar>(0);
    int meanBeg = 0;
    for(int angle = 0; angle != maxEngle; angle += 15)
    {
        cv::Mat rotateRef;
        if(angle == 0)
        {
            rotateRef = grayImg_;
        }
        else
        {
            cv::Point2f const center(static_cast<float>(grayImg_.cols),
                                     static_cast<float>(grayImg_.rows));
            cv::Mat const rotMat =
                    cv::getRotationMatrix2D(center,
                                            angle, 1.0);
            cv::warpAffine(grayImg_, rotateMean_, rotMat, grayImg_.size());
            rotateRef = rotateMean_;
        }

        for(int row = 0; row <= rowSize; row += pixRowStep)
        {
            for(int col = 0; col <= colSize; col += pixColStep)
            {
                mean_[blockIdx++] =
                        cv::mean(rotateRef(cv::Rect(col, row,
                                                    blockWidth, blockHeigth)))[0];
            }
        }
        createHash(hashPtr, cv::mean(rotateRef)[0],
                   meanBeg, blockIdx);
        meanBeg = blockIdx;
    }
}

void blockMeanHash(cv::Mat const &input, cv::Mat &hash,
                   size_t mode)
{
    BlockMeanHash(mode).compute(input, hash);
}

}

}
