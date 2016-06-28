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

void BlockMeanHash::compute(cv::InputArray inputArr,
                            cv::OutputArray outputArr)
{
    cv::Mat const input = inputArr.getMat();
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
    switch(mode_)
    {
    case 0:
    {
        numOfBlocks = blockPerCol * blockPerRow;
        break;
    }
    case 1:
    {
        pixColStep /= 2;
        pixRowStep /= 2;
        numOfBlocks = (blockPerCol*2-1) * (blockPerRow*2-1);
        break;
    }
    default:
        break;
    }

    mean_.resize(numOfBlocks);
    findMean(pixRowStep, pixColStep);
    outputArr.create(1, numOfBlocks/8 + numOfBlocks % 8, CV_8U);
    cv::Mat hash = outputArr.getMat();
    createHash(hash);
}

double BlockMeanHash::compare(cv::InputArray hashOne,
                              cv::InputArray hashTwo) const
{
    return norm(hashOne, hashTwo, NORM_HAMMING);
}

Ptr<BlockMeanHash> BlockMeanHash::create(size_t mode)
{
    return makePtr<BlockMeanHash>(mode);
}

void BlockMeanHash::setMode(size_t mode)
{
    CV_Assert(mode == 0 || mode == 1);
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

void BlockMeanHash::findMean(int pixRowStep, int pixColStep)
{
    size_t blockIdx = 0;
    for(int row = 0; row <= rowSize; row += pixRowStep)
    {
        for(int col = 0; col <= colSize; col += pixColStep)
        {
            mean_[blockIdx++] = cv::mean(grayImg_(cv::Rect(col, row, blockWidth, blockHeigth)))[0];
        }
    }
}

void blockMeanHash(cv::InputArray inputArr,
                   cv::OutputArray outputArr,
                   size_t mode)
{
    BlockMeanHash(mode).compute(inputArr, outputArr);
}

}

}
