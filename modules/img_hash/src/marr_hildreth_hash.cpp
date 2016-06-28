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

namespace{

void getMHKernel(float alpha, float level, cv::Mat &kernel)
{
    int const sigma = static_cast<int>(4*std::pow(alpha,level));

    float const ratio = std::pow(alpha, -level);
    kernel.create(2*sigma+1, 2*sigma+1, CV_32F);
    for(int row = 0; row != kernel.rows; ++row)
    {
        float const ydiff = static_cast<float>(row - sigma);
        float const ypos = ratio * ydiff;
        float const yposPow2 = ypos * ypos;
        float *kPtr = kernel.ptr<float>(row);
        for(int col = 0; col != kernel.cols; ++col)
        {
            float const xpos = ratio * static_cast<float>((col - sigma));
            float const a = xpos * xpos + yposPow2;
            kPtr[col] = (2-a)*std::exp(a/2);
        }
    }
}

void fillBlocks(cv::Mat const &freImg, cv::Mat &blocks)
{
    //TODO : use forEach may provide better speed, however,
    //it is quite tedious to apply without lambda
    blocks.setTo(0);
    for(int row = 0; row != blocks.rows; ++row)
    {
        float *bptr = blocks.ptr<float>(row);
        int const rOffset = row*16;
        for(int col = 0; col != blocks.cols; ++col)
        {
            cv::Rect const roi(rOffset,col*16,16,16);
            bptr[col] =
                    static_cast<float>(cv::sum(freImg(roi))[0]);
        }
    }
}

void createHash(cv::Mat const &blocks, cv::Mat &hash)
{
    int hash_index = 0;
    int bit_index = 0;
    uchar hashbyte = 0;
    uchar *hashPtr = hash.ptr<uchar>(0);
    for (int row=0; row < 29; row += 4)
    {
        for (int col=0; col < 29; col += 4)
        {
            cv::Rect const roi(col,row,3,3);
            cv::Mat const blockROI = blocks(roi);
            float const avg =
                    static_cast<float>(cv::sum(blockROI)[0]/9.0);
            for(int i = 0; i != blockROI.rows; ++i)
            {
                float const *bptr = blockROI.ptr<float>(i);
                for(int j = 0; j != blockROI.cols; ++j)
                {
                    hashbyte <<= 1;
                    if (bptr[j] > avg)
                    {
                        hashbyte |= 0x01;
                    }
                    ++bit_index;
                    if ((bit_index%8) == 0)
                    {
                        hash_index = (bit_index/8) - 1;
                        hashPtr[hash_index] = hashbyte;
                        hashbyte = 0x00;
                    }
                }
            }
        }
    }
}

}

MarrHildrethHash::MarrHildrethHash(float alpha, float scale) :
    alphaVal(alpha),
    scaleVal(scale)
{
    getMHKernel(alphaVal, scaleVal, mhKernel);
    blocks.create(31,31, CV_32F);
}

MarrHildrethHash::~MarrHildrethHash()
{

}

void MarrHildrethHash::compute(cv::InputArray inputArr,
                               cv::OutputArray outputArr)
{
    cv::Mat const input = inputArr.getMat();
    CV_Assert(input.type() == CV_8UC3 ||
              input.type() == CV_8U);

    if (input.type() == CV_8UC3){
        cv::cvtColor(input, grayImg, CV_BGR2GRAY);
    } else{
        input.copyTo(grayImg);
    }
    //pHash use Canny-deritch filter to blur the image
    cv::GaussianBlur(grayImg, blurImg, cv::Size(7, 7), 0);
    cv::resize(blurImg, resizeImg, cv::Size(512, 512), 0, 0, INTER_CUBIC);
    cv::equalizeHist(resizeImg, equalizeImg);

    //extract frequency info by mh kernel
    cv::filter2D(equalizeImg, freImg, CV_32F, mhKernel);
    fillBlocks(freImg, blocks);

    outputArr.create(1, 72, CV_8U);
    cv::Mat hash = outputArr.getMat();
    createHash(blocks, hash);
}

double MarrHildrethHash::compare(cv::InputArray hashOne,
                                 cv::InputArray hashTwo) const
{
    return norm(hashOne, hashTwo, NORM_HAMMING);
}

float MarrHildrethHash::getAlpha() const
{
    return alphaVal;
}

float MarrHildrethHash::getScale() const
{
    return scaleVal;
}

void MarrHildrethHash::setKernelParam(float alpha, float scale)
{
    alphaVal = alpha;
    scaleVal = scale;
    getMHKernel(alphaVal, scaleVal, mhKernel);
}

Ptr<MarrHildrethHash> MarrHildrethHash::create(float alpha, float scale)
{
    return makePtr<MarrHildrethHash>(alpha, scale);
}

void marrHildrethHash(cv::InputArray inputArr,
                      cv::OutputArray outputArr,
                      float alpha, float scale)
{
    MarrHildrethHash(alpha, scale).compute(inputArr, outputArr);
}

}

}
