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

PHash::~PHash()
{

}

void PHash::compute(cv::InputArray inputArr,
                    cv::OutputArray outputArr)
{
    cv::Mat const input = inputArr.getMat();
    CV_Assert(input.type() == CV_8UC3 ||
              input.type() == CV_8U);

    cv::resize(input, resizeImg, cv::Size(32,32));
    if(input.type() == CV_8UC3)
    {
        cv::cvtColor(resizeImg, grayImg, CV_BGR2GRAY);
    }
    else
    {
        grayImg = resizeImg;
    }

    grayImg.convertTo(grayFImg, CV_32F);
    cv::dct(grayFImg, dctImg);
    dctImg(cv::Rect(0, 0, 8, 8)).copyTo(topLeftDCT);
    topLeftDCT.at<float>(0, 0) = 0;
    float const imgMean = static_cast<float>(cv::mean(topLeftDCT)[0]);

    cv::compare(topLeftDCT, imgMean, bitsImg, CMP_GT);
    bitsImg /= 255;
    outputArr.create(1, 8, CV_8U);
    cv::Mat hash = outputArr.getMat();
    uchar *hash_ptr = hash.ptr<uchar>(0);
    uchar const *bits_ptr = bitsImg.ptr<uchar>(0);
    std::bitset<8> bits;
    for(size_t i = 0, j = 0; i != bitsImg.total(); ++j)
    {
        for(size_t k = 0; k != 8; ++k)
        {
            //avoid warning C4800, casting do not work
            bits[k] = bits_ptr[i++] != 0;
        }
        hash_ptr[j] = static_cast<uchar>(bits.to_ulong());
    }
}

double PHash::compare(cv::InputArray hashOne,
                      cv::InputArray hashTwo) const
{
    return norm(hashOne, hashTwo, NORM_HAMMING);
}

Ptr<PHash> PHash::create()
{
    return makePtr<PHash>();
}

void pHash(cv::InputArray inputArr,
           cv::OutputArray outputArr)
{
    PHash().compute(inputArr, outputArr);
}

}

}
