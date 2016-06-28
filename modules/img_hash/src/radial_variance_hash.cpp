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
#include <limits>
#include <numeric>

namespace cv{

namespace img_hash{

namespace{

enum{

hashSize = 40,

};

inline
float roundingFactor(float val)
{
    return val >= 0 ? 0.5f : -0.5f;
}

inline
int createOffSet(int length)
{
    float const center = static_cast<float>(length/2);
    return static_cast<int>(std::floor(center +
                                       roundingFactor(center)));
}

}

RadialVarianceHash::RadialVarianceHash(double sigma,
                                       int numOfAngleLine) :
    numOfAngelLine_(numOfAngleLine),
    sigma_(sigma)
{
}

RadialVarianceHash::~RadialVarianceHash()
{

}

void RadialVarianceHash::compute(cv::InputArray inputArr,
                                 cv::OutputArray outputArr)
{
    cv::Mat const input = inputArr.getMat();
    CV_Assert(input.type() == CV_8UC3 ||
              input.type() == CV_8U);

    if(input.type() == CV_8UC3)
    {
        cv::cvtColor(input, grayImg_, CV_BGR2GRAY);
    }
    else{
        grayImg_ = input;
    }

    cv::GaussianBlur(grayImg_, blurImg_, cv::Size(0,0), sigma_, sigma_);
    radialProjections(blurImg_);
    findFeatureVector();
    outputArr.create(1, hashSize, CV_8U);
    cv::Mat hash = outputArr.getMat();
    hashCalculate(hash);
}

double RadialVarianceHash::compare(cv::InputArray hashOne,
                                   cv::InputArray hashTwo) const
{
    cv::Mat const hashOneF = hashOne.getMat();
    cv::Mat const hashTwoF = hashTwo.getMat();
    CV_Assert(hashOneF.cols == hashSize && hashOneF.cols == hashTwoF.cols);

    float bufferOne[hashSize];
    cv::Mat hashFloatOne(1, hashSize, CV_32F, bufferOne);
    hashOneF.convertTo(hashFloatOne, CV_32F);

    float bufferTwo[hashSize];
    cv::Mat hashFloatTwo(1, hashSize, CV_32F, bufferTwo);
    hashTwoF.convertTo(hashFloatTwo, CV_32F);

    int const pixNum = hashFloatOne.rows * hashFloatOne.cols;
    cv::Scalar hOneMean, hOneStd, hTwoMean, hTwoStd;
    cv::meanStdDev(hashFloatOne, hOneMean, hOneStd);
    cv::meanStdDev(hashFloatTwo, hTwoMean, hTwoStd);

    // Compute covariance and correlation coefficient
    hashFloatOne -= hOneMean;
    hashFloatTwo -= hTwoMean;
    double max = std::numeric_limits<double>::min();
    for(int i = 0; i != hashSize; ++i)
    {
        double const covar = (hashFloatOne).dot(hashFloatTwo) / pixNum;
        double const corre = covar / (hOneStd[0] * hTwoStd[0] + 1e-20);
        max = std::max(corre, max);
        //move last value to first position, first value to second position,
        //second value to third position and so on
        float const preValue = bufferTwo[hashSize-1];
        std::copy_backward(bufferTwo, bufferTwo + hashSize - 1, bufferTwo + hashSize);
        bufferTwo[0] = preValue;
    }

    //return peak correlation coefficient
    return max;
}

Ptr<RadialVarianceHash> RadialVarianceHash::create(double sigma,
                                                   int numOfAngleLine)
{
    return makePtr<RadialVarianceHash>(sigma, numOfAngleLine);
}

void RadialVarianceHash::
afterHalfProjections(const Mat &input, int D, int xOff, int yOff)
{
    int *pplPtr = pixPerLine_.ptr<int>(0);
    int const init = 3*numOfAngelLine_/4;
    for(int k = init, j = 0; k < numOfAngelLine_; ++k, j += 2)
    {
        float const theta = k*3.14159f/numOfAngelLine_;
        float const alpha = std::tan(theta);
        uchar *projDown = projections_.ptr<uchar>(k);
        uchar *projUp = projections_.ptr<uchar>(k-j);
        for(int x = 0; x < D; ++x)
        {
            float const y = alpha*(x-xOff);
            int const yd = static_cast<int>(std::floor(y + roundingFactor(y)));
            if((yd + yOff >= 0)&&(yd + yOff < input.rows) && (x < input.cols))
            {
                projDown[x] = input.at<uchar>(yd+yOff, x);
                pplPtr[k] += 1;
            }
            if ((yOff - yd >= 0)&&(yOff - yd < input.cols)&&
                    (2*yOff - x >= 0)&&(2*yOff- x < input.rows)&&
                    (k != init))
            {
                projUp[x] =
                        input.at<uchar>(-(x-yOff)+yOff, -yd+yOff);
                pplPtr[k-j] += 1;
            }
        }
    }
}

void RadialVarianceHash::findFeatureVector()
{
    features_.resize(numOfAngelLine_);
    double sum = 0.0;
    double sumSqd = 0.0;
    int const *pplPtr = pixPerLine_.ptr<int>(0);
    for(int k=0; k < numOfAngelLine_; ++k)
    {
        double lineSum = 0.0;
        double lineSumSqd = 0.0;
        //original implementation of pHash may generate zero pixNum, this
        //will cause NaN value and make the features become less discriminative
        //to avoid this problem, I add a small value--0.00001
        double const pixNum = pplPtr[k] + 0.00001;
        double const pixNumPow2 = pixNum * pixNum;
        uchar const *projPtr = projections_.ptr<uchar>(k);
        for(int i = 0; i < projections_.cols; ++i)
        {
            double const value = projPtr[i];
            lineSum += value;
            lineSumSqd += value * value;
        }
        features_[k] = (lineSumSqd/pixNum) -
                (lineSum*lineSum)/(pixNumPow2);
        sum += features_[k];
        sumSqd += features_[k]*features_[k];
    }
    double const numOfALPow2 = numOfAngelLine_ * numOfAngelLine_;
    double const mean = sum/numOfAngelLine_;
    double const var  = std::sqrt((sumSqd/numOfAngelLine_) - (sum*sum)/(numOfALPow2));
    for(int i = 0; i < numOfAngelLine_; ++i)
    {
        features_[i] = (features_[i] - mean)/var;
    }
}

void RadialVarianceHash::
firstHalfProjections(Mat const &input, int D, int xOff, int yOff)
{
    int *pplPtr = pixPerLine_.ptr<int>(0);
    for(int k = 0; k < numOfAngelLine_/4+1; ++k)
    {
        float const theta = k*3.14159f/numOfAngelLine_;
        float const alpha = std::tan(theta);
        uchar *projOne = projections_.ptr<uchar>(k);
        uchar *projTwo = projections_.ptr<uchar>(numOfAngelLine_/2-k);
        for(int x = 0; x < D; ++x)
        {
            float const y = alpha*(x-xOff);
            int const yd = static_cast<int>(std::floor(y + roundingFactor(y)));
            if((yd + yOff >= 0)&&(yd + yOff < input.rows) && (x < input.cols))
            {
                projOne[x] = input.at<uchar>(yd+yOff, x);
                pplPtr[k] += 1;
            }
            if((yd + xOff >= 0) && (yd + xOff < input.cols) &&
                    (k != numOfAngelLine_/4) && (x < input.rows))
            {
                projTwo[x] =
                        input.at<uchar>(x, yd+xOff);
                pplPtr[numOfAngelLine_/2-k] += 1;
            }
        }
    }
}

void RadialVarianceHash::hashCalculate(cv::Mat &hash)
{
    double temp[hashSize];
    double max = 0;
    double min = 0;
    size_t const featureSize = features_.size();
    //constexpr is a better choice
    double const sqrtTwo = 1.4142135623730950488016887242097;
    for(int k = 0; k < hash.cols; ++k)
    {
        double sum = 0;
        for(size_t n = 0; n < featureSize; ++n)
        {
            sum += features_[n]*std::cos((3.14159*(2*n+1)*k)/(2*featureSize));
        }
        temp[k] = k == 0 ? sum/std::sqrt(featureSize) :
                           sum*sqrtTwo/std::sqrt(featureSize);
        if(temp[k] > max)
        {
            max = temp[k];
        }
        else if(temp[k] < min)
        {
            min = temp[k];
        }
    }

    double const range = max - min;
    if(range != 0)
    {
        //std::transform is a better choice if lambda supported
        uchar *hashPtr = hash.ptr<uchar>(0);
        for(int i = 0; i < hash.cols; ++i)
        {
            hashPtr[i] = static_cast<uchar>((255*(temp[i] - min)/range));
        }
    }
    else
    {
        hash = 0;
    }
}

void RadialVarianceHash::radialProjections(const Mat &input)
{
    int const D = (input.cols > input.rows) ? input.cols : input.rows;
    //Different with PHash, this part reverse the row size and col size,
    //because cv::Mat is row major but not column major
    projections_.create(numOfAngelLine_, D, CV_8U);
    projections_ = 0;
    pixPerLine_.create(1, numOfAngelLine_, CV_32S);
    pixPerLine_ = 0;
    int const xOff = createOffSet(input.cols);
    int const yOff = createOffSet(input.rows);

    firstHalfProjections(input, D, xOff, yOff);
    afterHalfProjections(input, D, xOff, yOff);
}

int RadialVarianceHash::getNumOfAngleLine() const
{
    return numOfAngelLine_;
}

double RadialVarianceHash::getSigma() const
{
    return sigma_;
}

void RadialVarianceHash::setNumOfAngleLine(int value)
{
    CV_Assert(value > 0);
    numOfAngelLine_ = value;
}

void RadialVarianceHash::setSigma(double value)
{
    CV_Assert(value >= 1.0);
    sigma_ = value;
}

void radialVarianceHash(cv::InputArray inputArr,
                        cv::OutputArray outputArr,
                        double sigma, int numOfAngleLine)
{
    RadialVarianceHash(sigma, numOfAngleLine).compute(inputArr, outputArr);
}

}

}
