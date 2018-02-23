/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2016, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

/*
Contributed by Gregor Kovalcik <gregor dot kovalcik at gmail dot com>
    based on code provided by Martin Krulis, Jakub Lokoc and Tomas Skopal.

References:
    Martin Krulis, Jakub Lokoc, Tomas Skopal.
    Efficient Extraction of Clustering-Based Feature Signatures Using GPU Architectures.
    Multimedia tools and applications, 75(13), pp.: 8071–8103, Springer, ISSN: 1380-7501, 2016

    Christian Beecks, Merih Seran Uysal, Thomas Seidl.
    Signature quadratic form distance.
    In Proceedings of the ACM International Conference on Image and Video Retrieval, pages 438-445.
    ACM, 2010.
*/

#include "../precomp.hpp"

#include "pct_sampler.hpp"

namespace cv
{
    namespace xfeatures2d
    {
        namespace pct_signatures
        {
            class PCTSampler_Impl : public PCTSampler
            {
            private:
                /**
                * @brief Initial sampling point coordinates.
                */
                std::vector<Point2f> mInitSamplingPoints;

                /**
                * @brief Number of bits per pixel in grayscale image used for computing contrast and entropy.
                */
                int mGrayscaleBits;

                /**
                * @brief Radius of scanning window around a sampled point used for computing contrast and entropy.
                */
                int mWindowRadius;

                /**
                * @brief Weights of different feauture dimensions.
                *       Default values are 1;
                */
                std::vector<float> mWeights;

                /**
                * @brief Translation of different feauture dimensions.
                *       Default values are 0;
                */
                std::vector<float> mTranslations;


            public:

                PCTSampler_Impl(
                    const std::vector<Point2f>& initSamplingPoints,
                    int grayscaleBits,
                    int windowRadius)
                    : mInitSamplingPoints(initSamplingPoints),
                    mGrayscaleBits(grayscaleBits),
                    mWindowRadius(windowRadius)
                {
                    // Initialize weights and translation vectors to neutral values.
                    for (int i = 0; i < SIGNATURE_DIMENSION; i++)
                    {
                        mWeights.push_back(1.0);
                        mTranslations.push_back(0.0);
                    }
                }


                /**** Acessors ****/

                int getSampleCount() const      { return (int)mInitSamplingPoints.size(); }
                int getGrayscaleBits() const    { return mGrayscaleBits; }
                int getWindowRadius() const     { return mWindowRadius; }

                float getWeightX() const               { return mWeights[X_IDX]; }
                float getWeightY() const               { return mWeights[Y_IDX]; }
                float getWeightL() const               { return mWeights[L_IDX]; }
                float getWeightA() const               { return mWeights[A_IDX]; }
                float getWeightB() const               { return mWeights[B_IDX]; }
                float getWeightContrast() const        { return mWeights[CONTRAST_IDX]; }
                float getWeightEntropy() const         { return mWeights[ENTROPY_IDX]; }

                std::vector<Point2f> getSamplingPoints() const
                                                        { return mInitSamplingPoints; }


                void setGrayscaleBits(int grayscaleBits)    { mGrayscaleBits = grayscaleBits; }
                void setWindowRadius(int windowRadius)      { mWindowRadius = windowRadius; }

                void setWeightX(float weight)          { mWeights[X_IDX] = weight; }
                void setWeightY(float weight)          { mWeights[Y_IDX] = weight; }
                void setWeightL(float weight)          { mWeights[L_IDX] = weight; }
                void setWeightA(float weight)          { mWeights[A_IDX] = weight; }
                void setWeightB(float weight)          { mWeights[B_IDX] = weight; }
                void setWeightContrast(float weight)   { mWeights[CONTRAST_IDX] = weight; }
                void setWeightEntropy(float weight)    { mWeights[ENTROPY_IDX] = weight; }


                void setWeight(int idx, float value)
                {
                    mWeights[idx] = value;
                }

                void setWeights(const std::vector<float>& weights)
                {
                    if (weights.size() != mWeights.size())
                    {
                        CV_Error_(Error::StsUnmatchedSizes,
                            ("Invalid weights dimension %d (max %d)", weights.size(), mWeights.size()));
                    }
                    else
                    {
                        for (int i = 0; i < (int)(mWeights.size()); ++i)
                        {
                            mWeights[i] = weights[i];
                        }
                    }
                }

                void setTranslation(int idx, float value)
                {
                    mTranslations[idx] = value;
                }

                void setTranslations(const std::vector<float>& translations)
                {
                    if (translations.size() != mTranslations.size())
                    {
                        CV_Error_(Error::StsUnmatchedSizes,
                            ("Invalid translations dimension %d (max %d)", translations.size(), mTranslations.size()));
                    }
                    else
                    {
                        for (int i = 0; i < (int)(mTranslations.size()); ++i)
                        {
                            mTranslations[i] = translations[i];
                        }
                    }
                }

                void setSamplingPoints(std::vector<Point2f> samplingPoints) { mInitSamplingPoints = samplingPoints; }


                void sample(InputArray _image, OutputArray _samples) const
                {
                    // prepare matrices
                    Mat image = _image.getMat();
                    _samples.create((int)(mInitSamplingPoints.size()), SIGNATURE_DIMENSION, CV_32F);
                    Mat samples = _samples.getMat();
                    GrayscaleBitmap grayscaleBitmap(image, mGrayscaleBits);

                    // sample each sample point
                    for (int iSample = 0; iSample < (int)(mInitSamplingPoints.size()); iSample++)
                    {
                        // sampling points are in range [0..1)
                        int x = (int)(mInitSamplingPoints[iSample].x * (image.cols));
                        int y = (int)(mInitSamplingPoints[iSample].y * (image.rows));

                        // x, y normalized
                        samples.at<float>(iSample, X_IDX) = (float)((float)x / (float)image.cols * mWeights[X_IDX] + mTranslations[X_IDX]);
                        samples.at<float>(iSample, Y_IDX) = (float)((float)y / (float)image.rows * mWeights[Y_IDX] + mTranslations[Y_IDX]);

                        // get Lab pixel color
                        Mat rgbPixel(image, Rect(x, y, 1, 1));
                        Mat labPixel;
                        rgbPixel.convertTo(rgbPixel, CV_32FC3, 1.0 / 255);
                        cvtColor(rgbPixel, labPixel, COLOR_BGR2Lab);
                        Vec3f labColor = labPixel.at<Vec3f>(0, 0);

                        // Lab color normalized
                        samples.at<float>(iSample, L_IDX) = (float)(std::floor(labColor[0] + 0.5) / L_COLOR_RANGE * mWeights[L_IDX] + mTranslations[L_IDX]);
                        samples.at<float>(iSample, A_IDX) = (float)(std::floor(labColor[1] + 0.5) / A_COLOR_RANGE * mWeights[A_IDX] + mTranslations[A_IDX]);
                        samples.at<float>(iSample, B_IDX) = (float)(std::floor(labColor[2] + 0.5) / B_COLOR_RANGE * mWeights[B_IDX] + mTranslations[B_IDX]);

                        // contrast and entropy
                        float contrast = 0.0, entropy = 0.0;
                        grayscaleBitmap.getContrastEntropy(x, y, contrast, entropy, mWindowRadius);     // HOT PATH: 30%
                        samples.at<float>(iSample, CONTRAST_IDX)
                            = (float)(contrast / SAMPLER_CONTRAST_NORMALIZER * mWeights[CONTRAST_IDX] + mTranslations[CONTRAST_IDX]);
                        samples.at<float>(iSample, ENTROPY_IDX)
                            = (float)(entropy / SAMPLER_ENTROPY_NORMALIZER * mWeights[ENTROPY_IDX] + mTranslations[ENTROPY_IDX]);
                    }
                }

            };


            Ptr<PCTSampler> PCTSampler::create(
                const std::vector<Point2f>& initPoints,
                int                         grayscaleBits,
                int                         windowRadius)
            {
                return makePtr<PCTSampler_Impl>(initPoints, grayscaleBits, windowRadius);
            }
        }
    }
}
