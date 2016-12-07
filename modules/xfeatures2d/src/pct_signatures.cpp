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
#include "precomp.hpp"

#include "pct_signatures/constants.hpp"
#include "pct_signatures/pct_sampler.hpp"
#include "pct_signatures/pct_clusterizer.hpp"

using namespace cv::xfeatures2d::pct_signatures;

namespace cv
{
    namespace xfeatures2d
    {
        namespace pct_signatures
        {
            class PCTSignatures_Impl : public PCTSignatures
            {
            public:
                PCTSignatures_Impl(const std::vector<Point2f>& initSamplingPoints, int initSeedCount)
                {
                    if (initSamplingPoints.size() == 0)
                    {
                        CV_Error(Error::StsBadArg, "No sampling points provided!");
                    }
                    if (initSeedCount <= 0)
                    {
                        CV_Error(Error::StsBadArg, "Not enough initial seeds, at least 1 required.");
                    }

                    mSampler = PCTSampler::create(initSamplingPoints);

                    initSeedCount = std::min(initSeedCount, (int)initSamplingPoints.size());
                    std::vector<int> initClusterSeedIndexes = pickRandomClusterSeedIndexes(initSeedCount);
                    mClusterizer = PCTClusterizer::create(initClusterSeedIndexes);
                }

                PCTSignatures_Impl(
                    const std::vector<Point2f>& initSamplingPoints,
                    const std::vector<int>& initClusterSeedIndexes)
                {
                    if (initSamplingPoints.size() == 0)
                    {
                        CV_Error(Error::StsBadArg, "No sampling points provided!");
                    }
                    if (initClusterSeedIndexes.size() == 0)
                    {
                        CV_Error(Error::StsBadArg, "Not enough initial seeds, at least 1 required.");
                    }
                    if (initClusterSeedIndexes.size() > initSamplingPoints.size())
                    {
                        CV_Error(Error::StsBadArg, "Too much cluster seeds or not enough sampling points.");
                    }
                    for (int iCluster = 0; iCluster < (int)(initClusterSeedIndexes.size()); iCluster++)
                    {
                        if (initClusterSeedIndexes[iCluster] < 0
                            || initClusterSeedIndexes[iCluster] >= (int)(initSamplingPoints.size()))
                        {
                            CV_Error(Error::StsBadArg,
                                "Initial cluster seed indexes contain an index outside the range of the sampling point list.");
                        }
                    }

                    mSampler = PCTSampler::create(initSamplingPoints);
                    mClusterizer = PCTClusterizer::create(initClusterSeedIndexes);
                }

                void computeSignature(InputArray image, OutputArray signature) const;

                void computeSignatures(const std::vector<Mat>& images, std::vector<Mat>& signatures) const;

                void getGrayscaleBitmap(OutputArray _grayscaleBitmap, bool normalize) const;

                /**** sampler ****/

                int getSampleCount() const                      { return mSampler->getGrayscaleBits(); }
                int getGrayscaleBits() const                    { return mSampler->getGrayscaleBits(); }
                int getWindowRadius() const                     { return mSampler->getWindowRadius(); }
                float getWeightX() const                        { return mSampler->getWeightX(); }
                float getWeightY() const                        { return mSampler->getWeightY(); }
                float getWeightL() const                        { return mSampler->getWeightL(); }
                float getWeightA() const                        { return mSampler->getWeightA(); }
                float getWeightB() const                        { return mSampler->getWeightB(); }
                float getWeightConstrast() const                { return mSampler->getWeightConstrast(); }
                float getWeightEntropy() const                  { return mSampler->getWeightEntropy(); }

                std::vector<Point2f> getSamplingPoints() const  { return mSampler->getSamplingPoints(); }


                void setGrayscaleBits(int grayscaleBits)        { mSampler->setGrayscaleBits(grayscaleBits); }
                void setWindowRadius(int windowRadius)          { mSampler->setWindowRadius(windowRadius); }
                void setWeightX(float weight)                   { mSampler->setWeightX(weight); }
                void setWeightY(float weight)                   { mSampler->setWeightY(weight); }
                void setWeightL(float weight)                   { mSampler->setWeightL(weight); }
                void setWeightA(float weight)                   { mSampler->setWeightA(weight); }
                void setWeightB(float weight)                   { mSampler->setWeightB(weight); }
                void setWeightContrast(float weight)            { mSampler->setWeightContrast(weight); }
                void setWeightEntropy(float weight)             { mSampler->setWeightEntropy(weight); }

                void setWeight(int idx, float value)                           { mSampler->setWeight(idx, value); }
                void setWeights(const std::vector<float>& weights)             { mSampler->setWeights(weights); }
                void setTranslation(int idx, float value)                      { mSampler->setTranslation(idx, value); }
                void setTranslations(const std::vector<float>& translations)   { mSampler->setTranslations(translations); }

                void setSamplingPoints(std::vector<Point2f> samplingPoints)    { mSampler->setSamplingPoints(samplingPoints); }


                /**** clusterizer ****/

                int getIterationCount() const                       { return mClusterizer->getIterationCount(); }
                std::vector<int> getInitSeedIndexes() const         { return mClusterizer->getInitSeedIndexes(); }
                int getInitSeedCount() const                        { return (int)mClusterizer->getInitSeedIndexes().size(); }
                int getMaxClustersCount() const                     { return mClusterizer->getMaxClustersCount(); }
                int getClusterMinSize() const                       { return mClusterizer->getClusterMinSize(); }
                float getJoiningDistance() const                    { return mClusterizer->getJoiningDistance(); }
                float getDropThreshold() const                      { return mClusterizer->getDropThreshold(); }
                int getDistanceFunction() const
                                                                    { return mClusterizer->getDistanceFunction(); }

                void setIterationCount(int iterations)              { mClusterizer->setIterationCount(iterations); }
                void setInitSeedIndexes(std::vector<int> initSeedIndexes)
                                                                    { mClusterizer->setInitSeedIndexes(initSeedIndexes); }
                void setMaxClustersCount(int maxClusters)           { mClusterizer->setMaxClustersCount(maxClusters); }
                void setClusterMinSize(int clusterMinSize)          { mClusterizer->setClusterMinSize(clusterMinSize); }
                void setJoiningDistance(float joiningDistance)      { mClusterizer->setJoiningDistance(joiningDistance); }
                void setDropThreshold(float dropThreshold)          { mClusterizer->setDropThreshold(dropThreshold); }
                void setDistanceFunction(int distanceFunction)
                                                                    { mClusterizer->setDistanceFunction(distanceFunction); }

            private:
                /**
                * @brief Samples used for sampling the input image and producing list of samples.
                */
                Ptr<PCTSampler> mSampler;

                /**
                * @brief Clusterizer using k-means algorithm to produce list of centroids from sampled points - the image signature.
                */
                Ptr<PCTClusterizer> mClusterizer;

                /**
                * @brief Creates vector of random indexes of sampling points
                *       which will be used as initial centroids for k-means clusterization.
                * @param initSeedCount Number of indexes of initial centroids to be produced.
                * @return The generated vector of random indexes.
                */
                static std::vector<int> pickRandomClusterSeedIndexes(int initSeedCount)
                {
                    std::vector<int> seedIndexes;
                    for (int iSeed = 0; iSeed < initSeedCount; iSeed++)
                    {
                        seedIndexes.push_back(iSeed);
                    }

                    randShuffle(seedIndexes);
                    return seedIndexes;
                }
            };


            /**
            * @brief Class implementing parallel computing of signatures for multiple images.
            */
            class Parallel_computeSignatures : public ParallelLoopBody
            {
            private:
                const PCTSignatures* mPctSignaturesAlgorithm;
                const std::vector<Mat>* mImages;
                std::vector<Mat>* mSignatures;

            public:
                Parallel_computeSignatures(
                    const PCTSignatures* pctSignaturesAlgorithm,
                    const std::vector<Mat>* images,
                    std::vector<Mat>* signatures)
                    : mPctSignaturesAlgorithm(pctSignaturesAlgorithm),
                    mImages(images),
                    mSignatures(signatures)
                {
                    mSignatures->resize(images->size());
                }

                void operator()(const Range& range) const
                {
                    for (int i = range.start; i < range.end; i++)
                    {
                        mPctSignaturesAlgorithm->computeSignature((*mImages)[i], (*mSignatures)[i]);
                    }
                }
            };


            /**
            * @brief Computes signature for one image.
            */
            void PCTSignatures_Impl::computeSignature(InputArray _image, OutputArray _signature) const
            {
                if (_image.empty())
                {
                    _signature.create(_image.size(), CV_32FC1);
                    return;
                }

                Mat image = _image.getMat();
                CV_Assert(image.depth() == CV_8U);

                // TODO: OpenCL
                //if (ocl::useOpenCL())
                //{
                //}

                // sample features
                Mat samples;
                mSampler->sample(image, samples);               // HOT PATH: 40%

                // kmeans clusterize, use feature samples, produce signature clusters
                Mat signature;
                mClusterizer->clusterize(samples, signature);   // HOT PATH: 60%


                // set result
                _signature.create(signature.size(), signature.type());
                Mat result = _signature.getMat();
                signature.copyTo(result);
            }

            /**
            * @brief Parallel function computing signatures for multiple images.
            */
            void PCTSignatures_Impl::computeSignatures(const std::vector<Mat>& images, std::vector<Mat>& signatures) const
            {
                parallel_for_(Range(0, (int)images.size()), Parallel_computeSignatures(this, &images, &signatures));
            }

        } // end of namespace pct_signatures



        Ptr<PCTSignatures> PCTSignatures::create(
            const int initSampleCount,
            const int initSeedCount,
            const int pointDistribution)
        {
            std::vector<Point2f> initPoints;
            generateInitPoints(initPoints, initSampleCount, pointDistribution);
            return create(initPoints, initSeedCount);
        }

        Ptr<PCTSignatures> PCTSignatures::create(
            const std::vector<Point2f>& initPoints,
            const int initSeedCount)
        {
            return makePtr<PCTSignatures_Impl>(initPoints, initSeedCount);
        }

        Ptr<PCTSignatures> PCTSignatures::create(
            const std::vector<Point2f>& initPoints,
            const std::vector<int>& initClusterSeedIndexes)
        {
            return makePtr<PCTSignatures_Impl>(initPoints, initClusterSeedIndexes);
        }


        void PCTSignatures::drawSignature(
            InputArray _source,
            InputArray _signature,
            OutputArray _result,
            float radiusToShorterSideRatio,
            int borderThickness)
        {
            // check source
            if (_source.empty())
            {
                return;
            }
            Mat source = _source.getMat();

            // create result
            _result.create(source.size(), source.type());
            Mat result = _result.getMat();
            source.copyTo(result);

            // check signature
            if (_signature.empty())
            {
                return;
            }
            Mat signature = _signature.getMat();
            if (signature.type() != CV_32F || signature.cols != SIGNATURE_DIMENSION)
            {
                CV_Error_(Error::StsBadArg, ("Invalid signature format. Type must be CV_32F and signature.cols must be %d.", SIGNATURE_DIMENSION));
            }

            // compute max radius using given ratio of shorter image side
            float maxRadius = ((source.rows < source.cols) ? source.rows : source.cols) * radiusToShorterSideRatio;

            // draw signature
            for (int i = 0; i < signature.rows; i++)
            {
                Vec3f labColor(
                    signature.at<float>(i, L_IDX) * L_COLOR_RANGE,      // convert Lab pixel to BGR
                    signature.at<float>(i, A_IDX) * A_COLOR_RANGE,
                    signature.at<float>(i, B_IDX) * B_COLOR_RANGE);
                Mat labPixel(1, 1, CV_32FC3);
                labPixel.at<Vec3f>(0, 0) = labColor;
                Mat rgbPixel;
                cvtColor(labPixel, rgbPixel, COLOR_Lab2BGR);
                rgbPixel.convertTo(rgbPixel, CV_8UC3, 255);
                Vec3b rgbColor = rgbPixel.at<Vec3b>(0, 0);

                // precompute variables
                Point center((int)(signature.at<float>(i, X_IDX) * source.cols), (int)(signature.at<float>(i, Y_IDX) * source.rows));
                int radius = (int)(maxRadius * signature.at<float>(i, WEIGHT_IDX));
                Vec3b borderColor(0, 0, 0);

                // draw filled circle
                circle(result, center, radius, rgbColor, -1);
                // draw circle outline
                circle(result, center, radius, borderColor, borderThickness);
            }
        }


        void PCTSignatures::generateInitPoints(
            std::vector<Point2f>& initPoints,
            const int count,
            const int pointDistribution)
        {
            RNG random;
            random.state = getTickCount();
            initPoints.resize(count);

            switch (pointDistribution)
            {
            case UNIFORM:
                for (int i = 0; i < count; i++)
                {
                    // returns uniformly distributed float random number from [0, 1) range
                    initPoints[i] = (Point2f(random.uniform((float)0.0, (float)1.0), random.uniform((float)0.0, (float)1.0)));
                }
                break;
            case REGULAR:
            {
                int gridSize = (int)ceil(sqrt(count));
                const float step = 1.0f / gridSize;
                const float halfStep = step / 2;
                float x = halfStep;
                float y = halfStep;
                for (int i = 0; i < count; i++)
                {
                    // returns regular grid
                    initPoints[i] = Point2f(x, y);
                    if ((i + 1) % gridSize == 0)
                    {
                        x = halfStep;
                        y += step;
                    }
                    else
                    {
                        x += step;
                    }
                }
                break;
            }
            case NORMAL:
                for (int i = 0; i < count; i++)
                {
                    // returns normally distributed float random number from (0, 1) range with mean 0.5
                    float sigma = 0.2f;
                    float x = (float)random.gaussian(sigma);
                    float y = (float)random.gaussian(sigma);
                    while (x <= -0.5f || x >= 0.5f)
                        x = (float)random.gaussian(sigma);
                    while (y <= -0.5f || y >= 0.5f)
                        y = (float)random.gaussian(sigma);
                    initPoints[i] = Point2f(x, y) + Point2f(0.5, 0.5);
                }
                break;
            default:
                CV_Error(Error::StsNotImplemented, "Generation of this init point distribution is not implemented!");
                break;
            }
        }


    } // end of namespace xfeatures2d
} // end of namespace cv
