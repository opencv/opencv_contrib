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
    Multimedia tools and applications, 75(13), pp.: 8071�8103, Springer, ISSN: 1380-7501, 2016

    Christian Beecks, Merih Seran Uysal, Thomas Seidl.
    Signature quadratic form distance.
    In Proceedings of the ACM International Conference on Image and Video Retrieval, pages 438-445.
    ACM, 2010.
*/

#include "../precomp.hpp"

#include "pct_clusterizer.hpp"

namespace cv
{
    namespace xfeatures2d
    {
        namespace pct_signatures
        {
            class PCTClusterizer_Impl CV_FINAL : public PCTClusterizer
            {
            public:

                PCTClusterizer_Impl(
                    const std::vector<int>& initSeedIndexes,
                    int iterationCount,
                    int maxClustersCount,
                    int clusterMinSize,
                    float joiningDistance,
                    float dropThreshold,
                    int distanceFunction)
                    : mInitSeedIndexes(initSeedIndexes),
                    mIterationCount(iterationCount),
                    mMaxClustersCount(maxClustersCount),
                    mClusterMinSize(clusterMinSize),
                    mJoiningDistance(joiningDistance),
                    mDropThreshold(dropThreshold),
                    mDistanceFunction(distanceFunction)
                {
                }

                ~PCTClusterizer_Impl()
                {
                }


                int getIterationCount() const CV_OVERRIDE                       { return mIterationCount; }
                std::vector<int> getInitSeedIndexes() const CV_OVERRIDE         { return mInitSeedIndexes; }
                int getMaxClustersCount() const CV_OVERRIDE                     { return mMaxClustersCount; }
                int getClusterMinSize() const CV_OVERRIDE                       { return mClusterMinSize; }
                float getJoiningDistance() const CV_OVERRIDE                    { return mJoiningDistance; }
                float getDropThreshold() const CV_OVERRIDE                      { return mDropThreshold; }
                int getDistanceFunction() const CV_OVERRIDE                     { return mDistanceFunction; }

                void setIterationCount(int iterationCount) CV_OVERRIDE                  { mIterationCount = iterationCount; }
                void setInitSeedIndexes(std::vector<int> initSeedIndexes) CV_OVERRIDE   { mInitSeedIndexes = initSeedIndexes; }
                void setMaxClustersCount(int maxClustersCount) CV_OVERRIDE              { mMaxClustersCount = maxClustersCount; }
                void setClusterMinSize(int clusterMinSize) CV_OVERRIDE                  { mClusterMinSize = clusterMinSize; }
                void setJoiningDistance(float joiningDistance) CV_OVERRIDE              { mJoiningDistance = joiningDistance; }
                void setDropThreshold(float dropThreshold) CV_OVERRIDE                  { mDropThreshold = dropThreshold; }
                void setDistanceFunction(int distanceFunction) CV_OVERRIDE              { mDistanceFunction = distanceFunction; }


                /**
                * @brief K-means algorithm over the sampled points producing centroids as signatures.
                * @param samples List of sampled points.
                * @param signature Output list of computed centroids - the signature of the image.
                */
                void clusterize(InputArray _samples, OutputArray _signature) CV_OVERRIDE
                {
                    CV_Assert(!_samples.empty());

                    // prepare matrices
                    Mat samples = _samples.getMat();

                    if ((int)(this->mInitSeedIndexes.size()) > samples.rows)
                    {
                        CV_Error_(Error::StsBadArg, ("Number of seeds %zu must be less or equal to the number of samples %d.",
                            mInitSeedIndexes.size(), samples.rows));
                    }

                    // Prepare initial centroids.
                    Mat clusters;
                    pickRandomClusters(samples, clusters);
                    clusters(Rect(WEIGHT_IDX, 0, 1, clusters.rows)) = 1;    // set initial weight to 1


                    // prepare for iterating
                    joinCloseClusters(clusters);
                    dropLightPoints(clusters);


                    // Main iterations cycle. Our implementation has fixed number of iterations.
                    for (int iteration = 0; iteration < mIterationCount; iteration++)
                    {
                        // Prepare space for new centroid values.
                        Mat tmpCentroids(clusters.size(), clusters.type());
                        tmpCentroids.setTo(cv::Scalar::all(0));

                        // Clear weights for new iteration.
                        clusters(Rect(WEIGHT_IDX, 0, 1, clusters.rows)).setTo(cv::Scalar::all(0));

                        // Compute affiliation of points and sum new coordinates for centroids.
                        for (int iSample = 0; iSample < samples.rows; iSample++)
                        {
                            int iClosest = findClosestCluster(clusters, samples, iSample);
                            for (int iDimension = 1; iDimension < SIGNATURE_DIMENSION; iDimension++)
                            {
                                tmpCentroids.at<float>(iClosest, iDimension) += samples.at<float>(iSample, iDimension);
                            }
                            clusters.at<float>(iClosest, WEIGHT_IDX)++;
                        }

                        // Compute average from tmp coordinates and throw away too small clusters.
                        int lastIdx = 0;
                        for (int i = 0; (int)i < tmpCentroids.rows; ++i)
                        {
                            // Skip clusters that are too small (large-enough clusters are packed right away)
                            if (clusters.at<float>(i, WEIGHT_IDX) >(iteration + 1) * this->mClusterMinSize)
                            {
                                for (int d = 1; d < SIGNATURE_DIMENSION; d++)
                                {
                                    clusters.at<float>(lastIdx, d) = tmpCentroids.at<float>(i, d) / clusters.at<float>(i, WEIGHT_IDX);
                                }
                                // weights must be compacted as well
                                clusters.at<float>(lastIdx, WEIGHT_IDX) = clusters.at<float>(i, WEIGHT_IDX);
                                lastIdx++;
                            }
                        }

                        // Crop the arrays if some centroids were dropped.
                        clusters.resize(lastIdx);
                        if (clusters.rows == 0)
                        {
                            break;
                        }

                        // Finally join clusters with too close centroids.
                        joinCloseClusters(clusters);
                        dropLightPoints(clusters);
                    }

                    // The result must not be empty!
                    if (clusters.rows == 0)
                    {
                        singleClusterFallback(samples, clusters);
                    }

                    cropClusters(clusters);

                    normalizeWeights(clusters);

                    // save the result
                    _signature.create(clusters.rows, SIGNATURE_DIMENSION, clusters.type());
                    Mat signature = _signature.getMat();
                    clusters.copyTo(signature);
                }


            private:

                /**
                * @brief Join clusters that are closer than joining distance.
                *       If two clusters are joined one of them gets its weight set to 0.
                * @param clusters List of clusters to be scaned and joined.
                */
                void joinCloseClusters(Mat& clusters)
                {
                    for (int i = 0; i < clusters.rows - 1; i++)
                    {
                        if (clusters.at<float>(i, WEIGHT_IDX) == 0)
                        {
                            continue;
                        }

                        for (int j = i + 1; j < clusters.rows; j++)
                        {
                            if (clusters.at<float>(j, WEIGHT_IDX) > 0
                                && computeDistance(mDistanceFunction, clusters, i, clusters, j) <= mJoiningDistance)
                            {
                                clusters.at<float>(i, WEIGHT_IDX) = 0;
                                break;
                            }
                        }
                    }
                }


                /**
                * @brief Remove points whose weight is lesser or equal to given threshold.
                *       The point list is compacted and relative order of points is maintained.
                * @param clusters List of clusters to be scaned and dropped.
                */
                void dropLightPoints(Mat& clusters)
                {
                    int frontIdx = 0;

                    // Skip leading continuous part of weighted-enough points.
                    while (frontIdx < clusters.rows && clusters.at<float>(frontIdx, WEIGHT_IDX) > mDropThreshold)
                    {
                        ++frontIdx;
                    }

                    // Mark first emptied position and advance front index.
                    int tailIdx = frontIdx++;

                    while (frontIdx < clusters.rows)
                    {
                        if (clusters.at<float>(frontIdx, WEIGHT_IDX) > mDropThreshold)
                        {
                            // Current (front) item is not dropped -> copy it to the tail.
                            clusters.row(frontIdx).copyTo(clusters.row(tailIdx));
                            ++tailIdx;  // grow the tail
                        }
                        ++frontIdx;
                    }
                    clusters.resize(tailIdx);
                }


                /**
                * @brief Find closest cluster to selected point.
                * @param clusters List of cluster centroids.
                * @param points List of points.
                * @param pointIdx Index to the list of points (the point for which the closest cluster is being found).
                * @return Index to clusters list pointing at the closest cluster.
                */
                int findClosestCluster(const Mat& clusters, const Mat& points, const int pointIdx) const    // HOT PATH: 35%
                {
                    int iClosest = 0;
                    float minDistance = computeDistance(mDistanceFunction, clusters, 0, points, pointIdx);

                    for (int iCluster = 1; iCluster < clusters.rows; iCluster++)
                    {
                        float distance = computeDistance(mDistanceFunction, clusters, iCluster, points, pointIdx);
                        if (distance < minDistance)
                        {
                            iClosest = iCluster;
                            minDistance = distance;
                        }
                    }
                    return iClosest;
                }


                /**
                * @brief Make sure that the number of clusters does not exceed maxClusters parameter.
                *       If it does, the clusters are sorted by their weights and the smallest clusters
                *       are cropped.
                * @param clusters List of clusters being scanned and cropped.
                * @note This method does nothing iff the number of clusters is within the range.
                */
                void cropClusters(Mat& clusters) const
                {
                    if (clusters.rows > mMaxClustersCount)
                    {
                        Mat duplicate(clusters);                // save original clusters
                        Mat sortedIdx;                          // sort using weight column
                        sortIdx(clusters(Rect(WEIGHT_IDX, 0, 1, clusters.rows)), sortedIdx, SORT_EVERY_COLUMN + SORT_DESCENDING);

                        clusters.resize(mMaxClustersCount);     // crop to max clusters
                        for (int i = 0; i < mMaxClustersCount; i++)
                        {                                       // copy sorted rows
                            duplicate.row(sortedIdx.at<int>(i, 0)).copyTo(clusters.row(i));
                        }
                    }
                }


                /**
                * @brief Fallback procedure invoked when the clustering eradicates all clusters.
                *       Single cluster consisting of all points is then created.
                * @param points List of sampled points.
                * @param clusters Output list of clusters where the single cluster will be written.
                */
                void singleClusterFallback(const Mat& points, Mat& clusters) const
                {
                    if (clusters.rows != 0)
                    {
                        return;
                    }
                    clusters.create(1, points.cols, CV_32FC1);

                    // Sum all points.
                    reduce(points, clusters, 0, REDUCE_SUM, CV_32FC1);

                    // Sum all weights, all points have the same weight -> sum is the point count
                    clusters.at<float>(0, WEIGHT_IDX) = static_cast<float>(points.rows);

                    // Divide centroid by number of points -> compute average in each dimension.
                    clusters /= clusters.at<float>(0, WEIGHT_IDX);
                }


                /**
                * @brief Normalizes cluster weights, so they fall into range [0..1]
                *       The other values are preserved.
                * @param clusters List of clusters which will be normalized.
                * @note Only weight of the clusters will be changed (the first column);
                */
                void normalizeWeights(Mat& clusters)
                {
                    // get max weight
                    float maxWeight = clusters.at<float>(0, WEIGHT_IDX);
                    for (int i = 1; i < clusters.rows; i++)
                    {
                        if (clusters.at<float>(i, WEIGHT_IDX) > maxWeight)
                        {
                            maxWeight = clusters.at<float>(i, WEIGHT_IDX);
                        }
                    }

                    // normalize weight
                    float weightNormalizer = 1 / maxWeight;
                    for (int i = 0; i < clusters.rows; i++)
                    {
                        clusters.at<float>(i, WEIGHT_IDX) = clusters.at<float>(i, WEIGHT_IDX) * weightNormalizer;
                    }
                }


                /**
                * @brief Chooses which sample points will be used as cluster centers,
                *       using list of random indexes that is stored in mInitSeedIndexes.
                * @param samples List of sampled points.
                * @param clusters Output list of random sampled points.
                */
                void pickRandomClusters(Mat& samples, Mat& clusters)
                {
                    clusters.create(0, SIGNATURE_DIMENSION, samples.type());
                    clusters.reserve(mInitSeedIndexes.size());

                    for (int iSeed = 0; iSeed < (int)(mInitSeedIndexes.size()); iSeed++)
                    {
                        clusters.push_back(samples.row(mInitSeedIndexes[iSeed]));
                    }
                }



                /**
                * @brief Indexes of initial clusters from sampling points.
                */
                std::vector<int> mInitSeedIndexes;

                /**
                * @brief Number of iterations performed (the number is fixed).
                */
                int mIterationCount;

                /**
                * @brief Maximal number of clusters. If the number of clusters is exceeded after
                *       all iterations are completed, the smallest clusters are thrown away.
                */
                int mMaxClustersCount;

                /**
                * @brief Minimal size of the cluster. After each (i-th) iteration, the cluster sizes are checked.
                *       Cluster smaller than i*mClusterMinSize are disposed of.
                */
                int mClusterMinSize;

                /**
                * @brief Distance threshold between two centroids. If two centroids become closer
                *       than this distance, the clusters are joined.
                */
                float mJoiningDistance;

                /**
                * @brief Weight threshold of centroids. If weight of a centroid is below this value at the end of the iteration, it is thrown away.
                */
                float mDropThreshold;

                /**
                * @brief L_p metric is used for computing distances.
                */
                int mDistanceFunction;

            };


            Ptr<PCTClusterizer> PCTClusterizer::create(
                const std::vector<int>& initSeedIndexes,
                int iterationCount,
                int maxClustersCount,
                int clusterMinSize,
                float joiningDistance,
                float dropThreshold,
                int distanceFunction)
            {
                return makePtr<PCTClusterizer_Impl>(
                    initSeedIndexes,
                    iterationCount,
                    maxClustersCount,
                    clusterMinSize,
                    joiningDistance,
                    dropThreshold,
                    distanceFunction);
            }
        }
    }
}
