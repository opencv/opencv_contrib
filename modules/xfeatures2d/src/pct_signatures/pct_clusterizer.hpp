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


#ifndef _OPENCV_XFEATURES_2D_PCT_SIGNATURES_CLUSTERIZER_HPP_
#define _OPENCV_XFEATURES_2D_PCT_SIGNATURES_CLUSTERIZER_HPP_

#ifdef __cplusplus


#include "constants.hpp"
#include "distance.hpp"


namespace cv
{
    namespace xfeatures2d
    {
        namespace pct_signatures
        {
            class PCTClusterizer : public Algorithm
            {
            public:

                static Ptr<PCTClusterizer> create(
                    const std::vector<int>& initSeedIndexes,
                    int iterations = 10,
                    int maxClusters = 768,  // max for Fermi GPU architecture
                    int clusterMinSize = 2,
                    float joiningDistance = 0.2,
                    float dropThreshold = 0,
                    int distanceFunction = PCTSignatures::L2);


                /**** Accessors ****/

                virtual int getIterationCount() const = 0;
                virtual std::vector<int> getInitSeedIndexes() const = 0;
                virtual int getMaxClustersCount() const = 0;
                virtual int getClusterMinSize() const = 0;
                virtual float getJoiningDistance() const = 0;
                virtual float getDropThreshold() const = 0;
                virtual int getDistanceFunction() const = 0;


                virtual void setIterationCount(int iterationCount) = 0;
                virtual void setInitSeedIndexes(std::vector<int> initSeedIndexes) = 0;
                virtual void setMaxClustersCount(int maxClustersCount) = 0;
                virtual void setClusterMinSize(int clusterMinSize) = 0;
                virtual void setJoiningDistance(float joiningDistance) = 0;
                virtual void setDropThreshold(float dropThreshold) = 0;
                virtual void setDistanceFunction(int distanceFunction) = 0;

                /**
                * @brief K-means algorithm over the sampled points producing centroids as signatures.
                * @param samples List of sampled points.
                * @param signature Output list of computed centroids - the signature of the image.
                */
                virtual void clusterize(InputArray samples, OutputArray signature) = 0;
            };
        }
    }
}


#endif

#endif
