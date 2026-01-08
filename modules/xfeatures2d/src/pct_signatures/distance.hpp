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


#ifndef _OPENCV_XFEATURES_2D_PCT_SIGNATURES_DISTANCE_HPP_
#define _OPENCV_XFEATURES_2D_PCT_SIGNATURES_DISTANCE_HPP_

#ifdef __cplusplus
#include "constants.hpp"

namespace cv
{
    namespace xfeatures2d
    {
        namespace pct_signatures
        {

            static inline float distanceL0_25(
                const Mat& points1, int idx1,
                const Mat& points2, int idx2)
            {
                float result = (float)0.0;
                for (int d = 1; d < SIGNATURE_DIMENSION; ++d)
                {
                    float difference = points1.at<float>(idx1, d) - points2.at<float>(idx2, d);
                    result += std::sqrt(std::sqrt(std::abs(difference)));
                }
                result *= result;
                return result * result;
            }


            static inline float distanceL0_5(
                const Mat& points1, int idx1,
                const Mat& points2, int idx2)
            {
                float result = (float)0.0;
                for (int d = 1; d < SIGNATURE_DIMENSION; ++d)
                {
                    float difference = points1.at<float>(idx1, d) - points2.at<float>(idx2, d);
                    result += std::sqrt(std::abs(difference));
                }
                return result * result;
            }


            static inline float distanceL1(
                const Mat& points1, int idx1,
                const Mat& points2, int idx2)
            {
                float result = (float)0.0;
                for (int d = 1; d < SIGNATURE_DIMENSION; ++d)
                {
                    float difference = points1.at<float>(idx1, d) - points2.at<float>(idx2, d);
                    result += std::abs(difference);
                }
                return result;
            }


            static inline float distanceL2(
                const Mat& points1, int idx1,
                const Mat& points2, int idx2)
            {
                float result = (float)0.0;
                for (int d = 1; d < SIGNATURE_DIMENSION; ++d)
                {
                    float difference = points1.at<float>(idx1, d) - points2.at<float>(idx2, d);
                    result += difference * difference;
                }
                return (float)std::sqrt(result);
            }


            static inline float distanceL2Squared(
                const Mat& points1, int idx1,
                const Mat& points2, int idx2)
            {
                float result = (float)0.0;
                for (int d = 1; d < SIGNATURE_DIMENSION; ++d)
                {
                    float difference = points1.at<float>(idx1, d) - points2.at<float>(idx2, d);
                    result += difference * difference;
                }
                return result;
            }


            static inline float distanceL5(
                const Mat& points1, int idx1,
                const Mat& points2, int idx2)
            {
                float result = (float)0.0;
                for (int d = 1; d < SIGNATURE_DIMENSION; ++d)
                {
                    float difference = points1.at<float>(idx1, d) - points2.at<float>(idx2, d);
                    result += std::abs(difference) * difference * difference * difference * difference;
                }
                return std::pow(result, (float)0.2);
            }


            static inline float distanceLInfinity(
                const Mat& points1, int idx1,
                const Mat& points2, int idx2)
            {
                float result = (float)0.0;
                for (int d = 1; d < SIGNATURE_DIMENSION; ++d)
                {
                    float difference = points1.at<float>(idx1, d) - points2.at<float>(idx2, d);
                    if (difference > result)
                    {
                        result = difference;
                    }
                }
                return result;
            }


            /**
            * @brief Computed distance between two centroids using given distance function.
            * @param distanceFunction Distance function selector.
            * @param points1 The first signature matrix - one centroid in each row.
            * @param idx1 ID of centroid in the first signature
            * @param points2 The second signature matrix - one centroid in each row.
            * @param idx2 ID of centroid in the first signature
            * @note The first column of a signature contains weights,
            *       so only rows 1 to SIGNATURE_DIMENSION are used.
            */
            static inline float computeDistance(
                const int distanceFunction,
                const Mat& points1, int idx1,
                const Mat& points2, int idx2)
            {
                switch (distanceFunction)
                {
                case PCTSignatures::L0_25:
                    return distanceL0_25(points1, idx1, points2, idx2);
                case PCTSignatures::L0_5:
                    return distanceL0_5(points1, idx1, points2, idx2);
                case PCTSignatures::L1:
                    return distanceL1(points1, idx1, points2, idx2);
                case PCTSignatures::L2:
                    return distanceL2(points1, idx1, points2, idx2);
                case PCTSignatures::L2SQUARED:
                    return distanceL2Squared(points1, idx1, points2, idx2);
                case PCTSignatures::L5:
                    return distanceL5(points1, idx1, points2, idx2);
                case PCTSignatures::L_INFINITY:
                    return distanceLInfinity(points1, idx1, points2, idx2);
                }
                CV_Error(Error::StsBadArg, "Distance function not implemented!");
            }
        }
    }
}


#endif

#endif
