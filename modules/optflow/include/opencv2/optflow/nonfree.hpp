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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
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

#ifndef __OPENCV_OPTFLOW_NONFREE_HPP__
#define __OPENCV_OPTFLOW_NONFREE_HPP__
#include "opencv2/optflow.hpp"

namespace cv
{
namespace optflow
{
    /** Scale maps implementation using scale propagation.
        References:
        M. Tau and T. Hassner, "Dense Correspondences Across Scenes and
        Scales," arXiv preprint arXiv:1406.6323, 24 Jun. 2014. Please see
        project webpage for information on more recent versions:
        http://www.openu.ac.il/home/hassner/scalemaps
    */
    class CV_EXPORTS_W ScaleMap : public Algorithm
    {
    public:
        /** 
            @param exponential Set weight function to either linear (false) or exponential (true).
            linearWeights = 1 + (1 / var) * (weights - mean) * (image(i, j) - mean)
            expWeights = exp(-(weights - image(i, j)) .^ 2 / (0.6 * var))
        */
        static Ptr<ScaleMap> create(bool exponential = false);
 
        /** Executes the scale map algorithm.
            @param image Grayscales floating point image in the range [0 1]
            @param keypoints Sparse features to seed from
            @param scalemap Output the scale for each pixel
        */
        virtual void compute(InputArray image,
            const std::vector<KeyPoint>& keypoints, Mat& scalemap) = 0;
    };

    /** @brief Abstract base class for 2D image feature detectors and descriptor extractors
    */
    class CV_EXPORTS_W SiftImageExtractor : public Algorithm
    {
    public:
        enum
        {
            SCALE_UNIFORM = 100, SCALE_LINEAR = 101, SCALE_EXP = 102
        };
        static Ptr<SiftImageExtractor> create(
            int scale_format = SiftImageExtractor::SCALE_UNIFORM);

        //! Create a sift image from per pixel sift descriptors
        virtual void compute(const Mat& img, Mat& siftImg) = 0;

        virtual void compute(const Mat& img0, const Mat& img1, Mat& siftImg0, Mat& siftImg1) = 0;
    };   

    //! Interface to the SIFT-Flow's algorithm
    CV_EXPORTS_W Ptr<DenseOpticalFlow> createOptFlow_SiftFlow(); 

} // namespace optflow
} // namespace cv

#endif
