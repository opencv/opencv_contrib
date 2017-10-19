/*M///////////////////////////////////////////////////////////////////////////////////////
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
//                              License Agreement
//                    For Open Source Computer Vision Library
//                           (3 - clause BSD License)
//
// Copyright(C) 2000 - 2016, Intel Corporation, all rights reserved.
// Copyright(C) 2009 - 2011, Willow Garage Inc., all rights reserved.
// Copyright(C) 2009 - 2016, NVIDIA Corporation, all rights reserved.
// Copyright(C) 2010 - 2013, Advanced Micro Devices, Inc., all rights reserved.
// Copyright(C) 2015 - 2016, OpenCV Foundation, all rights reserved.
// Copyright(C) 2015 - 2016, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met :
//
//      * Redistributions of source code must retain the above copyright notice,
//        this list of conditions and the following disclaimer.
//
//      * Redistributions in binary form must reproduce the above copyright notice,
//        this list of conditions and the following disclaimer in the documentation
//        and / or other materials provided with the distribution.
//
//      * Neither the names of the copyright holders nor the names of the contributors
//        may be used to endorse or promote products derived from this software
//        without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort(including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef _OPENCV_HFS_CORE_HPP_
#define _OPENCV_HFS_CORE_HPP_
#ifdef __cplusplus

#include "opencv2/core.hpp"

#include "magnitude.hpp"
#include "merge/merge.hpp"

#include "or_utils/or_types.hpp"
#include "gslic/gslic.hpp"

#define DOUBLE_EPS 1E-6

namespace cv { namespace hfs {


struct HfsSettings
{
    float egbThresholdI;
    int minRegionSizeI;
    float egbThresholdII;
    int minRegionSizeII;
    cv::hfs::gslic::GslicSettings slicSettings;
};

class HfsCore
{
public:
    HfsCore(int height, int width, 
        float segThresholdI, int minRegionSizeI, 
        float segThresholdII, int minRegionSizeII,
        float spatialWeight, int spixelSize, int numIter);
    ~HfsCore();

    void loadImage( const cv::Mat& inimg, Ptr<UChar4Image> outimg );
    cv::Mat getSLICIdx( const cv::Mat& img3u, int &num_css );

    inline float getEulerDistance( cv::Vec3f in1, cv::Vec3f in2 ) 
    {
        cv::Vec3f diff = in1 - in2;
        return sqrt(diff.dot(diff));
    }

    cv::Vec4f getColorFeature( const cv::Vec3f& in1, const cv::Vec3f& in2 );
    int getAvgGradientBdry( const cv::Mat& idx_mat, 
        const std::vector<cv::Mat> &mag1u, int num_css, cv::Mat &bd_num, 
        std::vector<cv::Mat> &gradients );

    void getSegmentationI( const cv::Mat& lab3u, 
        const cv::Mat& mag1u, const cv::Mat& idx_mat, 
        float c, int min_size, cv::Mat& seg, int& num_css);
    void getSegmentationII(
        const cv::Mat& lab3u, const cv::Mat& mag1u, const cv::Mat& idx_mat, 
        float c, int min_size, cv::Mat& seg, int &num_css );
    void drawSegmentationRes( const cv::Mat& seg, const cv::Mat& img3u, 
                              int num_css, cv::Mat& show );

    int processImage( const cv::Mat& img3u, cv::Mat& seg );
    void constructEngine();
    void reconstructEngine();

public:
    HfsSettings hfsSettings;

private:
    std::vector<float> w1, w2;
    std::vector<cv::Mat> wMat;

    Ptr<Magnitude> mag_engine;

    cv::Ptr<UChar4Image> in_img, out_img;
    cv::Ptr<gslic::engines::CoreEngine> gslic_engine;
};


}}

#endif
#endif
