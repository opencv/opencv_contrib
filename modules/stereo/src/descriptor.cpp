//By downloading, copying, installing or using the software you agree to this license.
//If you do not agree to this license, do not download, install,
//copy or use the software.
//
//
//                          License Agreement
//               For Open Source Computer Vision Library
//                       (3-clause BSD License)
//
//Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
//Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
//Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
//Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
//Copyright (C) 2015, OpenCV Foundation, all rights reserved.
//Copyright (C) 2015, Itseez Inc., all rights reserved.
//Third party copyrights are property of their respective owners.
//
//Redistribution and use in source and binary forms, with or without modification,
//are permitted provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
//  * Neither the names of the copyright holders nor the names of the contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
//This software is provided by the copyright holders and contributors "as is" and
//any express or implied warranties, including, but not limited to, the implied
//warranties of merchantability and fitness for a particular purpose are disclaimed.
//In no event shall copyright holders or contributors be liable for any direct,
//indirect, incidental, special, exemplary, or consequential damages
//(including, but not limited to, procurement of substitute goods or services;
//loss of use, data, or profits; or business interruption) however caused
//and on any theory of liability, whether in contract, strict liability,
//or tort (including negligence or otherwise) arising in any way out of
//the use of this software, even if advised of the possibility of such damage.

/*****************************************************************************************************************\
*   The file contains the implemented descriptors                                                                  *
\******************************************************************************************************************/
#include "descriptor.hpp"

using namespace cv;
using namespace stereo;

void cv::stereo::applyCensusOnImage(const cv::Mat &img, int kernelSize, cv::Mat &dist, const int type)
{
    CV_Assert(img.type() == CV_8UC1);
    CV_Assert(kernelSize <= 5);
    CV_Assert(type < 2 && type >= 0);
    int n2 = (kernelSize - 1) / 2;
    parallel_for_(cv::Range(n2, img.rows - n2), singleImageCensus(img.data, img.cols, img.rows, n2, (int *)dist.data, type));
}
void cv::stereo::applyCensusOnImages(const cv::Mat &im1,const cv::Mat &im2, int kernelSize, cv::Mat &dist, cv::Mat &dist2, const int type)
{
    CV_Assert(im1.size() == im2.size());
    CV_Assert(im1.type() == CV_8UC1 && im2.type() == CV_8UC1);
    CV_Assert(type < 2 && type >= 0);
    CV_Assert(kernelSize <= (type == 0 ? 5 : 10));
    int n2 = (kernelSize - 1) / 2;
    if(type == Dense_Census)
    {
        parallel_for_(cv::Range(n2, im1.rows - n2),
            CombinedDescriptor<1,1,1,CensusKernel>(im1.cols, im1.rows,n2,(int *)dist.data,(int *)dist2.data,CensusKernel(im1.data, im2.data),n2));
    }
    else if(type == Sparse_Census)
    {
        parallel_for_(cv::Range(n2, im1.rows - n2),
            CombinedDescriptor<2,2,1,CensusKernel>(im1.cols, im1.rows,n2,(int *)dist.data,(int *)dist2.data,CensusKernel(im1.data, im2.data),n2));
    }

}
void cv::stereo::applyMCTOnImages(const cv::Mat &img1, const cv::Mat &img2, int kernelSize, int t, cv::Mat &dist, cv::Mat &dist2, const int type)
{
    CV_Assert(img1.size() == img2.size());
    CV_Assert(img1.type() == CV_8UC1 && img2.type() == CV_8UC1);
    CV_Assert(type < 2 && type >= 0);
    CV_Assert(kernelSize <= 9);
    int n2 = (kernelSize - 1) >> 1;
    if(type == StandardMct)
    {
        parallel_for_(cv::Range(n2, img1.rows - n2),
            CombinedDescriptor<2,3,2,MCTKernel>(img1.cols, img1.rows,n2,(int *)dist.data,(int *)dist2.data,MCTKernel(img1.data, img2.data,t),n2));
    }
    else
    {
        //MV
    }
}
void cv::stereo::applySimetricCensus(const cv::Mat &img1, const cv::Mat &img2, int kernelSize, cv::Mat &dist, cv::Mat &dist2, const int type)
{
    CV_Assert(img1.size() ==  img2.size());
    CV_Assert(img1.type() == CV_8UC1 && img2.type() == CV_8UC1);
    CV_Assert(type < 2 && type >= 0);
    CV_Assert(kernelSize <= 7);
    int n2 = (kernelSize - 1) >> 1;
    if(type == ClassicCenterSymetricCensus)
    {
        parallel_for_(cv::Range(n2, img1.rows - n2), parallelSymetricCensus(img1.data, img2.data, img1.cols, img2.rows, n2, (int *)dist.data, (int *)dist2.data, type));
    }
    else if(type == ModifiedCenterSymetricCensus)
    {
        parallel_for_(cv::Range(n2, img1.rows - n2),
            CombinedDescriptor<1,1,1,ModifiedCsCensus>(img1.cols, img1.rows,n2,(int *)dist.data,(int *)dist2.data,ModifiedCsCensus(img1.data, img2.data,n2),1));
    }
}
void cv::stereo::applyBrifeDescriptor(const cv::Mat &image1, const cv::Mat &image2, int kernelSize, cv::Mat &dist, cv::Mat &dist2)
{
    //TO DO
    //marked the variables in order to avoid warnings
    (void)image1;
    (void)image2;
    (void)dist;
    (void)dist2;
    (void)kernelSize;
}
void  cv::stereo::applyRTDescriptor(const cv::Mat &image1, const cv::Mat &image2, int kernelSize, cv::Mat &dist, cv::Mat &dist2)
{
    //TO DO
    //marked the variables in order to avoid warnings
    (void)image1;
    (void)image2;
    (void)dist;
    (void)dist2;
    (void)kernelSize;
}
