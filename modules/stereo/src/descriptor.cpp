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

//function that performs the census transform on two images.
//Two variants of census are offered a sparse version whcih takes every second pixel as well as dense version
void cv::stereo::censusTransform(const cv::Mat &image1, const cv::Mat &image2, int kernelSize, cv::Mat &dist1, cv::Mat &dist2, const int type)
{
    CV_Assert(image1.size() == image2.size());
    CV_Assert(kernelSize % 2 != 0);
    CV_Assert(image1.type() == CV_8UC1 && image2.type() == CV_8UC1);
    CV_Assert(type < 2 && type >= 0);
    CV_Assert(kernelSize <= ((type == 0) ? 5 : 11));
    int n2 = (kernelSize) / 2;
    uint8_t *images[] = {image1.data, image2.data};
    int *costs[] = {(int *)dist1.data,(int *)dist2.data};
    int stride = (int)image1.step;
    if(type == CV_DENSE_CENSUS)
    {
        parallel_for_(cv::Range(n2, image1.rows - n2),
            CombinedDescriptor<1,1,1,2,CensusKernel>(image1.cols, image1.rows,stride,n2,costs,CensusKernel(images, 2),n2));
    }
    else if(type == CV_SPARSE_CENSUS)
    {
        parallel_for_(cv::Range(n2, image1.rows - n2),
            CombinedDescriptor<2,2,1,2,CensusKernel>(image1.cols, image1.rows, stride,n2,costs,CensusKernel(images, 2),n2));
    }
}
//function that performs census on one image
void cv::stereo::censusTransform(const cv::Mat &image1, int kernelSize, cv::Mat &dist1, const int type)
{
    CV_Assert(image1.size() == dist1.size());
    CV_Assert(kernelSize % 2 != 0);
    CV_Assert(image1.type() == CV_8UC1);
    CV_Assert(type < 2 && type >= 0);
    CV_Assert(kernelSize <= ((type == 0) ? 5 : 11));
    int n2 = (kernelSize) / 2;
    uint8_t *images[] = {image1.data};
    int *costs[] = {(int *)dist1.data};
    int stride = (int)image1.step;
    if(type == CV_DENSE_CENSUS)
    {
        parallel_for_(cv::Range(n2, image1.rows - n2),
            CombinedDescriptor<1,1,1,1,CensusKernel>(image1.cols, image1.rows,stride,n2,costs,CensusKernel(images, 1),n2));
    }
    else if(type == CV_SPARSE_CENSUS)
    {
        parallel_for_(cv::Range(n2, image1.rows - n2),
            CombinedDescriptor<2,2,1,1,CensusKernel>(image1.cols, image1.rows,stride,n2,costs,CensusKernel(images, 1),n2));
    }
}
//in a 9x9 kernel only certain positions are choosen for comparison
void cv::stereo::starCensusTransform(const cv::Mat &img1, const cv::Mat &img2, int kernelSize, cv::Mat &dist1, cv::Mat &dist2)
{
    CV_Assert(img1.size() == img2.size());
    CV_Assert(kernelSize % 2 != 0);
    CV_Assert(img1.type() == CV_8UC1 && img2.type() == CV_8UC1);
    CV_Assert(kernelSize >= 7);
    int n2 = (kernelSize) >> 1;
    Mat images[] = {img1, img2};
    int *date[] = { (int *)dist1.data, (int *)dist2.data};
    parallel_for_(cv::Range(n2, img1.rows - n2), StarKernelCensus(images, n2,date,2));
}
//single version of star census
void cv::stereo::starCensusTransform(const cv::Mat &img1, int kernelSize, cv::Mat &dist)
{
    CV_Assert(img1.size() == dist.size());
    CV_Assert(kernelSize % 2 != 0);
    CV_Assert(img1.type() == CV_8UC1);
    CV_Assert(kernelSize >= 7);
    int n2 = (kernelSize) >> 1;
    Mat images[] = {img1};
    int *date[] = { (int *)dist.data};
    parallel_for_(cv::Range(n2, img1.rows - n2), StarKernelCensus(images, n2,date,2));
}
//Modified census transforms
//the first one deals with small illumination changes
//the sencond modified census transform is invariant to noise; i.e.
//if the current pixel with whom we are dooing the comparison is a noise, this descriptor will provide a better result by comparing with the mean of the window
//otherwise if the pixel is not noise the information is strengthend
void cv::stereo::modifiedCensusTransform(const cv::Mat &img1, const cv::Mat &img2, int kernelSize, cv::Mat &dist1,cv::Mat &dist2, const int type, int t, const cv::Mat &IntegralImage1, const cv::Mat &IntegralImage2 )
{
    CV_Assert(img1.size() == img2.size());
    CV_Assert(kernelSize % 2 != 0);
    CV_Assert(img1.type() == CV_8UC1 && img2.type() == CV_8UC1);
    CV_Assert(type < 2 && type >= 0);
    CV_Assert(kernelSize <= 9);
    int n2 = (kernelSize - 1) >> 1;
    uint8_t *images[] = {img1.data, img2.data};
    int *date[] = { (int *)dist1.data, (int *)dist2.data};
    int stride = (int)img1.step;
    if(type == CV_MODIFIED_CENSUS_TRANSFORM)
    {
        //MCT
        parallel_for_(cv::Range(n2, img1.rows - n2),
            CombinedDescriptor<2,4,2, 2,MCTKernel>(img1.cols, img1.rows,stride,n2,date,MCTKernel(images,t,2),n2));
    }
    else if(type == CV_MEAN_VARIATION)
    {
        //MV
        int *integral[] = { (int *)IntegralImage1.data, (int *)IntegralImage2.data };
        parallel_for_(cv::Range(n2, img1.rows - n2),
            CombinedDescriptor<2,3,2,2, MVKernel>(img1.cols, img1.rows,stride,n2,date,MVKernel(images,integral,2),n2));
    }
}
void cv::stereo::modifiedCensusTransform(const cv::Mat &img1, int kernelSize, cv::Mat &dist, const int type, int t , cv::Mat const &IntegralImage)
{
    CV_Assert(img1.size() == dist.size());
    CV_Assert(kernelSize % 2 != 0);
    CV_Assert(img1.type() == CV_8UC1);
    CV_Assert(type < 2 && type >= 0);
    CV_Assert(kernelSize <= 9);
    int n2 = (kernelSize - 1) >> 1;
    uint8_t *images[] = {img1.data};
    int *date[] = { (int *)dist.data};
    int stride = (int)img1.step;
    if(type == CV_MODIFIED_CENSUS_TRANSFORM)
    {
        //MCT
        parallel_for_(cv::Range(n2, img1.rows - n2),
            CombinedDescriptor<2,3,2, 1,MCTKernel>(img1.cols, img1.rows,stride,n2,date,MCTKernel(images,t,1),n2));
    }
    else if(type == CV_MEAN_VARIATION)
    {
        //MV
        int *integral[] = { (int *)IntegralImage.data};
        parallel_for_(cv::Range(n2, img1.rows - n2),
            CombinedDescriptor<2,3,2,1, MVKernel>(img1.cols, img1.rows,stride,n2,date,MVKernel(images,integral,1),n2));
    }
}
//different versions of simetric census
//These variants since they do not compare with the center they are invariant to noise
void cv::stereo::symetricCensusTransform(const cv::Mat &img1, const cv::Mat &img2, int kernelSize, cv::Mat &dist1, cv::Mat &dist2, const int type)
{
    CV_Assert(img1.size() ==  img2.size());
    CV_Assert(kernelSize % 2 != 0);
    CV_Assert(img1.type() == CV_8UC1 && img2.type() == CV_8UC1);
    CV_Assert(type < 2 && type >= 0);
    CV_Assert(kernelSize <= 7);
    int n2 = kernelSize >> 1;
    uint8_t *images[] = {img1.data, img2.data};
    Mat imag[] = {img1, img2};
    int *date[] = { (int *)dist1.data, (int *)dist2.data};
    int stride = (int)img1.step;
    if(type == CV_CS_CENSUS)
    {
        parallel_for_(cv::Range(n2, img1.rows - n2), SymetricCensus(imag, n2,2,date));
    }
    else if(type == CV_MODIFIED_CS_CENSUS)
    {
        parallel_for_(cv::Range(n2, img1.rows - n2),
            CombinedDescriptor<1,1,1,2,ModifiedCsCensus>(img1.cols, img1.rows,stride,n2,date,ModifiedCsCensus(images,n2,2),1));
    }
}
void cv::stereo::symetricCensusTransform(const cv::Mat &img1, int kernelSize, cv::Mat &dist1, const int type)
{
    CV_Assert(img1.size() ==  dist1.size());
    CV_Assert(kernelSize % 2 != 0);
    CV_Assert(img1.type() == CV_8UC1);
    CV_Assert(type < 2 && type >= 0);
    CV_Assert(kernelSize <= 7);
    int n2 = kernelSize >> 1;
    uint8_t *images[] = {img1.data};
    Mat imag[] = {img1};
    int *date[] = { (int *)dist1.data};
    int stride = (int)img1.step;
    if(type == CV_CS_CENSUS)
    {
        parallel_for_(cv::Range(n2, img1.rows - n2), SymetricCensus(imag, n2,1,date));
    }
    else if(type == CV_MODIFIED_CS_CENSUS)
    {
        parallel_for_(cv::Range(n2, img1.rows - n2),
            CombinedDescriptor<1,1,1,1,ModifiedCsCensus>(img1.cols, img1.rows,stride,n2,date,ModifiedCsCensus(images,n2,1),1));
    }
}
//integral image computation used in the Mean Variation Census Transform
void cv::stereo::imageMeanKernelSize(const cv::Mat &image, int windowSize, cv::Mat &cost)
{
    CV_Assert(image.size > 0);
    CV_Assert(cost.size > 0);
    CV_Assert(windowSize % 2 != 0);
    int win = windowSize / 2;
    float scalling = ((float) 1) / (windowSize * windowSize);
    int height = image.rows;
    cost.setTo(0);
    int *c = (int *)cost.data;
    parallel_for_(cv::Range(win + 1, height - win - 1),MeanKernelIntegralImage(image,win,scalling,c));
}
