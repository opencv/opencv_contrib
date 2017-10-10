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

#ifndef _OPENCV_HFS_SEGMENT_HPP_
#define _OPENCV_HFS_SEGMENT_HPP_
#ifdef __cplusplus

#include "opencv2/core.hpp"

/** @brief hfs Hierarchical Feature Selection for Efficient Image Segmentation

The opencv_hfs module contains an efficient algorithm to segment an image.
This module is written based on the paper Hierarchical Feature Selection for Efficient
Image Segmentation @ECCV 2016. The original project was developed by
Yun Liu(https://github.com/yun-liu/hfs).


Introduction to Hierarchical Feature Selection
----------------------------------------------


This algorithm is executed in 3 stages:

In the first stage, the algorithm uses SLIC (simple linear iterative clustering) algorithm
to obtain the superpixel of the input image.

In the second stage, the algorithm view each superpixel as a node in the graph.
It will calculate a feature vector for each edge of the graph. It then calculates a weight
for each edge based on the feature vector and trained SVM parameters. After obtaining
weight for each edge, it will exploit  EGB (Efficient Graph-based Image Segmentation)
algorithm to merge some nodes in the graph thus obtaining a coarser segmentation
After these operations, a post process will be executed to merge regions that are smaller
then a specific number of pixels into their nearby region.

In the third stage, the algorithm exploits the similar mechanism to further merge
the small regions obtained in the second stage into even coarser segmentation.

After these three stages, we can obtain the final segmentation of the image.
For further details about the algorithm, please refer to the original paper:
Hierarchical Feature Selection for Efficient Image Segmentation @ECCV 2016

*/

namespace cv { namespace hfs {

class CV_EXPORTS HfsSegment : public Algorithm {
public:

/** @brief: set and get the parameter segEgbThresholdI.
* This parameter is used in the second stage mentioned above.
* It is a constant used to threshold weights of the edge when merging
* adjacent nodes when applying EGB algorithm. The segmentation result
* tends to have more regions remained if this value is large and vice versa.
*/
virtual void setSegEgbThresholdI(float c) = 0;
virtual float getSegEgbThresholdI() = 0;


/** @brief: set and get the parameter minRegionSizeI.
* This parameter is used in the second stage
* mentioned above. After the EGB segmentation, regions that have fewer
* pixels then this parameter will be merged into it's adjacent region.
*/
virtual void setMinRegionSizeI(int n) = 0;
virtual int getMinRegionSizeI() = 0;


/** @brief: set and get the parameter segEgbThresholdII.
* This parameter is used in the third stage
* mentioned above. It serves the same purpose as segEgbThresholdI.
* The segmentation result tends to have more regions remained if
* this value is large and vice versa.
*/
virtual void setSegEgbThresholdII(float c) = 0;
virtual float getSegEgbThresholdII() = 0;

/** @brief: set and get the parameter minRegionSizeII.
* This parameter is used in the third stage
* mentioned above. It serves the same purpose as minRegionSizeI
*/
virtual void setMinRegionSizeII(int n) = 0;
virtual int getMinRegionSizeII() = 0;

/** @brief: set and get the parameter spatialWeight.
* This parameter is used in the first stage
* mentioned above(the SLIC stage). It describes how important is the role
* of position when calculating the distance between each pixel and it's region
* center. The exact formula to calculate the distance is
* \f$colorDistance + spatialWeight \times spatialDistance\f$.
* The segmentation result tends to have more local consistency
* if this value is larger.
*/
virtual void setSpatialWeight(float w) = 0;
virtual float getSpatialWeight() = 0;


/** @brief: set and get the parameter slicSpixelSize.
* This parameter is used in the first stage mentioned
* above(the SLIC stage). It describes the size of each
* superpixel when initializing SLIC. Every superpixel
* approximately has slicSpixelSize \times slicSpixelSize
* pixels in the begining.
*/
virtual void setSlicSpixelSize(int n) = 0;
virtual int getSlicSpixelSize() = 0;


/** @brief: set and get the parameter numSlicIter.
* This parameter is used in the first stage. It
* describes how many iterations will be performed when executing SLIC.
*/
virtual void setNumSlicIter(int n) = 0;
virtual int getNumSlicIter() = 0;



/** @brief do segmentation
* @param src: the input image
* @param ifDraw: control whether to draw the image in the returned Mat. 
* If this parameter is false, then the content of the returned Mat is a 
* matrix of index, describing the region each pixel belongs to. And it's 
* data type is CV_16U. If this parameter is true, then the returned Mat is 
* a segmented picture, and color of each region is the average color of 
* all pixels in that region. And it's data type is the same as the input 
* image
*/
virtual Mat performSegment(const Mat& src, bool ifDraw = true) = 0;


/** @brief: create a hfs object
* @param height: the height of the input image
* @param width: the width of the input image
* @param segEgbThresholdI: parameter segEgbThresholdI
* @param minRegionSizeI: parameter minRegionSizeI
* @param segEgbThresholdII: parameter segEgbThresholdII
* @param minRegionSizeII: parameter minRegionSizeII
* @param spatialWeight: parameter spatialWeight
* @param slicSpxielSize: parameter slicSpixelSize
* @param numSlicIter: parameter numSlicIter
*/
static Ptr<HfsSegment> create(int height, int width,
    float segEgbThresholdI = 0.08, int minRegionSizeI = 100,
    float segEgbThresholdII = 0.28f, int minRegionSizeII = 200,
    float spatialWeight = 0.6f, int slicSpixelSize = 8, int numSlicIter = 5);

};

}} // namespace cv { namespace hfs {


#endif
#endif
