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
*   The interface contains the main descriptors that will be implemented in the descriptor class			      *
*																												  *
\******************************************************************************************************************/

#include "precomp.hpp"
#include <stdint.h>
#ifndef _OPENCV_DESCRIPTOR_HPP_
#define _OPENCV_DESCRIPTOR_HPP_
#ifdef __cplusplus
#define CV_DENSE_CENSUS 0
#define CV_SPARSE_CENSUS 1
#define CV_MODIFIED_CENTER_SIMETRIC_CENSUS 0
#define CV_CLASSIC_CENTER_SYMETRIC_CENSUS 1
#define CV_STANDARD_MCT 0
#define CV_MCT_MEAN_VARIATION 1
#define CV_SSE_CENSUS 1


namespace cv
{
	namespace stereo
	{
		class Descriptor
		{
		public:

			//Implementation for computing the Census transform on the given image
			void applyCensusOnImage(const cv::Mat image, int kernelSize, cv::Mat &dist, const int type = 0);
			//two variations of census applied on input images
			//Implementation of a census transform which is taking into account just the some pixels from the census kernel thus allowing for larger block sizes
			void applyCensusOnImages(const cv::Mat image1, cv::Mat image2, int kernelSize,cv::Mat &dist, cv::Mat &dist2, const int type = CV_SPARSE_CENSUS);
			// STANDARD_MCT - Modified census which is memorizing for each pixel 2 bits and includes a tolerance to the pixel comparison
			//MCT_MEAN_VARIATION - Implementation of a modified census transform which is also taking into account the variation to the mean of the window not just the center pixel
			void applyMCTOnImages(const cv::Mat image1, const cv::Mat image2, int kernelSize, int t, cv::Mat &dist, cv::Mat &dist2, const int type = CV_STANDARD_MCT);
			//The classical center symetric census
			//A modified version of cs census which is comparing the a pixel with its correspondent from the after the center
			void applySimetricCensus(const cv::Mat image1, const cv::Mat image2,int kernelSize, cv::Mat &dist, cv::Mat &dist2, const int type = CV_CLASSIC_CENTER_SYMETRIC_CENSUS);
			//The brief binary descriptor
			void applyBrifeDescriptor(const cv::Mat image1, const cv::Mat image2,int kernelSize, cv::Mat &dist, cv::Mat &dist2);
			//The classical Rank Transform
			void applyRTDescriptor(const cv::Mat image1, const cv::Mat image2, int kernelSize,  cv::Mat &dist,  cv::Mat &dist2);
			Descriptor();
			~Descriptor(void);
		};
	}
}
#endif
#endif
/*End of file*/
