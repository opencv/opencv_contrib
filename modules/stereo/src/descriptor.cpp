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


#include "descriptor.hpp"

using namespace cv;
using namespace stereo;

Descriptor::Descriptor()
{

}
//!Implementation for computing the Census transform on the given image
void Descriptor::applyCensusOnImage(const cv::Mat img, int kernelSize, cv::Mat &dist, const int type)
{
	int n2 = (kernelSize - 1) / 2;
	int height = img.rows;
	int width = img.cols;
	uint8_t * image = img.data;
	int *dst = (int *)dist.data;
//#pragma omp parallel for
	for (int i = n2; i <= height - n2; i++)
	{
		int rWidth = i * width;
		for (int j = n2; j <= width - n2; j++)
		{
			if(type == CV_SSE_CENSUS)
			{
				//to do
			}
			else
			{
				int c = 0;
				for (int ii = i - n2; ii <= i + n2; ii++)
				{
					int rrWidth = ii * width;
					for (int jj = j - n2; jj <= j + n2; jj++)
					{
						if (ii != i || jj != j)
						{
							if (image[(rrWidth + jj)] > image[(rWidth + j)])
							{
								c = c + 1;
							}
							c = c * 2;
						}
					}
				}
				dst[(rWidth + j)] = c;
			}
		}
	}
}
/**
Two variations of census applied on input images
Implementation of a census transform which is taking into account just the some pixels from the census kernel thus allowing for larger block sizes
**/
void Descriptor::applyCensusOnImages(const cv::Mat im1, cv::Mat im2, int kernelSize,cv::Mat &dist, cv::Mat &dist2, const int type)
{
	int n2 = (kernelSize - 1) / 2;
	int width = im1.cols;
	int height = im2.rows;
	uint8_t * image1 = im1.data;
	uint8_t * image2 = im2.data;
	int *dst1 = (int *)dist.data;
	int *dst2 = (int *)dist2.data;
//#pragma omp parallel for
	for (int i =  n2; i <= height - n2; i++)
	{
		int rWidth = i * width;
		int distV = (i) * width;
		for (int j = n2; j <= width - n2; j++)
		{
			int c = 0;
			int c2 = 0;
			if(type == CV_DENSE_CENSUS)
			{
				for (int ii = i - n2; ii <= i + n2; ii++)
				{
					int rrWidth = ii * width;
					for (int jj = j - n2; jj <= j + n2; jj++)
					{
						if (ii != i || jj != j)
						{
							if (image1[rrWidth + jj] > image1[rWidth + j])
							{
								c = c + 1;
							}
							c = c * 2;
						}
						if (ii != i || jj != j)
						{
							if (image2[rrWidth + jj] > image2[rWidth + j])
							{
								c2 = c2 + 1;
							}
							c2 = c2 * 2;
						}
					}
				}
			}
			else if(type == CV_SPARSE_CENSUS)
			{
				for (int ii = i - n2; ii <= i + n2; ii += 2)
				{
					int rrWidth = ii * width;
					for (int jj = j - n2; jj <= j + n2; jj += 2)
					{
						if (ii != i || jj != j)
						{
							if (image1[(rrWidth + jj)] > image1[(rWidth + j)])
							{
								c = c + 1;
							}
							c = c * 2;
						}
						if (ii != i || jj != j)
						{
							if (image2[(rrWidth + jj)] > image2[(rWidth + j)])
							{
								c2 = c2 + 1;
							}
							c2 = c2 * 2;
						}
					}
				}
			}
			dst1[(distV + j)] = c;
			dst2[(distV + j)] = c2;
		}
	}
}
/** 
STANDARD_MCT - Modified census which is memorizing for each pixel 2 bits and includes a tolerance to the pixel comparison
MCT_MEAN_VARIATION - Implementation of a modified census transform which is also taking into account the variation to the mean of the window not just the center pixel
**/
void Descriptor::applyMCTOnImages(const cv::Mat img1, const cv::Mat img2, int kernelSize, int t, cv::Mat &dist, cv::Mat &dist2, const int type)
{
	int n2 = (kernelSize - 1) >> 1;
	int width = img1.cols;
	int height = img1.rows;
	uint8_t *image1 = img1.data;
	uint8_t *image2 = img2.data;
	int *dst1 = (int *)dist.data;
	int *dst2 = (int *)dist2.data;

//	#pragma omp parallel for
	for (int i = n2 + 2; i <= height - n2 - 2; i++)
	{
		int rWidth = i * width;
		int distV = (i) * width;
		for (int j = n2 + 2; j <= width - n2 - 2; j++)
		{
			int c = 0;
			int c2 = 0;
			if(type == CV_STANDARD_MCT)
			{
				for (int ii = i - n2; ii <= i + n2; ii += 2)
				{
					int rrWidth = ii * width;
					for (int jj = j - n2; jj <= j + n2; jj += 2)
					{
						if (ii != i || jj != j)
						{
							if (image1[rrWidth + jj] > image1[rWidth + j] - t)
							{
								c <<= 2;
								c |= 0x3;
							}
							else if (image1[rWidth + j] - t < image1[rrWidth + jj] && image1[rWidth + j] + t >= image1[rrWidth + jj])
							{
								c <<= 2;
								c = c + 1;
							}
							else
							{
								c <<= 2;
							}

						}
						if (ii != i || jj != j)
						{

							if (image2[rrWidth + jj]  > image2[rWidth + j] - t)
							{
								c2 <<= 2;
								c2 |= 0x3;
							}
							else if (image2[rWidth + j] - t < image2[rrWidth + jj] && image2[rWidth + j] + t >= image2[rrWidth + jj])
							{
								c2 <<= 2;
								c2 = c2 + 1;
							}
							else
							{
								c2 <<= 2;
							}
						}
					}

				}
				for (int ii = i - n2; ii <= i + n2; ii += 4)
				{
					int rrWidth = ii * width;
					for (int jj = j - n2; jj <= j + n2; jj += 4)
					{
						if (ii != i || jj != j)
						{
							if (image1[rrWidth + jj]  > image1[rWidth + j] - t)
							{
								c <<= 2;
								c |= 0x3;
							}
							else if (image1[rWidth + j] - t < image1[rrWidth + jj] && image1[rWidth + j] + t >= image1[rrWidth + jj])
							{
								c <<= 2;
								c += 1;
							}
							else
							{
								c <<= 2;
							}

						}
						if (ii != i || jj != j)
						{
							if (image2[rrWidth + jj]  > image2[rWidth + j] - t)
							{
								c2 <<= 2;
								c2 |= 0x3;
							}
							else if (image2[rWidth + j] - t < image2[rrWidth + jj] && image2[rWidth + j] + t >= image2[rrWidth + jj])
							{
								c2 <<= 2;
								c2 = c2 + 1;
							}
							else
							{
								c2 <<= 2;
							}
						}
					}
				}

			}
			else if(type == CV_MCT_MEAN_VARIATION)
			{
				//to do mean variation
			}
			dst1[distV + j] = c;
			dst2[distV + j] = c2;
		}
	}
}
/**The classical center symetric census
A modified version of cs census which is comparing a pixel with its correspondent after the center
**/
void Descriptor::applySimetricCensus(const cv::Mat img1, const cv::Mat img2,int kernelSize, cv::Mat &dist, cv::Mat &dist2, const int type)
{
	int n2 = (kernelSize - 1) / 2;
	int height = img1.rows;
	int width = img1.cols;
	uint8_t *image1 = img1.data;
	uint8_t *image2 = img2.data;
	int *dst1 = (int *)dist.data;
	int *dst2 = (int *)dist2.data;
	//#pragma omp parallel for
	for (int i = + n2; i <= height - n2; i++)
	{
		int distV = (i) * width;
		for (int j = n2; j <= width - n2; j++)
		{
			int c = 0;
			int c2 = 0;

			if(type == CV_CLASSIC_CENTER_SYMETRIC_CENSUS)
			{
				for (int ii = -n2; ii < 0; ii++)
				{
					int rrWidth = (ii + i) * width;
					for (int jj = -n2; jj <= +n2; jj++)
					{
						if (ii != i || jj != j)
						{
							if (image1[(rrWidth + (jj + j))] > image1[((ii * (-1) + i) * width + (-1 * jj) + j)])
							{
								c = c + 1;
							}
							c = c * 2;
						}
						if (ii != i || jj != j)
						{
							if (image2[(rrWidth + (jj + j))] > image2[((ii * (-1) + i) * width + (-1 * jj) + j)])
							{
								c2 = c2 + 1;
							}
							c2 = c2 * 2;
						}
					}
				}
				for (int jj = -n2; jj < 0; jj++)
				{
					if (image1[(i * width + (jj + j))] > image1[(i * width + (-1 * jj) + j)])
					{
						c = c + 1;
					}
					c = c * 2;
					if (image2[(i * width + (jj + j))] > image2[(i * width + (-1 * jj) + j)])
					{
						c2 = c2 + 1;
					}
					c2 = c2 * 2;
				}
			}
			else if(type == CV_MODIFIED_CENTER_SIMETRIC_CENSUS)
			{
				for (int ii = i - n2; ii <= i + 1; ii++)
				{
					int rrWidth = ii * width;
					int rrWidthC = (ii + n2) * width;
					for (int jj = j - n2; jj <= j + n2; jj += 2)
					{
						if (ii != i || jj != j)
						{
							if (image1[(rrWidth + jj)] > image1[(rrWidthC + (jj + n2))])
							{
								c = c + 1;
							}
							c = c * 2;
						}
						if (ii != i || jj != j)
						{
							if (image2[(rrWidth + jj)] > image2[(rrWidthC + (jj + n2))])
							{
								c2 = c2 + 1;
							}
							c2 = c2 * 2;
						}
					}
				}
			}
			dst1[(distV + j)] = c;
			dst2[(distV + j)] = c2;
		}
	}

}
//!brief binary descriptor used in stereo correspondence
void Descriptor::applyBrifeDescriptor(const cv::Mat image1, const cv::Mat image2,int kernelSize, cv::Mat &dist, cv::Mat &dist2)
{
	//TO DO
	//marked the variables in order to avoid warnings
	(void) image1;
	(void) image2;
	(void) dist;
	(void) dist2;
	(void) kernelSize;
}
//The classical Rank Transform
void  Descriptor::applyRTDescriptor(const cv::Mat image1, const cv::Mat image2, int kernelSize,  cv::Mat &dist,  cv::Mat &dist2)
{
	//TO DO
	//marked the variables in order to avoid warnings
	(void) image1;
	(void) image2;
	(void) dist;
	(void) dist2;
	(void) kernelSize;
}

Descriptor::~Descriptor(void)
{
}
