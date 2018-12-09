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
#ifndef _RLOF_LOCALFLOW_H_
#define _RLOF_LOCALFLOW_H_
#include <limits>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/optflow/rlofflow.hpp"
//! Fast median estimation method based on @cite Tibshirani2008. This implementation relates to http://www.stat.cmu.edu/~ryantibs/median/
using namespace cv;
template<typename T>
T quickselect(const Mat & inp, int k)
{
	unsigned long i;
	unsigned long ir;
	unsigned long j;
	unsigned long l;
	unsigned long mid;
	Mat values = inp.clone();
	T a;

	l = 0;
	ir = MAX(values.rows, values.cols) - 1;
	while (true) 
	{
		if (ir <= l + 1) 
		{
			if (ir == l + 1 && values.at<T>(ir) < values.at<T>(l))
				std::swap(values.at<T>(l), values.at<T>(ir));
			return values.at<T>(k);
		}
		else 
		{
			mid = (l + ir) >> 1;
			std::swap(values.at<T>(mid), values.at<T>(l+1));
			if (values.at<T>(l) > values.at<T>(ir))
				std::swap(values.at<T>(l), values.at<T>(ir));
			if (values.at<T>(l+1) > values.at<T>(ir))
				std::swap(values.at<T>(l+1), values.at<T>(ir));
			if (values.at<T>(l) > values.at<T>(l+1))
				std::swap(values.at<T>(l), values.at<T>(l+1));
			i = l + 1;
			j = ir;
			a = values.at<T>(l+1);
			while (true) 
			{
				do
				{
					i++;
				}
				while (values.at<T>(i) < a);
				do
				{
					j--;
				}
				while (values.at<T>(j) > a);
				if (j < i) break;
				std::swap(values.at<T>(i), values.at<T>(j));
			}
			values.at<T>(l+1) = values.at<T>(j);
			values.at<T>(j) = a;
			if (j >= static_cast<unsigned long>(k)) ir = j - 1;
			if (j <= static_cast<unsigned long>(k)) l = i;
		}
	}
}

namespace cv
{
	namespace optflow
	{
	class CImageBuffer
	{
	public:
		CImageBuffer()
			: m_Overwrite(true)
		{};
		void setGrayFromRGB(const cv::Mat & inp)
		{
			if(m_Overwrite)
				cv::cvtColor(inp, m_Image, cv::COLOR_BGR2GRAY);
		}
		void setImage(const cv::Mat & inp)
		{
			if(m_Overwrite)
				inp.copyTo(m_Image);
		}
		void setBlurFromRGB(const cv::Mat & inp)
		{
			//cv::medianBlur(constNextImage, blurNextImg, 7);
			if(m_Overwrite)
				cv::GaussianBlur(inp, m_BlurredImage, cv::Size(7,7), -1);
		}
		
		int buildPyramid(cv::Size winSize, int maxLevel, float levelScale[2]);
		cv::Mat & getImage(int level) {return m_ImagePyramid[level];}

		std::vector<cv::Mat>		m_ImagePyramid;
		cv::Mat						m_BlurredImage;
		cv::Mat						m_Image;
		std::vector<cv::Mat>		m_CrossPyramid;
		int							m_maxLevel;
		bool m_Overwrite;
	};
		
	void calcLocalOpticalFlow(
		const Mat prevImage,
		const Mat currImage,
		Ptr<CImageBuffer>  prevPyramids[2],
		Ptr<CImageBuffer>  currPyramids[2],
		const std::vector<Point2f> & prevPoints,
		std::vector<Point2f> & currPoints,
		const RLOFOpticalFlowParameter & param);
}
}
#endif