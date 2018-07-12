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
//################################################################################
//
//                    Created by Simon Reich
//
//################################################################################

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/epf.hpp>

using namespace cv;

int showWindow(cv::Mat image, std::string title);

int showWindow(cv::Mat image, std::string title)
{
	namedWindow(title);
	imshow(title, image);

	waitKey(0);

	return 0;
}

int main(int argc, char **argv)
{
	// Help text
	if (argc != 2)
	{
		std::cout << "usage: " << argv[0] << " image.jpg" << std::endl;
		std::cout << "image.jpg contains image, which will be smoothed."
		          << std::endl;
		return -1;
	}

	// Load image from first parameter
	std::string filename = argv[1];
	Mat image = imread(filename, 1), res;

	if (!image.data)
	{
		std::cerr << "No image data at " << argv[2] << std::endl;
		throw;
	}

	// Before filtering
	showWindow(image, "Original image");

	// Initialize filter. Kernel size 5x5, threshold 20
	Ptr<epf::edgepreservingFilter> filter =
	    epf::edgepreservingFilter::create(&image, &res, 5, 20);

	// After filtering
	showWindow(res, "Filtered image");

	return 0;
}
