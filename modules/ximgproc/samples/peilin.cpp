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
// Copyright (C) 2008, Willow Garage Inc., all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>

static inline cv::Mat operator& ( const cv::Mat& lhs, const cv::Matx23d& rhs )
{
	cv::Mat ret;
	cv::warpAffine ( lhs, ret, rhs, lhs.size(), cv::INTER_LINEAR );
	return ret;
}

static inline cv::Mat operator& ( const cv::Matx23d& lhs, const cv::Mat& rhs )
{
	cv::Mat ret;
	cv::warpAffine ( rhs, ret, lhs, rhs.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP );
	return ret;
}

int main()
{
	cv::Mat I = cv::imread ( "../data/peilin_plane.png", 0 );
	cv::Mat N = I & cv::PeiLinNormalization ( I );
	cv::Mat J = cv::imread ( "../data/peilin_shape.png", 0 );
	cv::Mat D = cv::PeiLinNormalization ( J ) & I;
	cv::imshow ( "I", I );
	cv::imshow ( "N", N );
	cv::imshow ( "J", J );
	cv::imshow ( "D", D );
	cv::waitKey();
	return 0;
}

