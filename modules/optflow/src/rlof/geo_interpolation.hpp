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
#ifndef _GEO_INTERPOLATION_HPP_
#define _GEO_INTERPOLATION_HPP_
#include <opencv2/core.hpp>
namespace cv
{
	namespace optflow
	{
		typedef Vec<float, 8> Vec8f;
		Mat getGraph(const Mat & image, float edge_length);
		Mat sgeo_dist(const Mat& gra, int y, int x, float max, Mat &prev);
		Mat sgeo_dist(const Mat& gra, const std::vector<Point2f> & points, float max, Mat &prev);
		Mat interpolate_irregular_nw(const Mat &in, const Mat &mask, const Mat &color_img, float max_d, float bandwidth, float pixeldistance);
		Mat interpolate_irregular_nn(
			const std::vector<Point2f> & prevPoints,
			const std::vector<Point2f> & nextPoints,
			const std::vector<uchar> & status,
			const Mat &color_img,
			float pixeldistance);
		Mat interpolate_irregular_knn(
			const std::vector<Point2f> & _prevPoints,
			const std::vector<Point2f> & _nextPoints,
			const std::vector<uchar> & status,
			const Mat &color_img,
			int k,
			float pixeldistance);

		Mat interpolate_irregular_nn_raster(const std::vector<Point2f> & prevPoints,
			const std::vector<Point2f> & nextPoints,
			const std::vector<uchar> & status,
			const Mat & i1);

	}
}
#endif
