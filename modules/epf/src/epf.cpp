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


#include "precomp.hpp"

namespace cv
{
namespace epf
{
using namespace std;

class edgepreservingFilterImpl CV_FINAL : public edgepreservingFilter
{
      public:
	edgepreservingFilterImpl(){};
	~edgepreservingFilterImpl(){};

	edgepreservingFilterImpl(const cv::Mat *src, cv::Mat *dst, int d,
	                         int threshold)
	{

		CV_Assert(src->type() == CV_8UC3 && src->data != dst->data);

		if (d < 3)
			d = 3;
		int subwindowX = d, subwindowY = d;

		if (threshold < 0)
			threshold = 0;

		// number of image channels
		int nChannel = src->channels();

		src->copyTo(*dst);

		vector<double> pixel(nChannel, 0);
		vector<vector<double>> line1(src->rows, pixel);
		vector<vector<vector<double>>> weight(src->cols,
		                                      line1); // global weights
		vector<vector<vector<double>>> imageResult(
		    src->cols, line1); // global normalized image

		// do algorithm
		cv::Mat subwindow;
		for (int posX = 0; posX < src->cols - subwindowX; posX++)
		{
			for (int posY = 0; posY < src->rows - subwindowY;
			     posY++)
			{
				cv::Rect roi = cv::Rect(posX, posY, subwindowX,
				                        subwindowY);
				subwindow = (*src)(roi);
				cv::Mat subwindow1 = (*src)(roi);
				cv::GaussianBlur(subwindow1, subwindow,
				                 cv::Size(5, 5), 0.3, 0.3);

				// compute arithmetic mean of subwindow
				int nCounter = 0;
				vector<int> colorValue(nChannel, 0);

				for (int subPosX = 0; subPosX < subwindow.cols;
				     subPosX++)
				{
					for (int subPosY = 0;
					     subPosY < subwindow.rows;
					     subPosY++)
					{
						cv::Vec3b intensity =
						    subwindow.at<cv::Vec3b>(
						        subPosY, subPosX);
						colorValue[0] +=
						    (int)intensity.val[0];
						colorValue[1] +=
						    (int)intensity.val[1];
						colorValue[2] +=
						    (int)intensity.val[2];
						nCounter++;
					};
				};

				vector<double> ArithmeticMean(nChannel);
				ArithmeticMean[0] =
				    colorValue[0] / nCounter; // B
				ArithmeticMean[1] =
				    colorValue[1] / nCounter; // G
				ArithmeticMean[2] =
				    colorValue[2] / nCounter; // R

				// compute pixelwise distance
				vector<vector<double>> pixelwiseDist;

				for (int subPosX = 0; subPosX < subwindow.cols;
				     subPosX++)
				{
					vector<double> line;
					for (int subPosY = 0;
					     subPosY < subwindow.rows;
					     subPosY++)
					{
						cv::Vec3b intensity =
						    subwindow.at<cv::Vec3b>(
						        subPosY, subPosX);
						double distance =
						    ((double)intensity.val[0] -
						     ArithmeticMean[0]) *
						        ((double)
						             intensity.val[0] -
						         ArithmeticMean[0]) +
						    ((double)intensity.val[1] -
						     ArithmeticMean[1]) *
						        ((double)
						             intensity.val[1] -
						         ArithmeticMean[1]) +
						    ((double)intensity.val[2] -
						     ArithmeticMean[2]) *
						        ((double)
						             intensity.val[2] -
						         ArithmeticMean[2]);
						distance = sqrt(distance);

						line.push_back(distance);
					};

					pixelwiseDist.push_back(line);
				};

				// compute mean pixelwise distance
				double meanPixelwiseDist = 0;

				for (int i = 0; i < (int)pixelwiseDist.size();
				     i++)
					for (int j = 0;
					     j < (int)pixelwiseDist[i].size();
					     j++)
						meanPixelwiseDist +=
						    pixelwiseDist[i][j];

				meanPixelwiseDist /=
				    ((int)pixelwiseDist.size() *
				     (int)pixelwiseDist[0].size());

				// detect edge
				for (int subPosX = 0; subPosX < subwindow.cols;
				     subPosX++)
				{
					for (int subPosY = 0;
					     subPosY < subwindow.rows;
					     subPosY++)
					{
						if (meanPixelwiseDist <=
						        threshold &&
						    pixelwiseDist[subPosX]
						                 [subPosY] <=
						        threshold)
						{
							// global Position
							int globalPosX =
							    posX + subPosX;
							int globalPosY =
							    posY + subPosY;

							// compute global weight
							cv::Vec3b intensity =
							    subwindow
							        .at<cv::Vec3b>(
							            subPosY,
							            subPosX);
							weight[globalPosX]
							      [globalPosY][0] +=
							    intensity.val[0] *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]) *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]);
							weight[globalPosX]
							      [globalPosY][1] +=
							    intensity.val[1] *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]) *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]);
							weight[globalPosX]
							      [globalPosY][2] +=
							    intensity.val[2] *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]) *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]);

							// compute final image
							imageResult[globalPosX]
							           [globalPosY]
							           [0] +=
							    intensity.val[0] *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]) *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]) *
							    ArithmeticMean[0];
							imageResult[globalPosX]
							           [globalPosY]
							           [1] +=
							    intensity.val[1] *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]) *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]) *
							    ArithmeticMean[1];
							imageResult[globalPosX]
							           [globalPosY]
							           [2] +=
							    intensity.val[2] *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]) *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]) *
							    ArithmeticMean[2];
						}
						else if (meanPixelwiseDist <=
						             threshold &&
						         pixelwiseDist
						                 [subPosX]
						                 [subPosY] >
						             threshold)
						{
							// global Position
							int globalPosX =
							    posX + subPosX;
							int globalPosY =
							    posY + subPosY;

							// compute global weight
							cv::Vec3b intensity =
							    subwindow
							        .at<cv::Vec3b>(
							            subPosY,
							            subPosX);
							weight[globalPosX]
							      [globalPosY][0] +=
							    intensity.val[0] *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]) *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]);
							weight[globalPosX]
							      [globalPosY][1] +=
							    intensity.val[1] *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]) *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]);
							weight[globalPosX]
							      [globalPosY][2] +=
							    intensity.val[2] *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]) *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]);

							// compute final image
							imageResult[globalPosX]
							           [globalPosY]
							           [0] +=
							    intensity.val[0] *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]) *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]) *
							    ArithmeticMean[0];
							imageResult[globalPosX]
							           [globalPosY]
							           [1] +=
							    intensity.val[1] *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]) *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]) *
							    ArithmeticMean[1];
							imageResult[globalPosX]
							           [globalPosY]
							           [2] +=
							    intensity.val[2] *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]) *
							    (threshold -
							     pixelwiseDist
							         [subPosX]
							         [subPosY]) *
							    ArithmeticMean[2];
						};
					};
				};
			};
		};

		// compute final image
		for (int globalPosX = 0; globalPosX < (int)imageResult.size();
		     globalPosX++)
		{
			for (int globalPosY = 0;
			     globalPosY < (int)imageResult[globalPosX].size();
			     globalPosY++)
			{
				// cout << "globalPosX: " << globalPosX << "/"
				// << dst->cols << "," << imageResult.size () <<
				// "\tglobalPosY: " << globalPosY << "/" <<
				// dst->rows << "," <<imageResult.at
				// (globalPosX).size () << endl;

				// add image to result
				cv::Vec3b intensity =
				    src->at<cv::Vec3b>(globalPosY, globalPosX);
				imageResult[globalPosX][globalPosY][0] +=
				    (double)intensity.val[0];
				imageResult[globalPosX][globalPosY][1] +=
				    (double)intensity.val[1];
				imageResult[globalPosX][globalPosY][2] +=
				    (double)intensity.val[2];

				// normalize using weight
				imageResult[globalPosX][globalPosY][0] /=
				    (weight[globalPosX][globalPosY][0] + 1);
				imageResult[globalPosX][globalPosY][1] /=
				    (weight[globalPosX][globalPosY][1] + 1);
				imageResult[globalPosX][globalPosY][2] /=
				    (weight[globalPosX][globalPosY][2] + 1);

				// copy to output image frame
				dst->at<cv::Vec3b>(globalPosY, globalPosX)[0] =
				    (uchar)
				        imageResult[globalPosX][globalPosY][0];
				dst->at<cv::Vec3b>(globalPosY, globalPosX)[1] =
				    (uchar)
				        imageResult[globalPosX][globalPosY][1];
				dst->at<cv::Vec3b>(globalPosY, globalPosX)[2] =
				    (uchar)
				        imageResult[globalPosX][globalPosY][2];
			};
		};
	}
};

Ptr<edgepreservingFilter> edgepreservingFilter::create(const cv::Mat *src,
                                                       cv::Mat *dst, int d,
                                                       int threshold)
{
	return Ptr<edgepreservingFilterImpl>(
	    new edgepreservingFilterImpl(src, dst, d, threshold));
}
} // namespace epf
} // namespace cv
