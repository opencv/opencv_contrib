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
// Copyright (C) 2014, Itseez Inc, all rights reserved.
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
// In no event shall the Itseez Inc or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "opencv2/datasets/track_vot.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <cstdlib>

#include <string>
#include <vector>
#include <set>

using namespace std;
using namespace cv;
using namespace cv::datasets;

int main(int argc, char *argv[])
{
	const char *keys =
		"{ help h usage ? |    | show this message }"
		"{ path p         |true| path to folder with dataset }"
		"{ datasetID id         |1| Dataset ID}";
	CommandLineParser parser(argc, argv, keys);
	string path(parser.get<string>("path"));
	int datasetID(parser.get<int>("datasetID"));
	if (parser.has("help") || path == "true")
	{
		parser.printMessage();
		getchar();
		return -1;
	}

	Ptr<TRACK_vot> dataset = TRACK_vot::create();
	dataset->load(path);
	printf("Datasets number: %d\n", dataset->getDatasetsNum());
	for (int i = 1; i <= dataset->getDatasetsNum(); i++)
		printf("\tDataset #%d size: %d\n", i, dataset->getDatasetLength(i));

	dataset->initDataset(datasetID);

	for (int i = 0; i < dataset->getDatasetLength(datasetID); i++)
	{
		Mat frame;
		if (!dataset->getNextFrame(frame))
			break;
		//Draw Ground Truth BB
		vector <Point2d> gtPoints =  dataset->getGT();
		for (int j = 0; j < (int)(gtPoints.size()-1); j++)
			line(frame, gtPoints[j], gtPoints[j + 1], Scalar(0, 255, 0), 2);
		line(frame, gtPoints[0], gtPoints[(int)(gtPoints.size()-1)], Scalar(0, 255, 0), 2);

		imshow("VOT 2015 DATASET TEST...", frame);
		waitKey(100);
	}

	getchar();
	return 0;
}