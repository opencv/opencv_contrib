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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#include "opencv2/datasets/track_vot.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::datasets;

#define NUM_TEST_FRAMES 1000

static Mat image;
static bool paused;
static bool selectObjects = false;
static bool startSelection = false;
vector<Rect2d> boundingBoxes;
int targetsCnt = 0;
int targetsNum = 0;
Rect2d boundingBox;

static const char* keys =
{ "{@tracker_algorithm | | Tracker algorithm }"
"{@target_num     |1| Number of targets }"
"{@dataset_path     |true| Dataset path     }"
"{@dataset_id     |1| Dataset ID     }"
};

static void onMouse(int event, int x, int y, int, void*)
{
	if (!selectObjects)
	{
		switch (event)
		{
		case EVENT_LBUTTONDOWN:
			//set origin of the bounding box
			startSelection = true;
			boundingBox.x = x;
			boundingBox.y = y;
			boundingBox.width = boundingBox.height = 0;
			break;
		case EVENT_LBUTTONUP:
			//sei with and height of the bounding box
			boundingBox.width = std::abs(x - boundingBox.x);
			boundingBox.height = std::abs(y - boundingBox.y);
			boundingBoxes.push_back(boundingBox);
			targetsCnt++;
			if (targetsCnt == targetsNum)
			{
				paused = false;
				selectObjects = true;
			}
			startSelection = false;
			break;
		case EVENT_MOUSEMOVE:

			if (startSelection && !selectObjects)
			{
				//draw the bounding box
				Mat currentFrame;
				image.copyTo(currentFrame);
				for (int i = 0; i < (int)boundingBoxes.size(); i++)
					rectangle(currentFrame, boundingBoxes[i], Scalar(255, 0, 0), 2, 1);
				rectangle(currentFrame, Point((int)boundingBox.x, (int)boundingBox.y), Point(x, y), Scalar(255, 0, 0), 2, 1);
				imshow("Tracking API", currentFrame);
			}
			break;
		}
	}
}

static void help()
{
	cout << "\nThis example shows the functionality of \"Long-term optical tracking API\""
		"TLD dataset ID: 1~10, VOT2015 dataset ID: 1~60\n"
		"-- pause video [p] and draw a bounding boxes around the targets to start the tracker\n"
		"Example:\n"
		"./example_tracking_multiTracker_dataset<tracker_algorithm> <number_of_targets> <dataset_path> <dataset_id>\n"
		<< endl;

	cout << "\n\nHot keys: \n"
		"\tq - quit the program\n"
		"\tp - pause video\n";
}

int main(int argc, char *argv[])
{
	CommandLineParser parser(argc, argv, keys);
	string tracker_algorithm = parser.get<string>(0);
	targetsNum = parser.get<int>(1);
	string datasetRootPath = parser.get<string>(2);
	int datasetID = parser.get<int>(3);
	if (tracker_algorithm.empty() || datasetRootPath.empty() || targetsNum < 1)
	{
		help();
		return -1;
	}

	Mat frame;
	paused = false;
	namedWindow("Tracking API", 0);
	setMouseCallback("Tracking API", onMouse, 0);

	MultiTrackerTLD mt;
	//Init Dataset
	Ptr<TRACK_vot> dataset = TRACK_vot::create();
	dataset->load(datasetRootPath);
	dataset->initDataset(datasetID);

	//Read first frame
	dataset->getNextFrame(frame);
	frame.copyTo(image);
	for (int i = 0; i < (int)boundingBoxes.size(); i++)
		rectangle(image, boundingBoxes[i], Scalar(255, 0, 0), 2, 1);
	imshow("Tracking API", image);

	bool initialized = false;
	paused = true;
	int frameCounter = 0;

	//Time measurment
	int64 e3 = getTickCount();

	for (;;)
	{
		if (!paused)
		{
			//Time measurment
			int64 e1 = getTickCount();
			if (initialized){
				if (!dataset->getNextFrame(frame))
					break;
				frame.copyTo(image);
			}

			if (!initialized && selectObjects)
			{
				//Initialize the tracker and add targets
				for (int i = 0; i < (int)boundingBoxes.size(); i++)
				{
					if (!mt.addTarget(frame, boundingBoxes[i], tracker_algorithm))
					{
						cout << "Trackers Init Error!!!";
						return 0;
					}
					rectangle(frame, boundingBoxes[i], mt.colors[0], 2, 1);
				}
				initialized = true;
			}
			else if (initialized)
			{
				//Update all targets
				if (mt.update(frame))
				{
					for (int i = 0; i < mt.targetNum; i++)
					{
						rectangle(frame, mt.boundingBoxes[i], mt.colors[i], 2, 1);
					}
				}
			}
			imshow("Tracking API", frame);
			frameCounter++;
			//Time measurment
			int64 e2 = getTickCount();
			double t1 = (e2 - e1) / getTickFrequency();
			cout << frameCounter << "\tframe :  " << t1 * 1000.0 << "ms" << endl;
		}

		char c = (char)waitKey(2);
		if (c == 'q')
			break;
		if (c == 'p')
			paused = !paused;

		//waitKey(0);
	}

	//Time measurment
	int64 e4 = getTickCount();
	double t2 = (e4 - e3) / getTickFrequency();
	cout << "Average Time for Frame:  " << t2 * 1000.0 / frameCounter << "ms" << endl;
	cout << "Average FPS:  " << 1.0 / t2*frameCounter << endl;


	waitKey(0);

	return 0;
}