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

#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

#define NUM_TEST_FRAMES 100
#define TEST_VIDEO_INDEX 7		//TLD Dataset Video Index from 1-10
//#define RECORD_VIDEO_FLG

static Mat image;
static Rect2d boundingBox;
static bool paused;
static bool selectObject = false;
static bool startSelection = false;

static void onMouse(int event, int x, int y, int, void*)
{
	if (!selectObject)
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
			paused = false;
			selectObject = true;
			break;
		case EVENT_MOUSEMOVE:

			if (startSelection && !selectObject)
			{
				//draw the bounding box
				Mat currentFrame;
				image.copyTo(currentFrame);
				rectangle(currentFrame, Point((int)boundingBox.x, (int)boundingBox.y), Point(x, y), Scalar(255, 0, 0), 2, 1);
				imshow("Tracking API", currentFrame);
			}
			break;
		}
	}
}

int main()
{
	//
	//  "MIL", "BOOSTING", "MEDIANFLOW", "TLD"
	//
	string tracker_algorithm_name = "TLD";

	Mat frame;
	paused = false;
	namedWindow("Tracking API", 0);
	setMouseCallback("Tracking API", onMouse, 0);

	Ptr<Tracker> tracker = Tracker::create(tracker_algorithm_name);
	if (tracker == NULL)
	{
		cout << "***Error in the instantiation of the tracker...***\n";
		getchar();
		return 0;
	}

	//Get the first frame
	////Open the capture
	//  VideoCapture cap(0);
	//  if( !cap.isOpened() )
	//  {
	// cout << "Video stream error";
	//    return;
	//  }
	//cap >> frame;

	//From TLD dataset
	selectObject = true;
	boundingBox = tld::tld_InitDataset(TEST_VIDEO_INDEX, "D:/opencv/TLD_dataset");

	frame = tld::tld_getNextDatasetFrame();
	frame.copyTo(image);

	// Setup output video
#ifdef RECORD_VIDEO_FLG
	String outputFilename = "test.avi";
	VideoWriter outputVideo;
	outputVideo.open(outputFilename, -1, 30, Size(image.cols, image.rows));

	if (!outputVideo.isOpened())
	{
		std::cout << "!!! Output video could not be opened" << std::endl;
		getchar();
		return;
	}
#endif

	rectangle(image, boundingBox, Scalar(255, 0, 0), 2, 1);
	imshow("Tracking API", image);


	bool initialized = false;
	int frameCounter = 0;

	//Time measurment
	int64 e3 = getTickCount();

	for (;;)
	{
		//Time measurment
		int64 e1 = getTickCount();
		//Frame num
		frameCounter++;
		if (frameCounter == NUM_TEST_FRAMES) break;

		char c = (char)waitKey(2);
		if (c == 'q' || c == 27)
			break;
		if (c == 'p')
			paused = !paused;

		if (!paused)
		{
			//cap >> frame;
			frame = tld::tld_getNextDatasetFrame();
			if (frame.empty())
			{
				break;
			}
			frame.copyTo(image);

			if (selectObject)
			{
				if (!initialized)
				{
					//initializes the tracker
					if (!tracker->init(frame, boundingBox))
					{
						cout << "***Could not initialize tracker...***\n";
						return 0;
					}
					initialized = true;
					rectangle(image, boundingBox, Scalar(255, 0, 0), 2, 1);
				}
				else
				{
					//updates the tracker
					if (tracker->update(frame, boundingBox))
					{
						rectangle(image, boundingBox, Scalar(255, 0, 0), 2, 1);
					}
				}
			}
			imshow("Tracking API", image);

#ifdef RECORD_VIDEO_FLG
			outputVideo << image;
#endif


			//Time measurment
			int64 e2 = getTickCount();
			double t1 = (e2 - e1) / getTickFrequency();
			cout << frameCounter << "\tframe :  " << t1 * 1000.0 << "ms" << endl;

			//waitKey(0);
		}
	}

	//Time measurment
	int64 e4 = getTickCount();
	double t2 = (e4 - e3) / getTickFrequency();
	cout << "Average Time for Frame:  " << t2 * 1000.0 / frameCounter << "ms" << endl;
	cout << "Average FPS:  " << 1.0 / t2*frameCounter << endl;


	waitKey(0);

	return 0;
}