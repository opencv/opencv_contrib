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

//Demo of GOTURN tracker
//In order to use GOTURN tracker, GOTURN architecture goturn.prototxt and goturn.caffemodel are required to exist in root folder.
//There are 2 ways to get caffemodel:
//1 - Train you own GOTURN model using <https://github.com/Auron-X/GOTURN_Training_Toolkit>
//2 - Download pretrained caffemodel from <https://github.com/opencv/opencv_extra>

#include "opencv2/opencv_modules.hpp"
#if defined(HAVE_OPENCV_DNN) && defined(HAVE_OPENCV_DATASETS)

#include "opencv2/datasets/track_alov.hpp"
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
Rect2d boundingBox;

static const char* keys =
{ "{@dataset_path     || Dataset path   }"
  "{@dataset_id      |1| Dataset ID     }"
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
            paused = false;
            selectObjects = true;
            startSelection = false;
            break;
        case EVENT_MOUSEMOVE:

            if (startSelection && !selectObjects)
            {
                //draw the bounding box
                Mat currentFrame;
                image.copyTo(currentFrame);
                rectangle(currentFrame, Point((int)boundingBox.x, (int)boundingBox.y), Point(x, y), Scalar(255, 0, 0), 2, 1);
                imshow("GOTURN Tracking", currentFrame);
            }
            break;
        }
    }
}

static void help()
{
    cout << "\nThis example is a simple demo of GOTURN tracking on ALOV300++ dataset"
        "ALOV dataset contains videos with ID range: 1~314\n"
        "-- pause video [p] and draw a bounding boxes around the targets to start the tracker\n"
        "Example:\n"
        "./goturnTracker <dataset_path> <dataset_id>\n"
        << endl;

    cout << "\n\nHot keys: \n"
        "\tq - quit the program\n"
        "\tp - pause video\n";
}

int main(int argc, char *argv[])
{
    CommandLineParser parser(argc, argv, keys);
    string datasetRootPath = parser.get<string>(0);
    int datasetID = parser.get<int>(1);

    if (datasetRootPath.empty())
    {
        help();
        return -1;
    }

    Mat frame;
    paused = false;
    namedWindow("GOTURN Tracking", 0);
    setMouseCallback("GOTURN Tracking", onMouse, 0);

    //Create GOTURN tracker
    Ptr<Tracker> tracker = TrackerGOTURN::create();

    //Load and init full ALOV300++ dataset with a given datasetID, as alternative you can use loadAnnotatedOnly(..)
    //to load only frames with labled ground truth ~ every 5-th frame
    Ptr<cv::datasets::TRACK_alov> dataset = TRACK_alov::create();
    dataset->load(datasetRootPath);
    dataset->initDataset(datasetID);
    //Read first frame
    dataset->getNextFrame(frame);
    if (frame.empty())
    {
        cout << "invalid dataset: " << datasetRootPath << endl;
        return -2;
    }

    frame.copyTo(image);
    rectangle(image, boundingBox, Scalar(255, 0, 0), 2, 1);
    imshow("GOTURN Tracking", image);

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
                if (!tracker->init(frame, boundingBox))
                {
                    cout << "Tracker Init Error!!!";
                    return 0;
                }
                rectangle(frame, boundingBox, Scalar(0, 0, 255), 2, 1);
                initialized = true;
            }
            else if (initialized)
            {
                //Update all targets
                if (tracker->update(frame, boundingBox))
                {
                    rectangle(frame, boundingBox, Scalar(0, 0, 255), 2, 1);
                }
            }
            imshow("GOTURN Tracking", frame);
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
    }

    //Time measurment
    int64 e4 = getTickCount();
    double t2 = (e4 - e3) / getTickFrequency();
    cout << "Average Time for Frame:  " << t2 * 1000.0 / frameCounter << "ms" << endl;
    cout << "Average FPS:  " << 1.0 / t2*frameCounter << endl;


    waitKey(0);

    return 0;
}

#else // ! HAVE_OPENCV_DNN && HAVE_OPENCV_DATASETS
#include <opencv2/core.hpp>
int main() {
    CV_Error(cv::Error::StsNotImplemented , "this sample needs to be built with opencv_datasets and opencv_dnn !");
    return -1;
}
#endif
