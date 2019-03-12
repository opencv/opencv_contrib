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
// Copyright (C) 2015, Itseez Inc, all rights reserved.
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
// Author: Jiaolong Xu <jiaolongxu AT gmail.com>
//M*/

#include <opencv2/dpm.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>

#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace cv::dpm;
using namespace std;

static void help()
{
    cout << "\nThis is a demo of \"Deformable Part-based Model (DPM) cascade detection API\" using web camera.\n"
       "Call:\n"
       "./example_dpm_cascade_detect_camera <model_path>\n"
       << endl;
}

void drawBoxes(Mat &frame,
        vector<DPMDetector::ObjectDetection> ds,
        Scalar color,
        string text);

int main( int argc, char** argv )
{
    const char* keys =
    {
        "{@model_path    | | Path of the DPM cascade model}"
    };

    CommandLineParser parser(argc, argv, keys);
    string model_path(parser.get<string>(0));

    if( model_path.empty() )
    {
        help();
        return -1;
    }

    cv::Ptr<DPMDetector> detector = \
    DPMDetector::create(vector<string>(1, model_path));

    // use web camera
    VideoCapture capture(0);
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

    if ( !capture.isOpened() )
    {
        cerr << "Fail to open default camera (0)!" << endl;
        return -1;
    }

    Mat frame;
    namedWindow("DPM Cascade Detection", 1);
    // the color of the rectangle
    Scalar color(0, 255, 255); // yellow

    while( capture.read(frame) )
    {
        vector<DPMDetector::ObjectDetection> ds;

        Mat image;
        frame.copyTo(image);

        double t = (double) getTickCount();
        // detection
        detector->detect(image, ds);
        // compute frame per second (fps)
        t = ((double) getTickCount() - t)/getTickFrequency();//elapsed time

        // draw boxes
        string text = format("%0.1f fps", 1.0/t);
        drawBoxes(frame, ds, color, text);

        imshow("DPM Cascade Detection", frame);

        if ( waitKey(30) >= 0)
            break;
    }

    return 0;
}

void drawBoxes(Mat &frame,
        vector<DPMDetector::ObjectDetection> ds,
        Scalar color,
        string text)
{
    for (unsigned int i = 0; i < ds.size(); i++)
    {
        rectangle(frame, ds[i].rect, color, 2);
    }

    // draw text on image
    Scalar textColor(0,0,250);
    putText(frame, text, Point(10,50), FONT_HERSHEY_PLAIN, 2, textColor, 2);
}
