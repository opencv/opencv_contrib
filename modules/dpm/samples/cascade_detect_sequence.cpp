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

#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dpm;
using namespace std;

int save_results(const string id, const vector<DPMDetector::ObjectDetection> ds, ofstream &out);

static void help()
{
    cout << "\nThis example shows object detection on image sequences using \"Deformable Part-based Model (DPM) cascade detection API\n"
       "Call:\n"
       "./example_dpm_cascade_detect_sequence <model_path> <image_dir>\n"
       "The image names has to be provided in \"files.txt\" under <image_dir>.\n"
       << endl;
}

static bool readImageLists( const string &file, vector<string> &imgFileList)
{
    ifstream in(file.c_str(), ios::binary);

    if (in.is_open())
    {
        while (in)
        {
            string line;
            getline(in, line);
            imgFileList.push_back(line);
        }
        return true;
    }
    else
    {
        cerr << "Invalid image index file: " << file  << endl;
        return false;
    }
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
        "{@image_dir     | | Directory of the images      }"
    };

    CommandLineParser parser(argc, argv, keys);
    string model_path(parser.get<string>(0));
    string image_dir(parser.get<string>(1));
    string image_list = image_dir + "/files.txt";

    if( model_path.empty() || image_dir.empty() )
    {
        help();
        return -1;
    }

    vector<string> imgFileList;
    if ( !readImageLists(image_list, imgFileList) )
        return -1;

    cv::Ptr<DPMDetector> detector = \
    DPMDetector::create(vector<string>(1, model_path));

    namedWindow("DPM Cascade Detection", 1);
    // the color of the rectangle
    Scalar color(0, 255, 255); // yellow
    Mat frame;

    for (size_t i = 0; i < imgFileList.size(); i++)
    {
        double t = (double) getTickCount();
        vector<DPMDetector::ObjectDetection> ds;

        Mat image = imread(image_dir + "/" + imgFileList[i]);
        frame = image.clone();

        if (image.empty()) {
            cerr << "\nInvalid image:\n" << imgFileList[i] << endl;
            return -1;
        }

        // detection
        detector->detect(image, ds);
        // compute frame per second (fps)
        t = ((double) getTickCount() - t)/getTickFrequency();//elapsed time

        // draw boxes
        string text = format("%0.1f fps", 1.0/t);
        drawBoxes(frame, ds, color, text);

        // show detections
        imshow("DPM Cascade Detection", frame);

        if ( waitKey(30) >= 0)
            break;
    }

    return 0;
}

void drawBoxes(Mat &frame, \
        vector<DPMDetector::ObjectDetection> ds, Scalar color, string text)
{
    for (unsigned int i = 0; i < ds.size(); i++)
    {
        rectangle(frame, ds[i].rect, color, 2);
    }

    // draw text on image
    Scalar textColor(0,0,250);
    putText(frame, text, Point(10,50), FONT_HERSHEY_PLAIN, 2, textColor, 2);
}
