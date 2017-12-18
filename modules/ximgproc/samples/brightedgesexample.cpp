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
// Copyright (C) 2017, IBM Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//      Marc Fiammante marc.fiammante@fr.ibm.com
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
//   * The name of OpenCV Foundation or contributors may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <iostream>
#include "opencv2/ximgproc.hpp"
using namespace cv;
using namespace ximgproc;
using namespace std;
// Default values for sliders and contrast
int edgeThresh = 1;
int contrast_slider = 2;
int canny_slider = 50;
static void help()
{
    printf("\nThis sample demonstrates BrightEdge detection\n"
        "Call:\n"
        "    brightedges [image_name -- Default is ../data/ml.png] | [-video -- for webcam]\n\n");
}
const char* keys =
{
    "{help h||}"
    "{video v| |video capture}"
    "{@image |../data/ml.png|input image name}"
};
int main(int argc, const char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    if (parser.has("video"))
    {
        namedWindow("Brightedges", 1);
        createTrackbar("Contrast", "Brightedges", &contrast_slider, 25);
        namedWindow("Canny", 1);
        createTrackbar("Threshold", "Canny", &canny_slider, 255);
        VideoCapture cap(0); // open the default camera
        if (!cap.isOpened())  // check if we succeeded
            return -1;
        Mat edges;
        for (;;)
        {
            Mat frame;
            cap.read(frame); // get a new frame from camera
            Mat edge;
            BrightEdges(frame, edge, -contrast_slider); // Apply brighedges with noise reduction
            Mat notedge;
            if (contrast_slider == 0) {
                threshold(edge, notedge, 155, 255, THRESH_BINARY);
            }
            else {
                bitwise_not(edge, notedge); // make comparable with Canny
            }
            imshow("Brightedges", notedge);
            imshow("frame", frame);
            Canny(frame, edges, canny_slider, 255);
            imshow("Canny", edges);
            if (waitKey(30) >= 0) break;
        }
        // the camera will be deinitialized automatically in VideoCapture destructor
        return 0;
    }
    else {
        string filename = parser.get<string>(0);
        Mat image = imread(filename, IMREAD_COLOR);
        if (image.empty())
        {
            printf("Cannot read image file: %s\n", filename.c_str());
            help();
            return -1;
        }
        // Create a window
        // " original ";
        imshow("Original", image);
        //  " absdiff ";
        Mat edge;
        BrightEdges(image, edge, 0); //  No contrast
        imshow("Absolute Difference", edge);
        // " default contrast 1 ";
        BrightEdges(image, edge);
        imshow("Default contrast", edge); // Default contrast 1
        BrightEdges(image, edge, -1);  // Reduce noise by specifying negative contrast
        imshow("Reduce noise", edge);
        //  "wait key ";
        waitKey(0);
        //  "end  ";
        return 0;
    }
}