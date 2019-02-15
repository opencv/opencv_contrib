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

static void help()
{
    printf("\nThis sample demonstrates BrightEdge detection\n"
        "Call:\n"
        "    /.edge [image_name -- Default is ../data/ml.png]\n\n");
}
const char* keys =
{
    "{help h||}{@image |../data/ml.png|input image name}"
};
int main(int argc, const char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    string filename = parser.get<string>(0);
    Mat image = imread(filename, IMREAD_COLOR);
    if (image.empty())
    {
        printf("Cannot read image file: %s\n", filename.c_str());
        help();
        return -1;
    }
    // Create a window
    // //  " original ";
    namedWindow("Original");
    imshow("Original", image);
    //  " absdiff ";
    Mat edge;
    BrightEdges(image, edge, 0); //  No contrast
    namedWindow("Absolute Difference");
    imshow("Absolute Difference", edge);
    // " default contrast 1 ";
    BrightEdges(image, edge);
    namedWindow("Default contrast");
    imshow("Default contrast", edge);// Default contrast 1
    // " Contrast 5  \n";
    BrightEdges(image, edge, 5);
    namedWindow("Contrast 5");
    imshow("Contrast 5", edge);
    // " Contrast 10  \n";
    BrightEdges(image, edge, 10);
    namedWindow("Contrast 10");
    imshow("Contrast 10", edge);
    //  "wait key ";
    waitKey(0);
    //  "end  ";
    return 0;
}
