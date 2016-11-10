/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2016, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

/*
Contributed by Gregor Kovalcik <gregor dot kovalcik at gmail dot com>
    based on code provided by Martin Krulis, Jakub Lokoc and Tomas Skopal.

References:
    Martin Krulis, Jakub Lokoc, Tomas Skopal.
    Efficient Extraction of Clustering-Based Feature Signatures Using GPU Architectures.
    Multimedia tools and applications, 75(13), pp.: 8071–8103, Springer, ISSN: 1380-7501, 2016

    Christian Beecks, Merih Seran Uysal, Thomas Seidl.
    Signature quadratic form distance.
    In Proceedings of the ACM International Conference on Image and Video Retrieval, pages 438-445.
    ACM, 2010.
*/

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <iostream>
#include <string>

using namespace std;
using namespace cv;
using namespace xfeatures2d;


void printHelpMessage(void);
void printHelpMessage(void)
{
    cout << "Example of the PCTSignatures algorithm.\n\n"
        "This program computes and visualizes position-color-texture signatures\n"
        "using images from webcam if available.\n\n"
        "Usage:\n"
        "pct_webcam [sample_count] [seed_count]\n"
        "Note: sample_count must be greater or equal to seed_count.";
}


/** @brief

Example of the PCTSignatures algorithm.

This program computes and visualizes position-color-texture signatures
of images taken from webcam if available.
*/
int main(int argc, char** argv)
{
    // define variables
    Mat frame, signature, result;
    int initSampleCount = 2000;
    int initSeedCount = 400;
    int grayscaleBitsPerPixel = 4;

    // parse for help argument
    {
        for (int i = 1; i < argc; i++)
        {
            if ((string)argv[i] == "-h" || (string)argv[i] == "--help")
            {
                printHelpMessage();
                return 0;
            }
        }
    }

    // parse optional arguments
    if (argc > 1)               // sample count
    {
        initSampleCount = atoi(argv[1]);
        if (initSampleCount <= 0)
        {
            cerr << "Sample count have to be a positive integer: " << argv[1] << endl;
            return 1;
        }
        initSeedCount = (int)floor(initSampleCount / 4);
        initSeedCount = std::max(1, initSeedCount);     // fallback if sample count == 1
    }
    if (argc > 2)               // seed count
    {
        initSeedCount = atoi(argv[2]);
        if (initSeedCount <= 0)
        {
            cerr << "Seed count have to be a positive integer: " << argv[2] << endl;
            return 1;
        }
        if (initSeedCount > initSampleCount)
        {
            cerr << "Seed count have to be lower or equal to sample count!" << endl;
            return 1;
        }
    }

    // create algorithm
    Ptr<PCTSignatures> pctSignatures = PCTSignatures::create(initSampleCount, initSeedCount, PCTSignatures::UNIFORM);
    pctSignatures->setGrayscaleBits(grayscaleBitsPerPixel);

    // open video capture device
    VideoCapture videoCapture;
    if (!videoCapture.open(0))
    {
        cerr << "Unable to open the first video capture device with ID = 0!" << endl;
        return 1;
    }

    // Create windows for display.
    namedWindow("Source", WINDOW_AUTOSIZE);
    namedWindow("Result", WINDOW_AUTOSIZE);

    // run drawing loop
    for (;;)
    {
        videoCapture >> frame;
        if (frame.empty()) break; // end of video stream

        pctSignatures->computeSignature(frame, signature);
        PCTSignatures::drawSignature(Mat::zeros(frame.size(), frame.type()), signature, result);

        imshow("Source", frame);                // Show our images inside the windows.
        imshow("Result", result);

        if (waitKey(1) == 27) break;            // stop videocapturing by pressing ESC
    }

    return 0;
}
