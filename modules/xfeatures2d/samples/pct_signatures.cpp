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
    cout << "Example of the PCTSignatures algorithm computing and visualizing\n"
        "image signature for one image, or comparing multiple images with the first\n"
        "image using the signature quadratic form distance.\n\n"
        "Usage: pct_signatures ImageToProcessAndDisplay\n"
        "or:    pct_signatures ReferenceImage [ImagesToCompareWithTheReferenceImage]\n\n"
        "The program has 2 modes:\n"
        "- single argument: program computes and visualizes the image signature\n"
        "- multiple arguments: program compares the first image to the others\n"
        "  using pct signatures and signature quadratic form distance (SQFD)";
}

/** @brief

Example of the PCTSignatures algorithm.

The program has 2 modes:
- single argument mode, where the program computes and visualizes the image signature
- multiple argument mode, where the program compares the first image to the others
using signatures and signature quadratic form distance (SQFD)

*/
int main(int argc, char** argv)
{
    if (argc < 2)                               // Check arguments
    {
        printHelpMessage();
        return 1;
    }

    Mat source;
    source = imread(argv[1]);                   // Read the file

    if (!source.data)                           // Check for invalid input
    {
        cerr << "Could not open or find the image: " << argv[1];
        return -1;
    }

    Mat signature, result;                      // define variables
    int initSampleCount = 2000;
    int initSeedCount = 400;
    int grayscaleBitsPerPixel = 4;
    vector<Point2f> initPoints;

    namedWindow("Source", WINDOW_AUTOSIZE);     // Create windows for display.
    namedWindow("Result", WINDOW_AUTOSIZE);

                                                // create the algorithm
    PCTSignatures::generateInitPoints(initPoints, initSampleCount, PCTSignatures::UNIFORM);
    Ptr<PCTSignatures> pctSignatures = PCTSignatures::create(initPoints, initSeedCount);
    pctSignatures->setGrayscaleBits(grayscaleBitsPerPixel);

                                                // compute and visualize the first image
    double start = (double)getTickCount();
    pctSignatures->computeSignature(source, signature);
    double end = (double)getTickCount();
    cout << "Signature of the reference image computed in " << (end - start) / (getTickFrequency() * 1.0f) << " seconds." << endl;
    PCTSignatures::drawSignature(source, signature, result);

    imshow("Source", source);                   // show the result
    imshow("Result", result);

    if (argc == 2)          // single image -> finish right after the visualization
    {
        waitKey(0);         // Wait for user input
        return 0;
    }
    // multiple images -> compare to the first one
    else
    {
        vector<Mat> images;
        vector<Mat> signatures;
        vector<float> distances;

        for (int i = 2; i < argc; i++)
        {
            Mat image = imread(argv[i]);
            if (!source.data)                               // Check for invalid input
            {
                cerr << "Could not open or find the image: " << argv[i] << std::endl;
                return 1;
            }
            images.push_back(image);
        }

        pctSignatures->computeSignatures(images, signatures);
        Ptr<PCTSignaturesSQFD> pctSQFD = PCTSignaturesSQFD::create();
        pctSQFD->computeQuadraticFormDistances(signature, signatures, distances);

        for (int i = 0; i < (int)(distances.size()); i++)
        {
            cout << "Image: " << argv[i + 2] << ", similarity: " << distances[i] << endl;
        }
        waitKey(0); // Wait for user input
    }

    return 0;
}
