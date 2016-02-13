/****************************************************************************************
 * By downloading, copying, installing or using the software you agree to this license. *
 * If you do not agree to this license, do not download, install,                       *
 * copy or use the software.                                                            *
 *                                                                                      *
 *                                                                                      *
 *                         License Agreement                                            *
 *              For Open Source Computer Vision Library                                 *
 *                     (3-clause BSD License)                                           *
 *                                                                                      *
 * Copyright (C) 2000-2016, Intel Corporation, all rights reserved.                     *
 * Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.                    *
 * Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.                    *
 * Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.          *
 * Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.                     *
 * Copyright (C) 2015-2016, Itseez Inc., all rights reserved.                           *
 * Third party copyrights are property of their respective owners.                      *
 *                                                                                      *
 * Redistribution and use in source and binary forms, with or without modification,     *
 * are permitted provided that the following conditions are met:                        *
 *                                                                                      *
 * Redistributions of source code must retain the above copyright notice,               *
 * this list of conditions and the following disclaimer.                                *
 *                                                                                      *
 * Redistributions in binary form must reproduce the above copyright notice,            *
 * this list of conditions and the following disclaimer in the documentation            *
 * and/or other materials provided with the distribution.                               *
 *                                                                                      *
 * Neither the names of the copyright holders nor the names of the contributors         *
 * may be used to endorse or promote products derived from this software                *
 * without specific prior written permission.                                           *
 *                                                                                      *
 * This software is provided by the copyright holders and contributors "as is" and      *
 * any express or implied warranties, including, but not limited to, the implied        *
 * warranties of merchantability and fitness for a particular purpose are disclaimed.   *
 * In no event shall copyright holders or contributors be liable for any direct,        *
 * indirect, incidental, special, exemplary, or consequential damages                   *
 * (including, but not limited to, procurement of substitute goods or services;         *
 * loss of use, data, or profits; or business interruption) however caused              *
 * and on any theory of liability, whether in contract, strict liability,               *
 * or tort (including negligence or otherwise) arising in any way out of                *
 * the use of this software, even if advised of the possibility of such damage.         *
 ****************************************************************************************/

 /**
  * @author {aravind | arvindsuresh2009@gmail.com}
  * Created on Mon, Feb 15 2016
  */

 /**
  *
  * C++ Sample program for Radon transform
  *
  */

// Necessary headers
#include <math.h>
#include <stdlib.h>
#include <string>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>

#include <opencv2/ximgproc.hpp>

using namespace cv::ximgproc;
using namespace cv;
using namespace std;

static void help ( const char **argv ) {
  std::cout << std::endl                                                  \
            << "This program demonstrates the usage of Radon transform."  \
            << std::endl << std::endl << "USAGE: " << std::endl           \
            << argv[0] << " <filename> <angle-range-option> <operation>"  \
            << std::endl << std::endl;

  std::cout << "Default for <angle-range-option> computes for "           \
            << "all angles between 1 and 180 degrees."                    \
            << " ( @see cv::ximgproc::RadonAngleRange )" << std::endl     \
            << "Default for <operation> is RT_SUM."                       \
            << " ( @see cv::ximgproc::RadonOp )" << std::endl             \
            << std::endl;                                                 \

}

static bool argParser( int argc, const char **argv,
                       cv::Mat & img,
                       int & radonAngleRange,
                       int & radonOperation ) {
    if (argc > 4) {
        std::cout << "Incorrect arguments" << std::endl;
        return false;
    }

    const char *filename = argc >= 2 ? argv[1]
                                     : "./radon_input.jpg";
    img = imread(filename, 0);
    if( img.empty() ) {
        std::cout << "Unable to load image: " << filename << std::endl;
        return false;
    }

    radonAngleRange = ( argc >= 3 ) ? atoi(argv[2]) : 63;     // 1 to 180 degrees
    radonOperation = ( argc >= 4 ) ? atoi(argv[3]) : RT_SUM;  // Sum up elements

    return true;
}

int main( int argc, const char ** argv ) {

  cv::Mat img;
  int radonAngleRange, radonOperation;

  // Display help
  help( argv );

  if( !argParser( argc, argv, img, radonAngleRange, radonOperation) ) {
    return -1;
  }

  cv::Mat radonTr, operImg;

  // Computing the Radon transform with appropriate params
  radonTransform( img, radonTr, radonAngleRange, radonOperation, operImg );

  // Mat for displaying the Radon transform
  cv::Mat radonTrDisp;

  double minVal, maxVal;
  minMaxLoc(radonTr, &minVal, &maxVal);

  // Normalizing radonTr so as to display as a CV_8U image
  radonTr -= minVal;
  radonTr.convertTo( radonTrDisp, CV_8U, 255.0/(maxVal-minVal) );

  // Normalizing operImg so as to display as a CV_8U image
  minMaxLoc(operImg, &minVal, &maxVal);
  operImg -= minVal;
  operImg.convertTo(operImg, CV_8U, 255.0/(maxVal-minVal));

  // Displaying the images
  cv::imshow("Input", img);
  cv::imshow("Radon transform", radonTrDisp);
  cv::imshow("Operation(image)", operImg);

  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}
