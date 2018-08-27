/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>

using namespace cv;
using namespace cv::ximgproc;

#include <iostream>
using namespace std;

int sColor = 10, sSpace = 3,nbIter=4;

const char* window_name = "Rolling guidance filter";



/**
 * @function paillouFilter
 * @brief Trackbar callback
 */
static void rollingFilter(int, void *pm)
{
    Mat img = *((Mat*)pm);
    double sigmaColor(sColor/10.0), sigmaSpace(sSpace);
    Mat dst;
    rollingGuidanceFilter(img, dst, -1, sigmaColor, sigmaSpace, nbIter);
    imshow("rollingGuidanceFilter",dst );
}


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        cout << "usage: rollingGuidanceFilter_demo [image]" << endl;
        return 1;
    }
    Mat img = imread(argv[1]);
    if (img.empty())
    {
        cout << "File not found or empty image\n";
        return 1;
    }

    imshow("Original",img);
    Mat imgF;
    img.convertTo(imgF, CV_32F,1.0/255);
    namedWindow( window_name, WINDOW_KEEPRATIO);
    imshow(window_name, img);

    /// Create a Trackbar for user to enter threshold
    createTrackbar( "sColor",window_name, &sColor, 10, rollingFilter, &imgF );
    createTrackbar("sSpace", window_name, &sSpace, 400, rollingFilter, &imgF);
    createTrackbar("iter", window_name, &nbIter, 10, rollingFilter, &imgF);
    rollingFilter(0, &imgF);
    waitKey();
    return 0;
}
