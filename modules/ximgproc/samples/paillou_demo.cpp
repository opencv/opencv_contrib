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
#include "opencv2/ximgproc/paillou_filter.hpp"

using namespace cv;
using namespace cv::ximgproc;

#include <iostream>
using namespace std;

int aa = 100, ww = 10;

const char* window_name = "Gradient Modulus";

static void DisplayImage(Mat x,string s)
{
    vector<Mat> sx;
    split(x, sx);
    vector<double> minVal(3), maxVal(3);
    for (int i = 0; i < static_cast<int>(sx.size()); i++)
    {
        minMaxLoc(sx[i], &minVal[i], &maxVal[i]);
    }
    maxVal[0] = *max_element(maxVal.begin(), maxVal.end());
    minVal[0] = *min_element(minVal.begin(), minVal.end());
    Mat uc;
    x.convertTo(uc, CV_8U,255/(maxVal[0]-minVal[0]),-255*minVal[0]/(maxVal[0]-minVal[0]));
    imshow(s, uc);
}


/**
 * @function paillouFilter
 * @brief Trackbar callback
 */
static void PaillouFilter(int, void*pm)
{
    Mat img = *((Mat*)pm);
    Mat dst;
    double a=aa/100.0, w=ww/100.0;
    Mat rx,ry;
    GradientPaillouX(img, rx, a, w);
    GradientPaillouY(img, ry, a, w);
    DisplayImage(rx, "Gx");
    DisplayImage(ry, "Gy");
    add(rx.mul(rx), ry.mul(ry), dst);
    sqrt(dst, dst);
    DisplayImage(dst, window_name );
}


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        cout << "usage: paillou_demo [image]" << endl;
        return 1;
    }
    Mat img = imread(argv[1]);
    if (img.empty())
    {
        cout << "File not found or empty image\n";
        return 1;
    }
    imshow("Original",img);
    namedWindow( window_name, WINDOW_AUTOSIZE );

    /// Create a Trackbar for user to enter threshold
    createTrackbar( "a:",window_name, &aa, 400, PaillouFilter, &img );
    createTrackbar( "w:", window_name, &ww, 400, PaillouFilter, &img );
    PaillouFilter(0, &img);
    waitKey();
    return 0;
}
