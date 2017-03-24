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
 *  *Redistributions of source code must retain the above copyright notice,
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
#include "opencv2/ximgproc/deriche_filter.hpp"

using namespace cv;
using namespace cv::ximgproc;

#include <iostream>
using namespace std;

int alDerive=100;
int alMean=100;
Ptr<Mat> img;
const string & winName = "Gradient Modulus";

static void DisplayImage(Mat x,string s)
{
	vector<Mat> sx;
	split(x, sx);
	vector<double> minVal(3), maxVal(3);
	for (size_t i = 0; i < sx.size(); i++)
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
 * @function DericheFilter
 * @brief Trackbar callback
 */
static void DericheFilter(int, void*)
{
    Mat dst;
    double d=alDerive/100.0,m=alMean/100.0;
    Mat rx,ry;
    GradientDericheX(*img.get(),rx,d,m);
    GradientDericheY(*img.get(),ry,d,m);
    DisplayImage(rx, "Gx");
    DisplayImage(ry, "Gy");
    add(rx.mul(rx),ry.mul(ry),dst);
    sqrt(dst,dst);
    DisplayImage(dst, winName );
}

int main(int argc, char* argv[])
{
    Mat *m=new Mat;
    cv::CommandLineParser parser(argc, argv, "{help h | | show help message}{@input | | input image}");
    if (parser.has("help"))
    {
        parser.printMessage();
        return -1;
    }
    string input_image = parser.get<string>("@input");
    if (input_image.empty())
    {
        parser.printMessage();
        parser.printErrors();
        return -2;
    }
    if (argc==2)
        *m = imread(input_image);
    if (m->empty())
    {
        cout << "File not found or empty image\n";
        return -3;
    }
    imshow("Original", *m);
    img =Ptr<Mat>(m);
    namedWindow( winName, WINDOW_AUTOSIZE );
    /// Create a Trackbar for user to enter threshold
    createTrackbar( "Derive:",winName, &alDerive, 400, DericheFilter );
    createTrackbar( "Mean:", winName, &alMean, 400, DericheFilter );
    DericheFilter(0,NULL);
    waitKey();
    return 0;
}