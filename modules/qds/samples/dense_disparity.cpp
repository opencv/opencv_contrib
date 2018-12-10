/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2015-2018, OpenCV Foundation, all rights reserved.
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


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/qds.hpp>




using namespace cv;
using namespace std;


int main()
{

    Mat rightImg, leftImg;

    //Read video meta-data to determine the correct frame size for initialization.
    leftImg = imread("./imgLeft.png", IMREAD_COLOR);
    rightImg = imread("./imgRight.png", IMREAD_COLOR);
    cv::Size frameSize = leftImg.size();
    // Initialize qds and start process.
    qds::QuasiDenseStereo stereo(frameSize);

    int displvl = 80;					// Number of disparity levels
    cv::Mat disp;

    // Compute dense stereo.
    stereo.process(leftImg, rightImg);

    // Compute disparity between left and right channel of current frame.
    disp = stereo.getDisparity(displvl);

    vector<qds::Match> matches;
    stereo.getDenseMatches(matches);

    // Create three windows and show images.
    cv::namedWindow("right channel");
    cv::namedWindow("left channel");
    cv::namedWindow("disparity map");
    cv::imshow("disparity map", disp);
    cv::imshow("left channel", leftImg);
    cv::imshow("right channel", rightImg);

    std::ofstream dense("./dense.txt", std::ios::out);

    for (uint i=0; i< matches.size(); i++)
    {
        dense << matches[i].p0 << matches[i].p1 << endl;
    }
    dense.close();



    cv::waitKey(0);

    return 0;
}
