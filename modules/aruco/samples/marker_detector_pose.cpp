/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/


#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

    if (argc < 2) {
        std::cerr << "Use: marker_detector_pose video" << std::endl;
        return 0;
    }

    cv::VideoCapture input;
    input.open(argv[1]);

    // Parameters for the Aruco test videos
    cv::Mat camMatrix(3, 3, CV_64FC1, cv::Scalar::all(0));
    camMatrix.at<double>(0, 0) = 628.158;
    camMatrix.at<double>(0, 2) = 324.099;
    camMatrix.at<double>(1, 1) = 628.156;
    camMatrix.at<double>(1, 2) = 260.908;
    camMatrix.at<double>(2, 2) = 1.;

    cv::Mat distCoeffs(5, 1, CV_64FC1, cv::Scalar::all(0));
    distCoeffs.ptr<double>(0)[0] = 0.0995485;
    distCoeffs.ptr<double>(0)[1] = -0.206384;
    distCoeffs.ptr<double>(0)[2] = 0.00754589;
    distCoeffs.ptr<double>(0)[3] = 0.00336531;

    float markerSize = 0.04; // meters

    while (input.grab()) {
        cv::Mat image, imageCopy;
        input.retrieve(image);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > imgPoints;
        std::vector<cv::Mat> rvecs, tvecs;

        // detect markers and estimate pose
        cv::aruco::detectMarkers(image, cv::aruco::DICT_ARUCO, imgPoints, ids);
        if (ids.size() > 0)
            cv::aruco::estimatePoseSingleMarkers(imgPoints, markerSize, camMatrix, distCoeffs,
                                                 rvecs, tvecs);

        // draw results
        if (ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(image, imageCopy, imgPoints, ids);
            for (int i = 0; i < ids.size(); i++)
                cv::aruco::drawAxis(imageCopy, imageCopy, camMatrix, distCoeffs, rvecs[i], tvecs[i],
                                    0.04);
        } else
            image.copyTo(imageCopy);

        cv::imshow("out", imageCopy);
        char key = cv::waitKey(0);
        if (key == 27)
            break;
    }

    return 0;
}
