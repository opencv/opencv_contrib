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


/**
 */
static void help() {
    std::cout << "Basic marker detection" << std::endl;
    std::cout << "Parameters: " << std::endl;
    std::cout << "-d <dictionary> # 0: ARUCO, ..." << std::endl;
    std::cout << "[-v <videoFile>] # Input from video file, if ommited, input comes from camera"
                 << std::endl;
    std::cout << "[-ci <int>] # Camera id if input doesnt come from video (-v). Default is 0"
                 << std::endl;
    std::cout << "[-c <cameraParams>] # Camera intrinsic parameters. Needed for camera pose"
              << std::endl;
    std::cout << "[-l <markerLength>] # Marker side lenght (in meters). Needed for correct" <<
                 "scale in camera pose, default 0.1" << std::endl;
    std::cout << "[-dp <detectorParams>] # File of marker detector parameters" << std::endl;
    std::cout << "[-r] # show rejected candidates too" << std::endl;
}



/**
 */
static bool isParam(string param, int argc, char **argv ) {
    for (int i=0; i<argc; i++)
        if (string(argv[i]) == param )
            return true;
    return false;

}


/**
 */
static string getParam(string param, int argc, char **argv, string defvalue = "") {
    int idx=-1;
    for (int i=0; i<argc && idx==-1; i++)
        if (string(argv[i]) == param)
            idx = i;
    if (idx == -1 || (idx + 1) >= argc)
        return defvalue;
    else
        return argv[idx+1];
}



/**
 */
static void readCameraParameters(string filename, cv::Mat &camMatrix, cv::Mat &distCoeffs) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
}



/**
 */
static void readDetectorParameters(string filename, cv::aruco::DetectorParameters &params) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    fs["adaptiveThreshWinSize"] >> params.adaptiveThreshWinSize;
    fs["adaptiveThreshConstant"] >> params.adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params.minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params.maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params.polygonalApproxAccuracyRate;
    fs["minCornerDistance"] >> params.minCornerDistance;
    fs["minDistanceToBorder"] >> params.minDistanceToBorder;
    fs["minMarkerDistance"] >> params.minMarkerDistance;
    fs["cornerRefinementWinSize"] >> params.cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params.cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params.cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params.markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params.perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params.perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params.maxErroneousBitsInBorderRate;
}



/**
 */
int main(int argc, char *argv[]) {

    if (!isParam("-d", argc, argv)) {
        help();
        return 0;
    }

    int dictionaryId = atoi( getParam("-d", argc, argv).c_str() );
    cv::aruco::DICTIONARY dictionary = cv::aruco::DICTIONARY(dictionaryId);

    bool showRejected = false;
    if (isParam("-r", argc, argv))
      showRejected = true;

    bool estimatePose = false;
    cv::Mat camMatrix, distCoeffs;
    if (isParam("-c", argc, argv)) {
      readCameraParameters(getParam("-c", argc, argv), camMatrix, distCoeffs);
      estimatePose = true;
    }
    float markerLength = (float)atof( getParam("-l", argc, argv, "0.1").c_str() );

    cv::aruco::DetectorParameters detectorParams;
    if (isParam("-dp", argc, argv)) {
      readDetectorParameters(getParam("-dp", argc, argv), detectorParams);
    }

    cv::VideoCapture inputVideo;
    int waitTime;
    if (isParam("-v", argc, argv)) {
        inputVideo.open(getParam("-v", argc, argv));
        waitTime = 0;
    }
    else {
        int camId = 0;
        if (isParam("-ci", argc, argv))
            camId = atoi( getParam("-ci", argc, argv).c_str() );
        inputVideo.open(camId);
        waitTime = 10;
    }

    double totalTime = 0;
    int totalIterations = 0;

    while (inputVideo.grab()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);

        double tick = (double)cv::getTickCount();

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > corners, rejected;
        std::vector<cv::Mat> rvecs, tvecs;

        // detect markers and estimate pose
        cv::aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);
        if (estimatePose && ids.size() > 0)
            cv::aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs,
                                                 rvecs, tvecs);

        double currentTime = ((double)cv::getTickCount()-tick)/cv::getTickFrequency();
        totalTime += currentTime;
        totalIterations++;
        if(totalIterations%30 == 0) {
            std::cout << "Detection Time = " << currentTime*1000 << " ms " <<
                         "(Mean = " << 1000*totalTime/double(totalIterations) << " ms)" << std::endl;
        }

        // draw results
        image.copyTo(imageCopy);
        if (ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(imageCopy, imageCopy, corners, ids);

            if (estimatePose) {
                for (unsigned int i = 0; i < ids.size(); i++)
                        cv::aruco::drawAxis(imageCopy, imageCopy, camMatrix, distCoeffs, rvecs[i],
                                            tvecs[i], markerLength*0.5f);
            }
        }

        if (showRejected && rejected.size() > 0)
            cv::aruco::drawDetectedMarkers(imageCopy, imageCopy, rejected,
                                               cv::noArray(), cv::Scalar(100, 0, 255));

        cv::imshow("out", imageCopy);
        char key = (char) cv::waitKey(waitTime);
        if (key == 27)
            break;
    }

    return 0;
}
