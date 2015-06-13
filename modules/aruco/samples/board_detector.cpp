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
    std::cout << "Pose estimation using a ArUco Planar Grid board" << std::endl;
    std::cout << "Parameters: " << std::endl;
    std::cout << "-w <nmarkers> # Number of markers in X direction" << std::endl;
    std::cout << "-h <nmarkers> # Number of markers in Y direction" << std::endl;
    std::cout << "-l <markerLength> # Marker side lenght (in meters)" << std::endl;
    std::cout << "-s <markerSeparation> # Separation between two consecutive" << 
                 "markers in the grid (in meters)" << std::endl;
    std::cout << "-d <dictionary> # 0: ARUCO, ..." << std::endl;
    std::cout << "-c <cameraParams> # Camera intrinsic parameters file"
              << std::endl;    
    std::cout << "[-v <videoFile>] # Input from video file, if ommited, input comes from camera"
                 << std::endl;
    std::cout << "[-dp <detectorParams>] # File of marker detector parameters" << std::endl;
    std::cout << "[-r] # show rejected candidates too" << std::endl;    
}


/**
 */
bool isParam(string param, int argc, char **argv ) {
    for (int i=0; i<argc; i++)
        if (string(argv[i]) == param ) 
            return true;
    return false;

}


/**
 */
string getParam(string param, int argc, char **argv, string defvalue = "") {
    int idx=-1;
    for (int i=0; i<argc && idx==-1; i++)
        if (string(argv[i]) == param) 
            idx = i;
    if (idx == -1 || (idx + 1) >= argc)
        return defvalue;
    else
        return argv[idx+1] ;
} 


/**
 */
void readCameraParameters(string filename, cv::Mat &camMatrix, cv::Mat &distCoeffs) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;  
}


/**
 */
void readDetectorParameters(string filename, cv::aruco::DetectorParameters &params) {
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

    if (!isParam("-w", argc, argv) || !isParam("-h", argc, argv) || !isParam("-l", argc, argv) ||
        !isParam("-s", argc, argv) || !isParam("-d", argc, argv) || !isParam("-c", argc, argv) ) {
        help();
        return 0;
    }

    int markersX = atoi( getParam("-w", argc, argv).c_str() );
    int markersY = atoi( getParam("-h", argc, argv).c_str() );
    float markerLength = atof( getParam("-l", argc, argv).c_str() );
    float markerSeparation = atof( getParam("-s", argc, argv).c_str() );
    int dictionaryId = atoi( getParam("-d", argc, argv).c_str() );
    cv::aruco::DICTIONARY dictionary = cv::aruco::DICTIONARY(dictionaryId);
    
    bool showRejected = false;
    if (isParam("-r", argc, argv))
      showRejected = true;    
    
    cv::Mat camMatrix, distCoeffs;
    if (isParam("-c", argc, argv)) {
      readCameraParameters(getParam("-c", argc, argv), camMatrix, distCoeffs);
    }   
    
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
        inputVideo.open(0);
        waitTime = 10;
    }
    
    double axisLength = 0.5*(std::min(markersX, markersY)*(markerLength + markerSeparation) + markerSeparation);


    cv::aruco::GridBoard board = cv::aruco::GridBoard::create(markersX, markersY, markerLength, 
                                                              markerSeparation, dictionary);
//     board.ids.clear();
//     board.ids.push_back(985);
//     board.ids.push_back(838);
//     board.ids.push_back(908);
//     board.ids.push_back(299);
//     board.ids.push_back(428);
//     board.ids.push_back(177);
// 
//     board.ids.push_back(64);
//     board.ids.push_back(341);
//     board.ids.push_back(760);
//     board.ids.push_back(882);
//     board.ids.push_back(982);
//     board.ids.push_back(977);
// 
//     board.ids.push_back(477);
//     board.ids.push_back(125);
//     board.ids.push_back(717);
//     board.ids.push_back(791);
//     board.ids.push_back(618);
//     board.ids.push_back(76);
// 
//     board.ids.push_back(181);
//     board.ids.push_back(1005);
//     board.ids.push_back(175);
//     board.ids.push_back(684);
//     board.ids.push_back(233);
//     board.ids.push_back(461);

    while (inputVideo.grab()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > corners, rejected;
        cv::Mat rvec, tvec;

        // detect markers and estimate pose
        cv::aruco::detectMarkers(image, cv::aruco::DICT_ARUCO, corners, ids, detectorParams,
                                 rejected);
        int markersOfBoardDetected = 0;
        if (ids.size() > 0)
            markersOfBoardDetected = cv::aruco::estimatePoseBoard(corners, ids, board, camMatrix, 
                                                                  distCoeffs, rvec, tvec);

        // draw results
        image.copyTo(imageCopy);
        if (ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(imageCopy, imageCopy, corners, ids);
        }
        
        if (showRejected && rejected.size() > 0)
            cv::aruco::drawDetectedMarkers(imageCopy, imageCopy, rejected, 
                                               cv::noArray(), cv::Scalar(100, 0, 255));           
        
        if (markersOfBoardDetected > 0)
            cv::aruco::drawAxis(imageCopy, imageCopy, camMatrix, distCoeffs, rvec, tvec, axisLength);

        cv::imshow("out", imageCopy);
        char key = cv::waitKey(waitTime);
        if (key == 27)
            break;
    }

    return 0;
}
