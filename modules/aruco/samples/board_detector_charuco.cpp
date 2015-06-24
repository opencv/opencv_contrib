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
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;


/**
 */
static void help() {
    std::cout << "Pose estimation using a ChArUco board" << std::endl;
    std::cout << "Parameters: " << std::endl;
    std::cout << "-w <nmarkers> # Number of markers in X direction" << std::endl;
    std::cout << "-h <nsquares> # Number of squares in Y direction" << std::endl;
    std::cout << "-sl <squareLength> # Square side lenght (in meters)" << std::endl;
    std::cout << "-ml <markerLength> # Marker side lenght (in meters)" << std::endl;
    std::cout << "-d <dictionary> # 0: ARUCO, ..." << std::endl;
    std::cout << "-c <cameraParams> # Camera intrinsic parameters file"
              << std::endl;    
    std::cout << "[-v <videoFile>] # Input from video file, if ommited, input comes from camera"
                 << std::endl;
    std::cout << "[-ci <int>] # Camera id if input doesnt come from video (-v). Default is 0"
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

    if (!isParam("-w", argc, argv) || !isParam("-h", argc, argv) || !isParam("-sl", argc, argv) ||
        !isParam("-ml", argc, argv) || !isParam("-d", argc, argv) || !isParam("-c", argc, argv) ) {
        help();
        return 0;
    }

    int squaresX = atoi( getParam("-w", argc, argv).c_str() );
    int squaresY = atoi( getParam("-h", argc, argv).c_str() );
    float squareLength = atof( getParam("-sl", argc, argv).c_str() );
    float markerLength = atof( getParam("-ml", argc, argv).c_str() );
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
        int camId = 0;
        if (isParam("-ci", argc, argv)) 
            camId = atoi( getParam("-ci", argc, argv).c_str() );
        inputVideo.open(camId);
        waitTime = 10;
    }
    
    double axisLength = 0.5*(std::min(squaresX, squaresY) * (squareLength));


    cv::aruco::CharucoBoard board = cv::aruco::CharucoBoard::create(squaresX, squaresY, 
                                                                    squareLength, markerLength,
                                                                    dictionary);

    double totalTime = 0;
    int totalIterations = 0;        
    
    int interpolationMethod = 0;
    
    while (inputVideo.grab()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);

        double tick = (double)cv::getTickCount();            
        
        std::vector<int> markerIds, charucoIds;
        std::vector<std::vector<cv::Point2f> > markerCorners, rejectedMarkers;
        std::vector<cv::Point2f> charucoCorners;
        cv::Mat rvec, tvec;

        // detect markers and estimate pose
        cv::aruco::detectMarkers(image, dictionary, markerCorners, markerIds, detectorParams, 
                                 rejectedMarkers);
 
        
        int interpolatedCorners = 0;
        if (markerIds.size() > 0 && interpolationMethod == 0)
            interpolatedCorners = cv::aruco::interpolateCornersCharucoApproxCalib(markerCorners, 
                                                                                  markerIds, image,
                                                                                  board, camMatrix,
                                                                                  distCoeffs, 
                                                                                  charucoCorners,
                                                                                  charucoIds);

            
        if (markerIds.size() > 0 && interpolationMethod == 1)            
            interpolatedCorners = cv::aruco::interpolateCornersCharucoGlobalHom(markerCorners, 
                                                                                markerIds, image,
                                                                                board,
                                                                                charucoCorners,
                                                                                charucoIds);
            
        if (markerIds.size() > 0 && interpolationMethod == 2)            
            interpolatedCorners = cv::aruco::interpolateCornersCharucoLocalHom(markerCorners, 
                                                                               markerIds, image,
                                                                               board,
                                                                               charucoCorners,
                                                                               charucoIds);            
            
            
        bool validPose;
        validPose = cv::aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, board, 
                                                        camMatrix, distCoeffs, rvec, tvec);
   

        
        double currentTime = ((double)cv::getTickCount()-tick)/cv::getTickFrequency();
        totalTime += currentTime;
        totalIterations++;
        if(totalIterations%30 == 0) {
            std::cout << "Detection Time = " << currentTime*1000 << " ms " <<
                         "(Mean = " << 1000*totalTime/double(totalIterations) << " ms)" << std::endl;            
        }
        
        
        
        // draw results
        image.copyTo(imageCopy);
        if (markerIds.size() > 0) {
            cv::aruco::drawDetectedMarkers(imageCopy, imageCopy, markerCorners);
        }
        
        if (showRejected && rejectedMarkers.size() > 0)
            cv::aruco::drawDetectedMarkers(imageCopy, imageCopy, rejectedMarkers, 
                                           cv::noArray(), cv::Scalar(100, 0, 255));           
        
        if (interpolatedCorners > 0) {
            cv::Scalar color1, color2;
            stringstream s;
            if (interpolationMethod == 0) {
                color1 = cv::Scalar(0, 0, 255);
                color2 = color1;
                s << "Approximated Calibration";
            }
            else if (interpolationMethod == 1) {
                color1 = cv::Scalar(0, 255, 0);
                color2 = cv::Scalar(255, 0, 0);   
                s << "Global Homography";       
            }
            else {
                color1 = cv::Scalar(255, 0, 0);   
                color2 = cv::Scalar(0, 255, 0); 
                s << "Local Homography";    
            }
            cv::putText(imageCopy, s.str(), Point2f(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    color1, 2);            
            cv::aruco::drawDetectedCornersCharuco(imageCopy, imageCopy, charucoCorners, charucoIds,
                                                  color2);
        }
        
        if (validPose) 
            cv::aruco::drawAxis(imageCopy, imageCopy, camMatrix, distCoeffs, rvec, tvec, 
                                axisLength);        

        cv::imshow("out", imageCopy);
        char key = cv::waitKey(waitTime);
        if (key == 27)
            break;
        if (key == 'i') {
            interpolationMethod = (interpolationMethod+1)%3;
            totalTime = 0;
            totalIterations = 0;
            std::cout << "Interpolation method: " << interpolationMethod << std::endl;
        }
    }

    return 0;
}
 
