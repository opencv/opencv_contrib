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
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;


/**
 */
static void help() {
    std::cout << "Calibration using a ChArUco board" << std::endl;
    std::cout << "How to Use:" << std::endl;
    std::cout << "To capture a frame for calibration, press 'c'," << std::endl;
    std::cout << "If input comes from video, press any key for next frame" << std::endl;
    std::cout << "To finish capturing, press 'ESC' key and calibration starts." << std::endl;
    std::cout << "Parameters: " << std::endl;
    std::cout << "-w <nmarkers> # Number of markers in X direction" << std::endl;
    std::cout << "-h <nsquares> # Number of squares in Y direction" << std::endl;
    std::cout << "-sl <squareLength> # Square side lenght (in meters)" << std::endl;
    std::cout << "-ml <markerLength> # Marker side lenght (in meters)" << std::endl;
    std::cout << "-d <dictionary> # 0: ARUCO, ..." << std::endl;
    std::cout << "-o <outputFile> # Output file with calibrated camera parameters" << std::endl;
    std::cout << "[-v <videoFile>] # Input from video file, if ommited, input comes from camera"
                 << std::endl;
    std::cout << "[-ci <int>] # Camera id if input doesnt come from video (-v). Default is 0"
                 << std::endl;                 
    std::cout << "[-dp <detectorParams>] # File of marker detector parameters" << std::endl;
    std::cout << "[-zt] # Assume zero tangential distortion" << std::endl;
    std::cout << "[-a <aspectRatio>] # Fix aspect ratio (fx/fy)" << std::endl;
    std::cout << "[-p] # Fix the principal point at the center" << std::endl;
    std::cout << "[-sc] # Show detected chessboard corners after calibration" << std::endl;    
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
static void saveCameraParams(const string& filename,
                             cv::Size imageSize, float aspectRatio, int flags,
                             const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
                             double totalAvgErr ) {
    cv::FileStorage fs( filename, cv::FileStorage::WRITE );

    time_t tt;
    time( &tt );
    struct tm *t2 = localtime( &tt );
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", t2 );

    fs << "calibration_time" << buf;

    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;

    if ( flags & cv::CALIB_FIX_ASPECT_RATIO )
        fs << "aspectRatio" << aspectRatio;

    if ( flags != 0 ) {
        sprintf( buf, "flags: %s%s%s%s",
        flags & cv::CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
        flags & cv::CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
        flags & cv::CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
        flags & cv::CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "" );
    }

    fs << "flags" << flags;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;
}



/**
 */
int main(int argc, char *argv[]) {

    if (!isParam("-w", argc, argv) || !isParam("-h", argc, argv) || !isParam("-sl", argc, argv) ||
        !isParam("-ml", argc, argv) || !isParam("-d", argc, argv) || !isParam("-o", argc, argv) ) {
        help();
        return 0;
    }

    int squaresX = atoi( getParam("-w", argc, argv).c_str() );
    int squaresY = atoi( getParam("-h", argc, argv).c_str() );
    float squareLength = atof( getParam("-sl", argc, argv).c_str() );
    float markerLength = atof( getParam("-ml", argc, argv).c_str() );
    int dictionaryId = atoi( getParam("-d", argc, argv).c_str() );
    cv::aruco::DICTIONARY dictionary = cv::aruco::DICTIONARY(dictionaryId);
    string outputFile = getParam("-o", argc, argv);
    
    bool showChessboardCorners = isParam("-sc", argc, argv);
    
    int calibrationFlags = 0;
    float aspectRatio = 1;
    if (isParam("-a", argc, argv)) {
        calibrationFlags |= cv::CALIB_FIX_ASPECT_RATIO;
        aspectRatio = atof( getParam("-a", argc, argv).c_str() );
    }
    if (isParam("-zt", argc, argv))
        calibrationFlags |= cv::CALIB_ZERO_TANGENT_DIST;    
    if (isParam("-p", argc, argv))
        calibrationFlags |= cv::CALIB_FIX_PRINCIPAL_POINT;      
    
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

    cv::aruco::CharucoBoard board = cv::aruco::CharucoBoard::create(squaresX, squaresY, 
                                                                    squareLength, markerLength,
                                                                    dictionary);

    std::vector<std::vector<std::vector<cv::Point2f> > > allCorners;
    std::vector<std::vector<int> > allIds;
    std::vector<cv::Mat> allImgs;
    cv::Size imgSize;    

    while (inputVideo.grab()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > corners;

        // detect markers and estimate pose
        cv::aruco::detectMarkers(image, dictionary, corners, ids, detectorParams);

        // draw results
        image.copyTo(imageCopy);
        if (ids.size() > 0)
            cv::aruco::drawDetectedMarkers(imageCopy, imageCopy, corners, ids);

        cv::imshow("out", imageCopy);
        char key = cv::waitKey(waitTime);
        if (key == 27)
            break;
        if (key == 'c' && ids.size() > 0) {
            std::cout << "Frame captured" << std::endl;
            allCorners.push_back(corners);
            allIds.push_back(ids);
            allImgs.push_back(image);
            imgSize = image.size();            
        }
    }

    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    std::vector<cv::Mat> allChessboardCorners;
    double repError;
    
    if( calibrationFlags & CALIB_FIX_ASPECT_RATIO ) {
        cameraMatrix = Mat::eye(3, 3, CV_64F);
        cameraMatrix.at<double>(0,0) = aspectRatio;  
    }
        
    repError = cv::aruco::calibrateCameraCharuco(allCorners, allIds, allImgs, board, cameraMatrix,
                                                 distCoeffs, rvecs, tvecs, allChessboardCorners, 
                                                 calibrationFlags);

    saveCameraParams(outputFile, imgSize, aspectRatio, calibrationFlags, cameraMatrix, distCoeffs, 
                     repError );
    
    std::cout << "Rep Error: " << repError << std::endl;
    std::cout << "Calibration saved to " << outputFile << std::endl;
    
    if (showChessboardCorners) {
        for (unsigned int frame = 0; frame < allImgs.size(); frame++) {
            cv::Mat imageCopy = allImgs[frame].clone();
            if (allIds[frame].size() > 0) {
                
                cv::aruco::drawDetectedMarkers(imageCopy, imageCopy, allCorners[frame]);            

                if (allChessboardCorners[frame].total() > 0) { 
                    cv::drawChessboardCorners(imageCopy, cv::Size(squaresX-1, squaresY-1), 
                                            allChessboardCorners[frame], true);                
                }
            }
            
            cv::imshow("out", imageCopy);
            char key = cv::waitKey(0);
            if (key == 27) 
                break;
        }
    }

    return 0;
}
 
