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
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;


/**
 */
static void help() {
    cout << "Detect ChArUco markers" << endl;
    cout << "Parameters: " << endl;
    cout << "-sl <squareLength> # Square side lenght (in meters)" << endl;
    cout << "-ml <markerLength> # Marker side lenght (in meters)" << endl;
    cout << "-d <dictionary> # DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2, "
         << "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
         << "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
         << "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16" << endl;
    cout << "[-c <cameraParams>] # Camera intrinsic parameters file" << endl;
    cout << "[-as <float>] # Automatic scale. The provided number is multiplied by the last "
         << "diamond id becoming an indicator of the square length. In this case, the -sl and "
         << "-ml are only used to know the relative length relation between "
         << "squares and markers" << endl;
    cout << "[-v <videoFile>] # Input from video file, if ommited, input comes from camera" << endl;
    cout << "[-ci <int>] # Camera id if input doesnt come from video (-v). Default is 0" << endl;
    cout << "[-dp <detectorParams>] # File of marker detector parameters" << endl;
    cout << "[-r] # show rejected candidates too" << endl;
}


/**
 */
static bool isParam(string param, int argc, char **argv) {
    for(int i = 0; i < argc; i++)
        if(string(argv[i]) == param) return true;
    return false;
}


/**
 */
static string getParam(string param, int argc, char **argv, string defvalue = "") {
    int idx = -1;
    for(int i = 0; i < argc && idx == -1; i++)
        if(string(argv[i]) == param) idx = i;
    if(idx == -1 || (idx + 1) >= argc)
        return defvalue;
    else
        return argv[idx + 1];
}


/**
 */
static bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}


/**
 */
static bool readDetectorParameters(string filename, aruco::DetectorParameters &params) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["adaptiveThreshWinSizeMin"] >> params.adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params.adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params.adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params.adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params.minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params.maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params.polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params.minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params.minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params.minMarkerDistanceRate;
    fs["doCornerRefinement"] >> params.doCornerRefinement;
    fs["cornerRefinementWinSize"] >> params.cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params.cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params.cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params.markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params.perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params.perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params.maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params.minOtsuStdDev;
    fs["errorCorrectionRate"] >> params.errorCorrectionRate;
    return true;
}


/**
 */
int main(int argc, char *argv[]) {

    if(!isParam("-sl", argc, argv) || !isParam("-ml", argc, argv) || !isParam("-d", argc, argv)) {
        help();
        return 0;
    }

    float squareLength = (float)atof(getParam("-sl", argc, argv).c_str());
    float markerLength = (float)atof(getParam("-ml", argc, argv).c_str());
    int dictionaryId = atoi(getParam("-d", argc, argv).c_str());
    aruco::Dictionary dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    bool showRejected = false;
    if(isParam("-r", argc, argv)) showRejected = true;

    bool estimatePose = false;
    Mat camMatrix, distCoeffs;
    if(isParam("-c", argc, argv)) {
        bool readOk = readCameraParameters(getParam("-c", argc, argv), camMatrix, distCoeffs);
        if(!readOk) {
            cerr << "Invalid camera file" << endl;
            return 0;
        }
        estimatePose = true;
    }

    bool autoScale = false;
    float autoScaleFactor = 1.;
    if(isParam("-as", argc, argv)) {
        autoScaleFactor = (float)atof(getParam("-as", argc, argv).c_str());
        autoScale = true;
    }

    aruco::DetectorParameters detectorParams;
    if(isParam("-dp", argc, argv)) {
        bool readOk = readDetectorParameters(getParam("-dp", argc, argv), detectorParams);
        if(!readOk) {
            cerr << "Invalid detector parameters file" << endl;
            return 0;
        }
    }

    VideoCapture inputVideo;
    int waitTime;
    if(isParam("-v", argc, argv)) {
        inputVideo.open(getParam("-v", argc, argv));
        waitTime = 0;
    } else {
        int camId = 0;
        if(isParam("-ci", argc, argv)) camId = atoi(getParam("-ci", argc, argv).c_str());
        inputVideo.open(camId);
        waitTime = 10;
    }

    double totalTime = 0;
    int totalIterations = 0;

    while(inputVideo.grab()) {
        Mat image, imageCopy;
        inputVideo.retrieve(image);

        double tick = (double)getTickCount();

        vector< int > markerIds;
        vector< Vec4i > diamondIds;
        vector< vector< Point2f > > markerCorners, rejectedMarkers, diamondCorners;
        vector< Mat > rvecs, tvecs;

        // detect markers
        aruco::detectMarkers(image, dictionary, markerCorners, markerIds, detectorParams,
                             rejectedMarkers);

        // detect diamonds
        if(markerIds.size() > 0)
            aruco::detectCharucoDiamond(image, markerCorners, markerIds,
                                        squareLength / markerLength, diamondCorners, diamondIds,
                                        camMatrix, distCoeffs);

        // estimate diamond pose
        if(estimatePose && diamondIds.size() > 0) {
            if(!autoScale) {
                aruco::estimatePoseSingleMarkers(diamondCorners, squareLength, camMatrix,
                                                 distCoeffs, rvecs, tvecs);
            } else {
                // if autoscale, extract square size from last diamond id
                for(unsigned int i = 0; i < diamondCorners.size(); i++) {
                    float autoSquareLength = autoScaleFactor * float(diamondIds[i].val[3]);
                    vector< vector< Point2f > > currentCorners;
                    vector< Mat > currentRvec, currentTvec;
                    currentCorners.push_back(diamondCorners[i]);
                    aruco::estimatePoseSingleMarkers(currentCorners, autoSquareLength, camMatrix,
                                                     distCoeffs, currentRvec, currentTvec);
                    rvecs.push_back(currentRvec[0]);
                    tvecs.push_back(currentTvec[0]);
                }
            }
        }


        double currentTime = ((double)getTickCount() - tick) / getTickFrequency();
        totalTime += currentTime;
        totalIterations++;
        if(totalIterations % 30 == 0) {
            cout << "Detection Time = " << currentTime * 1000 << " ms "
                 << "(Mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << endl;
        }


        // draw results
        image.copyTo(imageCopy);
        if(markerIds.size() > 0)
            aruco::drawDetectedMarkers(imageCopy, markerCorners);


        if(showRejected && rejectedMarkers.size() > 0)
            aruco::drawDetectedMarkers(imageCopy, rejectedMarkers, noArray(), Scalar(100, 0, 255));

        if(diamondIds.size() > 0) {
            aruco::drawDetectedDiamonds(imageCopy, diamondCorners, diamondIds);

            if(estimatePose) {
                for(unsigned int i = 0; i < diamondIds.size(); i++)
                    aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvecs[i], tvecs[i],
                                    squareLength * 0.5f);
            }
        }

        imshow("out", imageCopy);
        char key = (char)waitKey(waitTime);
        if(key == 27) break;
    }

    return 0;
}
