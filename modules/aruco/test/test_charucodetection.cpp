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


#include "test_precomp.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <string>

using namespace std;
using namespace cv;


const double PI = 3.141592653589793238463;


static double deg2rad(double deg) {
    return deg*PI/180.;
}


static void getSyntheticRT(double yaw, double pitch, double distance, Mat &rvec,
                           Mat &tvec) {

    rvec = Mat(3,1,CV_64FC1);
    tvec = Mat(3,1,CV_64FC1);

    // Rvec
    // first put the Z axis aiming to -X (like the camera axis system)
    Mat rotZ(3,1,CV_64FC1);
    rotZ.ptr<double>(0)[0] = 0;
    rotZ.ptr<double>(0)[1] = 0;
    rotZ.ptr<double>(0)[2] = -0.5*PI;

    Mat rotX(3,1,CV_64FC1);
    rotX.ptr<double>(0)[0] = 0.5*PI;
    rotX.ptr<double>(0)[1] = 0;
    rotX.ptr<double>(0)[2] = 0;

    Mat camRvec, camTvec;
    composeRT(rotZ, Mat(3,1,CV_64FC1,Scalar::all(0)), rotX, Mat(3,1,CV_64FC1,Scalar::all(0)), camRvec, camTvec );

    // now pitch and yaw angles
    Mat rotPitch(3,1,CV_64FC1);
    rotPitch.ptr<double>(0)[0] = 0;
    rotPitch.ptr<double>(0)[1] = pitch;
    rotPitch.ptr<double>(0)[2] = 0;

    Mat rotYaw(3,1,CV_64FC1);
    rotYaw.ptr<double>(0)[0] = yaw;
    rotYaw.ptr<double>(0)[1] = 0;
    rotYaw.ptr<double>(0)[2] = 0;

    composeRT(rotPitch, Mat(3,1,CV_64FC1,Scalar::all(0)), rotYaw, Mat(3,1,CV_64FC1,Scalar::all(0)), rvec, tvec );

    // compose both rotations
    composeRT(camRvec, Mat(3,1,CV_64FC1,Scalar::all(0)), rvec, Mat(3,1,CV_64FC1,Scalar::all(0)), rvec, tvec );

    // Tvec, just move in z (camera) direction the specific distance
    tvec.ptr<double>(0)[0] = 0.;
    tvec.ptr<double>(0)[1] = 0.;
    tvec.ptr<double>(0)[2] = distance;

}


static void projectMarker(Mat &img, aruco::Dictionary dictionary, int id,
                          vector<Point3f> markerObjPoints, Mat cameraMatrix,
                          Mat rvec, Mat tvec, int markerBorder) {


    Mat markerImg;
    const int markerSizePixels = 100;
    aruco::drawMarker(dictionary, id, markerSizePixels, markerImg, markerBorder);

    Mat distCoeffs(5, 0, CV_64FC1, Scalar::all(0));
    vector<Point2f> corners;
    projectPoints(markerObjPoints, rvec, tvec, cameraMatrix, distCoeffs, corners);

    vector<Point2f> originalCorners;
    originalCorners.push_back( Point2f(0, 0) );
    originalCorners.push_back( Point2f((float)markerSizePixels, 0) );
    originalCorners.push_back( Point2f((float)markerSizePixels, (float)markerSizePixels) );
    originalCorners.push_back( Point2f(0, (float)markerSizePixels) );

    Mat transformation = getPerspectiveTransform(originalCorners, corners);

    Mat aux;
    const char borderValue = 127;
    warpPerspective(markerImg, aux, transformation, img.size(), INTER_NEAREST,
                        BORDER_CONSTANT, Scalar::all(borderValue));

    // copy only not-border pixels
    for (int y = 0; y < aux.rows; y++) {
        for (int x = 0; x < aux.cols; x++) {
            if (aux.at<unsigned char>(y, x) == borderValue)
                continue;
            img.at<unsigned char>(y, x) = aux.at<unsigned char>(y, x);
        }
    }

}


static Mat projectChessboard(int squaresX, int squaresY, float squareSize,
                                 Size imageSize, Mat cameraMatrix,
                                 Mat rvec, Mat tvec) {

    Mat img(imageSize, CV_8UC1, Scalar::all(255));
    Mat distCoeffs(5, 0, CV_64FC1, Scalar::all(0));

    for(int y=0; y<squaresY; y++) {
        float startY = float(y)*squareSize;
        for(int x=0; x<squaresX; x++) {
            if(y%2 != x%2) continue;
            float startX = float(x)*squareSize;

            vector<Point3f> squareCorners;
            squareCorners.push_back(Point3f(startX, startY, 0));
            squareCorners.push_back( squareCorners[0] + Point3f(squareSize, 0, 0) );
            squareCorners.push_back( squareCorners[0] + Point3f(squareSize, squareSize, 0) );
            squareCorners.push_back( squareCorners[0] + Point3f(0, squareSize, 0) );

            vector< vector<Point2f> > projectedCorners;
            projectedCorners.push_back( vector<Point2f>() );
            projectPoints(squareCorners, rvec, tvec, cameraMatrix, distCoeffs,
                              projectedCorners[0]);

            vector< vector<Point> > projectedCornersInt;
            projectedCornersInt.push_back( vector<Point>() );

            for(int k=0; k<4; k++)
                projectedCornersInt[0].push_back(Point((int)projectedCorners[0][k].x,
                                                           (int)projectedCorners[0][k].y));

            fillPoly(img, projectedCornersInt, Scalar::all(0));

        }
    }

    return img;

}



static Mat projectCharucoBoard(aruco::CharucoBoard board, Mat cameraMatrix,
                                   double yaw, double pitch, double distance, Size imageSize,
                                   int markerBorder, Mat &rvec, Mat &tvec) {

    getSyntheticRT(yaw, pitch, distance, rvec, tvec);

    Mat img = Mat(imageSize, CV_8UC1, Scalar::all(255));
    for(unsigned int m=0; m<board.ids.size(); m++) {
        projectMarker(img, board.dictionary, board.ids[m], board.objPoints[m], cameraMatrix,
                      rvec, tvec, markerBorder);
    }

    Mat chessboard = projectChessboard(board.getChessboardSize().width,
                                           board.getChessboardSize().height,
                                           board.getSquareLength(), imageSize, cameraMatrix,
                                           rvec, tvec);

    for(unsigned int i=0; i<chessboard.total(); i++) {
        if(chessboard.ptr<unsigned char>()[i] == 0)  {
            img.ptr<unsigned char>()[i] = 0;
        }
    }

    return img;

}




class CV_CharucoDetection : public cvtest::BaseTest
{
public:
    CV_CharucoDetection();
protected:
    void run(int);
};




CV_CharucoDetection::CV_CharucoDetection() {}


void CV_CharucoDetection::run(int) {

    int iter = 0;
    Mat cameraMatrix = Mat::eye(3,3, CV_64FC1);
    Size imgSize(500,500);
    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::CharucoBoard board = aruco::CharucoBoard::create(4, 4, 0.03f, 0.015f, dictionary);

    cameraMatrix.at<double>(0,0) = cameraMatrix.at<double>(1,1) = 650;
    cameraMatrix.at<double>(0,2) = imgSize.width / 2;
    cameraMatrix.at<double>(1,2) = imgSize.height / 2;

    Mat distCoeffs(5, 0, CV_64FC1, Scalar::all(0));

    for(double distance = 0.2; distance <= 0.4; distance += 0.1) {
        for(int yaw = 0; yaw < 360; yaw+=20) {
            for(int pitch = 30; pitch <=90; pitch+=20) {

                int markerBorder = iter%2+1;
                iter ++;

                Mat rvec, tvec;
                Mat img = projectCharucoBoard(board, cameraMatrix, deg2rad(pitch), deg2rad(yaw),
                                                  distance, imgSize, markerBorder, rvec, tvec);


                vector< vector<Point2f> > corners;
                vector< int > ids;
                aruco::DetectorParameters params;
                params.minDistanceToBorder = 3;
                params.doCornerRefinement = false;
                params.markerBorderBits = markerBorder;
                aruco::detectMarkers(img, dictionary, corners, ids, params);

                if(ids.size()==0) {
                    ts->printf( cvtest::TS::LOG, "Marker detection failed" );
                    ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                    return;
                }

                vector<Point2f> charucoCorners;
                vector<int> charucoIds;

                if(iter%2 == 0) {
                    aruco::interpolateCornersCharuco(corners, ids, img, board, charucoCorners,
                                                         charucoIds);
                }
                else {
                    aruco::interpolateCornersCharuco(corners, ids, img, board, charucoCorners,
                                                         charucoIds, cameraMatrix, distCoeffs);
                }


                vector<Point2f> projectedCharucoCorners;
                projectPoints(board.chessboardCorners, rvec, tvec, cameraMatrix, distCoeffs,
                                  projectedCharucoCorners);

                for(unsigned int i=0; i<charucoIds.size(); i++) {

                    int currentId = charucoIds[i];

                    if(currentId >= (int)board.chessboardCorners.size() ) {
                        ts->printf(cvtest::TS::LOG, "Invalid Charuco corner id" );
                        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                        return;
                    }

                    double repError = norm(charucoCorners[i] -
                                               projectedCharucoCorners[currentId]);


                    if(repError > 5.) {
                        ts->printf(cvtest::TS::LOG, "Charuco corner reprojection error too high" );
                        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                        return;
                    }

                }

            }
        }
    }


}





class CV_CharucoPoseEstimation : public cvtest::BaseTest
{
public:
    CV_CharucoPoseEstimation();
protected:
    void run(int);
};




CV_CharucoPoseEstimation::CV_CharucoPoseEstimation() {}


void CV_CharucoPoseEstimation::run(int) {

    int iter = 0;
    Mat cameraMatrix = Mat::eye(3,3, CV_64FC1);
    Size imgSize(500,500);
    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::CharucoBoard board = aruco::CharucoBoard::create(4, 4, 0.03f, 0.015f, dictionary);

    cameraMatrix.at<double>(0,0) = cameraMatrix.at<double>(1,1) = 650;
    cameraMatrix.at<double>(0,2) = imgSize.width / 2;
    cameraMatrix.at<double>(1,2) = imgSize.height / 2;

    Mat distCoeffs(5, 0, CV_64FC1, Scalar::all(0));

    for(double distance = 0.2; distance <= 0.4; distance += 0.1) {
        for(int yaw = 0; yaw < 360; yaw+=20) {
            for(int pitch = 30; pitch <=90; pitch+=20) {

                int markerBorder = iter%2+1;
                iter ++;

                Mat rvec, tvec;
                Mat img = projectCharucoBoard(board, cameraMatrix, deg2rad(pitch), deg2rad(yaw),
                                                  distance, imgSize, markerBorder, rvec, tvec);


                vector< vector<Point2f> > corners;
                vector< int > ids;
                aruco::DetectorParameters params;
                params.minDistanceToBorder = 3;
                params.doCornerRefinement = false;
                params.markerBorderBits = markerBorder;
                aruco::detectMarkers(img, dictionary, corners, ids, params);

                if(ids.size()==0) {
                    ts->printf( cvtest::TS::LOG, "Marker detection failed" );
                    ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                    return;
                }

                vector<Point2f> charucoCorners;
                vector<int> charucoIds;

                if(iter%2 == 0) {
                    aruco::interpolateCornersCharuco(corners, ids, img, board, charucoCorners,
                                                         charucoIds);
                }
                else {
                    aruco::interpolateCornersCharuco(corners, ids, img, board, charucoCorners,
                                                         charucoIds, cameraMatrix, distCoeffs);
                }

                if(charucoIds.size() == 0)
                    continue;

                aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, board,
                                                    cameraMatrix, distCoeffs, rvec, tvec);


                vector<Point2f> projectedCharucoCorners;
                projectPoints(board.chessboardCorners, rvec, tvec, cameraMatrix, distCoeffs,
                                  projectedCharucoCorners);

                for(unsigned int i=0; i<charucoIds.size(); i++) {

                    int currentId = charucoIds[i];

                    if(currentId >= (int)board.chessboardCorners.size() ) {
                        ts->printf(cvtest::TS::LOG, "Invalid Charuco corner id" );
                        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                        return;
                    }

                    double repError = norm(charucoCorners[i] -
                                               projectedCharucoCorners[currentId]);


                    if(repError > 5.) {
                        ts->printf(cvtest::TS::LOG, "Charuco corner reprojection error too high" );
                        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                        return;
                    }

                }

            }
        }
    }


}





TEST(CV_CharucoDetection, accuracy) {
    CV_CharucoDetection test;
    test.safe_run();
}


TEST(CV_CharucoPoseEstimation, accuracy) {
    CV_CharucoPoseEstimation test;
    test.safe_run();
}
