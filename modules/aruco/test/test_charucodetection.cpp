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


const double PI = 3.141592653589793238463;


static double deg2rad(double deg) {
    return deg*PI/180.;
}


static void getSyntheticRT(double yaw, double pitch, double distance, cv::Mat &rvec,
                           cv::Mat &tvec) {

    rvec = cv::Mat(3,1,CV_64FC1);
    tvec = cv::Mat(3,1,CV_64FC1);

    // Rvec
    // first put the Z axis aiming to -X (like the camera axis system)
    cv::Mat rotZ(3,1,CV_64FC1);
    rotZ.ptr<double>(0)[0] = 0;
    rotZ.ptr<double>(0)[1] = 0;
    rotZ.ptr<double>(0)[2] = -0.5*PI;

    cv::Mat rotX(3,1,CV_64FC1);
    rotX.ptr<double>(0)[0] = 0.5*PI;
    rotX.ptr<double>(0)[1] = 0;
    rotX.ptr<double>(0)[2] = 0;

    cv::Mat camRvec, camTvec;
    cv::composeRT(rotZ, cv::Mat(3,1,CV_64FC1,cv::Scalar::all(0)), rotX, cv::Mat(3,1,CV_64FC1,cv::Scalar::all(0)), camRvec, camTvec );

    // now pitch and yaw angles
    cv::Mat rotPitch(3,1,CV_64FC1);
    rotPitch.ptr<double>(0)[0] = 0;
    rotPitch.ptr<double>(0)[1] = pitch;
    rotPitch.ptr<double>(0)[2] = 0;

    cv::Mat rotYaw(3,1,CV_64FC1);
    rotYaw.ptr<double>(0)[0] = yaw;
    rotYaw.ptr<double>(0)[1] = 0;
    rotYaw.ptr<double>(0)[2] = 0;

    cv::composeRT(rotPitch, cv::Mat(3,1,CV_64FC1,cv::Scalar::all(0)), rotYaw, cv::Mat(3,1,CV_64FC1,cv::Scalar::all(0)), rvec, tvec );

    // compose both rotations
    cv::composeRT(camRvec, cv::Mat(3,1,CV_64FC1,cv::Scalar::all(0)), rvec, cv::Mat(3,1,CV_64FC1,cv::Scalar::all(0)), rvec, tvec );

    // Tvec, just move in z (camera) direction the specific distance
    tvec.ptr<double>(0)[0] = 0.;
    tvec.ptr<double>(0)[1] = 0.;
    tvec.ptr<double>(0)[2] = distance;

}


static void projectMarker(cv::Mat &img, cv::aruco::DICTIONARY dictionary, int id,
                          std::vector<cv::Point3f> markerObjPoints, cv::Mat cameraMatrix,
                          cv::Mat rvec, cv::Mat tvec, int markerBorder) {


    cv::Mat markerImg;
    const int markerSizePixels = 100;
    cv::aruco::drawMarker(dictionary, id, markerSizePixels, markerImg, markerBorder);

    cv::Mat distCoeffs(5, 0, CV_64FC1, cv::Scalar::all(0));
    std::vector<cv::Point2f> corners;
    cv::projectPoints(markerObjPoints, rvec, tvec, cameraMatrix, distCoeffs, corners);

    std::vector<cv::Point2f> originalCorners;
    originalCorners.push_back( cv::Point2f(0, 0) );
    originalCorners.push_back( cv::Point2f((float)markerSizePixels, 0) );
    originalCorners.push_back( cv::Point2f((float)markerSizePixels, (float)markerSizePixels) );
    originalCorners.push_back( cv::Point2f(0, (float)markerSizePixels) );

    cv::Mat transformation = cv::getPerspectiveTransform(originalCorners, corners);

    cv::Mat aux;
    const char borderValue = 127;
    cv::warpPerspective(markerImg, aux, transformation, img.size(), cv::INTER_NEAREST,
                        cv::BORDER_CONSTANT, cv::Scalar::all(borderValue));

    // copy only not-border pixels
    for (int y = 0; y < aux.rows; y++) {
        for (int x = 0; x < aux.cols; x++) {
            if (aux.at<unsigned char>(y, x) == borderValue)
                continue;
            img.at<unsigned char>(y, x) = aux.at<unsigned char>(y, x);
        }
    }

}


static cv::Mat projectChessboard(int squaresX, int squaresY, double squareSize,
                                 cv::Size imageSize, cv::Mat cameraMatrix,
                                 cv::Mat rvec, cv::Mat tvec) {

    cv::Mat img(imageSize, CV_8UC1, cv::Scalar::all(255));
    cv::Mat distCoeffs(5, 0, CV_64FC1, cv::Scalar::all(0));

    for(int y=0; y<squaresY; y++) {
        double startY = double(y)*squareSize;
        for(int x=0; x<squaresX; x++) {
            if(y%2 != x%2) continue;
            double startX = double(x)*squareSize;

            std::vector<cv::Point3f> squareCorners;
            squareCorners.push_back(cv::Point3f(startX, startY, 0));
            squareCorners.push_back( squareCorners[0] + cv::Point3f(squareSize, 0, 0) );
            squareCorners.push_back( squareCorners[0] + cv::Point3f(squareSize, squareSize, 0) );
            squareCorners.push_back( squareCorners[0] + cv::Point3f(0, squareSize, 0) );

            std::vector< std::vector<cv::Point2f> > projectedCorners;
            projectedCorners.push_back( std::vector<cv::Point2f>() );
            cv::projectPoints(squareCorners, rvec, tvec, cameraMatrix, distCoeffs,
                              projectedCorners[0]);

            std::vector< std::vector<cv::Point> > projectedCornersInt;
            projectedCornersInt.push_back( std::vector<cv::Point>() );

            for(int k=0; k<4; k++)
                projectedCornersInt[0].push_back(cv::Point(projectedCorners[0][k].x,
                                                           projectedCorners[0][k].y));

            cv::fillPoly(img, projectedCornersInt, cv::Scalar::all(0));

        }
    }

    return img;

}



static cv::Mat projectCharucoBoard(cv::aruco::CharucoBoard board, cv::Mat cameraMatrix,
                                   double yaw, double pitch, double distance, cv::Size imageSize,
                                   int markerBorder, cv::Mat &rvec, cv::Mat &tvec) {

    getSyntheticRT(yaw, pitch, distance, rvec, tvec);

    cv::Mat img = cv::Mat(imageSize, CV_8UC1, cv::Scalar::all(255));
    for(int m=0; m<board.ids.size(); m++) {
        projectMarker(img, board.dictionary, board.ids[m], board.objPoints[m], cameraMatrix,
                      rvec, tvec, markerBorder);
    }

    cv::Mat chessboard = projectChessboard(board.getChessboardSize().width,
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
    cv::Mat cameraMatrix = cv::Mat::eye(3,3, CV_64FC1);
    cv::Size imgSize(500,500);
    cv::aruco::CharucoBoard board = cv::aruco::CharucoBoard::create(4, 4, 0.03, 0.015,
                                                                    cv::aruco::DICT_6X6_250);

    cameraMatrix.at<double>(0,0) = cameraMatrix.at<double>(1,1) = 650;
    cameraMatrix.at<double>(0,2) = imgSize.width / 2;
    cameraMatrix.at<double>(1,2) = imgSize.height / 2;

    cv::Mat distCoeffs(5, 0, CV_64FC1, cv::Scalar::all(0));

    for(double distance = 0.2; distance <= 0.4; distance += 0.1) {
        for(int yaw = 0; yaw < 360; yaw+=20) {
            for(int pitch = 30; pitch <=90; pitch+=20) {

                int markerBorder = iter%2+1;
                iter ++;

                cv::Mat rvec, tvec;
                cv::Mat img = projectCharucoBoard(board, cameraMatrix, deg2rad(pitch), deg2rad(yaw),
                                                  distance, imgSize, markerBorder, rvec, tvec);


                std::vector< std::vector<cv::Point2f> > corners;
                std::vector< int > ids;
                cv::aruco::DetectorParameters params;
                params.minDistanceToBorder = 3;
                params.doCornerRefinement = false;
                params.markerBorderBits = markerBorder;
                cv::aruco::detectMarkers(img, cv::aruco::DICT_6X6_250, corners, ids, params);

                if(ids.size()==0) {
                    ts->printf( cvtest::TS::LOG, "Marker detection failed" );
                    ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                    return;
                }

                std::vector<cv::Point2f> charucoCorners;
                std::vector<int> charucoIds;

                if(iter%2 == 0) {
                    cv::aruco::interpolateCornersCharuco(corners, ids, img, board, charucoCorners,
                                                         charucoIds);
                }
                else {
                    cv::aruco::interpolateCornersCharuco(corners, ids, img, board, charucoCorners,
                                                         charucoIds, cameraMatrix, distCoeffs);
                }


                std::vector<cv::Point2f> projectedCharucoCorners;
                cv::projectPoints(board.chessboardCorners, rvec, tvec, cameraMatrix, distCoeffs,
                                  projectedCharucoCorners);

                for(unsigned int i=0; i<charucoIds.size(); i++) {

                    int currentId = charucoIds[i];

                    if(currentId >= board.chessboardCorners.size() ) {
                        ts->printf(cvtest::TS::LOG, "Invalid Charuco corner id" );
                        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                        return;
                    }

                    double repError = cv::norm(charucoCorners[i] -
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
    cv::Mat cameraMatrix = cv::Mat::eye(3,3, CV_64FC1);
    cv::Size imgSize(500,500);
    cv::aruco::CharucoBoard board = cv::aruco::CharucoBoard::create(4, 4, 0.03, 0.015,
                                                                    cv::aruco::DICT_6X6_250);

    cameraMatrix.at<double>(0,0) = cameraMatrix.at<double>(1,1) = 650;
    cameraMatrix.at<double>(0,2) = imgSize.width / 2;
    cameraMatrix.at<double>(1,2) = imgSize.height / 2;

    cv::Mat distCoeffs(5, 0, CV_64FC1, cv::Scalar::all(0));

    for(double distance = 0.2; distance <= 0.4; distance += 0.1) {
        for(int yaw = 0; yaw < 360; yaw+=20) {
            for(int pitch = 30; pitch <=90; pitch+=20) {

                int markerBorder = iter%2+1;
                iter ++;

                cv::Mat rvec, tvec;
                cv::Mat img = projectCharucoBoard(board, cameraMatrix, deg2rad(pitch), deg2rad(yaw),
                                                  distance, imgSize, markerBorder, rvec, tvec);


                std::vector< std::vector<cv::Point2f> > corners;
                std::vector< int > ids;
                cv::aruco::DetectorParameters params;
                params.minDistanceToBorder = 3;
                params.doCornerRefinement = false;
                params.markerBorderBits = markerBorder;
                cv::aruco::detectMarkers(img, cv::aruco::DICT_6X6_250, corners, ids, params);

                if(ids.size()==0) {
                    ts->printf( cvtest::TS::LOG, "Marker detection failed" );
                    ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                    return;
                }

                std::vector<cv::Point2f> charucoCorners;
                std::vector<int> charucoIds;

                if(iter%2 == 0) {
                    cv::aruco::interpolateCornersCharuco(corners, ids, img, board, charucoCorners,
                                                         charucoIds);
                }
                else {
                    cv::aruco::interpolateCornersCharuco(corners, ids, img, board, charucoCorners,
                                                         charucoIds, cameraMatrix, distCoeffs);
                }

                if(charucoIds.size() == 0)
                    continue;

                cv::aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, board,
                                                    cameraMatrix, distCoeffs, rvec, tvec);


                std::vector<cv::Point2f> projectedCharucoCorners;
                cv::projectPoints(board.chessboardCorners, rvec, tvec, cameraMatrix, distCoeffs,
                                  projectedCharucoCorners);

                for(unsigned int i=0; i<charucoIds.size(); i++) {

                    int currentId = charucoIds[i];

                    if(currentId >= board.chessboardCorners.size() ) {
                        ts->printf(cvtest::TS::LOG, "Invalid Charuco corner id" );
                        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                        return;
                    }

                    double repError = cv::norm(charucoCorners[i] -
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
