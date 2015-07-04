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
#include <string>
#include <opencv2/aruco.hpp>

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



static cv::Mat projectBoard(cv::aruco::GridBoard board, cv::Mat cameraMatrix,
                            double yaw, double pitch, double distance, cv::Size imageSize,
                            int markerBorder) {

    cv::Mat rvec, tvec;
    getSyntheticRT(yaw, pitch, distance, rvec, tvec);

    cv::Mat img = cv::Mat(imageSize, CV_8UC1, cv::Scalar::all(255));
    for(unsigned int m=0; m<board.ids.size(); m++) {
        projectMarker(img, board.dictionary, board.ids[m], board.objPoints[m], cameraMatrix,
                      rvec, tvec, markerBorder);
    }

    return img;

}




class CV_ArucoBoardPose : public cvtest::BaseTest
{
public:
    CV_ArucoBoardPose();
protected:
    void run(int);
};




CV_ArucoBoardPose::CV_ArucoBoardPose() {}


void CV_ArucoBoardPose::run(int) {

    int iter = 0;
    cv::Mat cameraMatrix = cv::Mat::eye(3,3, CV_64FC1);
    cv::Size imgSize(500,500);
    cv::aruco::GridBoard board = cv::aruco::GridBoard::create(3, 3, 0.02f, 0.005f,
                                                              cv::aruco::DICT_6X6_250);
    cameraMatrix.at<double>(0,0) = cameraMatrix.at<double>(1,1) = 650;
    cameraMatrix.at<double>(0,2) = imgSize.width / 2;
    cameraMatrix.at<double>(1,2) = imgSize.height / 2;
    for(double distance = 0.2; distance <= 0.4; distance += 0.1) {
        for(int yaw = 0; yaw < 360; yaw+=20) {
            for(int pitch = 30; pitch <=90; pitch+=20) {
                for(unsigned int i=0; i<board.ids.size(); i++)
                    board.ids[i] = (iter+int(i))%250;
                int markerBorder = iter%2+1;
                iter ++;

                cv::Mat img = projectBoard(board, cameraMatrix, deg2rad(pitch), deg2rad(yaw),
                                           distance, imgSize, markerBorder);


                std::vector< std::vector<cv::Point2f> > corners;
                std::vector< int > ids;
                cv::aruco::DetectorParameters params;
                params.minDistanceToBorder = 3;
                params.doCornerRefinement = false;
                params.markerBorderBits = markerBorder;
                cv::aruco::detectMarkers(img, cv::aruco::DICT_6X6_250, corners, ids, params);

                if(ids.size()==0) {
                    ts->printf( cvtest::TS::LOG, "Marker detection failed in Board test" );
                    ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                    return;
                }

                cv::Mat rvec, tvec;
                cv::Mat distCoeffs(5, 0, CV_64FC1, cv::Scalar::all(0));
                cv::aruco::estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs,
                                             rvec, tvec);

                for(unsigned int i=0; i<ids.size(); i++) {
                    int foundIdx = -1;
                    for(unsigned int j=0; j<board.ids.size(); j++) {
                        if(board.ids[j] == ids[i]) {
                            foundIdx = int(j);
                            break;
                        }
                    }

                    if(foundIdx == -1) {
                        ts->printf(cvtest::TS::LOG, "Marker detected with wrong ID in Board test" );
                        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                        return;
                    }

                    std::vector<cv::Point2f> projectedCorners;
                    cv::projectPoints(board.objPoints[foundIdx], rvec, tvec, cameraMatrix,
                                      distCoeffs, projectedCorners);

                    for(int c=0; c<4; c++) {
                        double repError = cv::norm(projectedCorners[c] - corners[i][c]);
                        if(repError > 5.) {
                        ts->printf(cvtest::TS::LOG, "Corner reprojection error too high" );
                            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                            return;
                        }
                    }

                }

            }
        }
    }


}




TEST(CV_ArucoBoardPose, accuracy) {
    CV_ArucoBoardPose test;
    test.safe_run();
}
