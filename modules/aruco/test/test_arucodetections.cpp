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

using namespace std;


class CV_ArucoDetectionSimple : public cvtest::BaseTest
{
public:
    CV_ArucoDetectionSimple();
protected:
    void run(int);
};




CV_ArucoDetectionSimple::CV_ArucoDetectionSimple() {}


void CV_ArucoDetectionSimple::run(int) {

    for(int i=0; i<20; i++) {

        std::vector< std::vector<cv::Point2f> > groundTruthCorners;
        std::vector< int > groundTruthIds;

        const int markerSidePixels = 100;
        int imageSize = markerSidePixels*2 + 3*(markerSidePixels/2);

        cv::Mat img = cv::Mat(imageSize, imageSize, CV_8UC1, cv::Scalar::all(255));
        for(int y=0; y<2; y++) {
            for(int x=0; x<2; x++) {
                cv::Mat marker;
                int id = i*4 + y*2 + x;
                cv::aruco::drawMarker(cv::aruco::DICT_6X6_250, id, markerSidePixels, marker);
                cv::Point2f firstCorner = cv::Point2f(markerSidePixels/2 + x*(1.5*markerSidePixels),
                                                      markerSidePixels/2 + y*(1.5*markerSidePixels)
                                                      );
                cv::Mat aux = img.colRange(firstCorner.x, firstCorner.x+markerSidePixels)
                                 .rowRange(firstCorner.y, firstCorner.y+markerSidePixels);
                marker.copyTo(aux);
                groundTruthIds.push_back(id);
                groundTruthCorners.push_back( std::vector<cv::Point2f>() );
                groundTruthCorners.back().push_back(firstCorner);
                groundTruthCorners.back().push_back(firstCorner +
                                                    cv::Point2f(markerSidePixels-1,0));
                groundTruthCorners.back().push_back(firstCorner +
                                                    cv::Point2f(markerSidePixels-1,
                                                                markerSidePixels-1));
                groundTruthCorners.back().push_back(firstCorner +
                                                    cv::Point2f(0,markerSidePixels-1));
            }
        }
        if(i%2==1) img.convertTo(img, CV_8UC3);

        std::vector< std::vector<cv::Point2f> > corners;
        std::vector< int > ids;
        cv::aruco::DetectorParameters params;
        params.doCornerRefinement = false;
        cv::aruco::detectMarkers(img, cv::aruco::DICT_6X6_250, corners, ids, params);
        for (int m=0; m<groundTruthIds.size(); m++) {
            int idx = -1;
            for(int k=0; k<ids.size(); k++) {
                if(groundTruthIds[m] == ids[k]) {
                    idx = k;
                    break;
                }
            }
            if(idx == -1) {
                ts->printf( cvtest::TS::LOG, "Marker not detected" );
                ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                return;
            }

            for(int c=0; c<4; c++) {
                double dist = cv::norm( groundTruthCorners[m][c] - corners[idx][c] );
                if(dist > 0.001) {
                    ts->printf( cvtest::TS::LOG, "Incorrect marker corners position" );
                    ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                    return;
                }
            }

        }

    }

}


const double PI = 3.141592653589793238463;


double deg2rad(double deg) {
    return deg*PI/180.;
}


void getSyntheticRT(double yaw, double pitch, double distance, cv::Mat &rvec, cv::Mat &tvec) {

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


cv::Mat projectMarker(cv::aruco::DICTIONARY dictionary, int id, cv::Mat cameraMatrix, double yaw,
                      double pitch, double distance, cv::Size imageSize, int markerBorder,
                      std::vector<cv::Point2f> &corners) {


    cv::Mat markerImg;
    const int markerSizePixels = 100;
    cv::aruco::drawMarker(dictionary, id, markerSizePixels, markerImg, markerBorder);

    cv::Mat rvec, tvec;
    getSyntheticRT(yaw, pitch, distance, rvec, tvec);

    const double markerLength = 0.05;
    std::vector<cv::Point3f> markerObjPoints;
    markerObjPoints.push_back( cv::Point3f(-markerLength/2., +markerLength/2., 0) );
    markerObjPoints.push_back( markerObjPoints[0] + cv::Point3f(markerLength, 0, 0) );
    markerObjPoints.push_back( markerObjPoints[0] + cv::Point3f(markerLength, -markerLength, 0) );
    markerObjPoints.push_back( markerObjPoints[0] + cv::Point3f(0, -markerLength, 0) );

    cv::Mat distCoeffs(5, 0, CV_64FC1, cv::Scalar::all(0));
    cv::projectPoints(markerObjPoints, rvec, tvec, cameraMatrix, distCoeffs, corners);


    std::vector<cv::Point2f> originalCorners;
    originalCorners.push_back( cv::Point2f(0, 0) );
    originalCorners.push_back( cv::Point2f(markerSizePixels, 0) );
    originalCorners.push_back( cv::Point2f(markerSizePixels, markerSizePixels) );
    originalCorners.push_back( cv::Point2f(0, markerSizePixels) );

    cv::Mat transformation = cv::getPerspectiveTransform(originalCorners, corners);

    cv::Mat img(imageSize, CV_8UC1, cv::Scalar::all(255));
    cv::Mat aux;
    const char borderValue = 127;
    cv::warpPerspective(markerImg, aux, transformation, imageSize, cv::INTER_NEAREST,
                        cv::BORDER_CONSTANT, cv::Scalar::all(borderValue));

    // copy only not-border pixels
    for (int y = 0; y < aux.rows; y++) {
        for (int x = 0; x < aux.cols; x++) {
            if (aux.at<unsigned char>(y, x) == borderValue)
                continue;
            img.at<unsigned char>(y, x) = aux.at<unsigned char>(y, x);
        }
    }

    return img;

}




class CV_ArucoDetectionPerspective : public cvtest::BaseTest
{
public:
    CV_ArucoDetectionPerspective();
protected:
    void run(int);
};




CV_ArucoDetectionPerspective::CV_ArucoDetectionPerspective() {}


void CV_ArucoDetectionPerspective::run(int) {

    int iter = 0;
    cv::Mat cameraMatrix = cv::Mat::eye(3,3, CV_64FC1);
    cv::Size imgSize(500,500);
    cameraMatrix.at<double>(0,0) = cameraMatrix.at<double>(1,1) = 650;
    cameraMatrix.at<double>(0,2) = imgSize.width / 2;
    cameraMatrix.at<double>(1,2) = imgSize.height / 2;
    for(double distance = 0.1; distance <= 0.5; distance += 0.1) {
        for(int yaw = 0; yaw < 360; yaw+=20) {
            for(int pitch = 30; pitch <=90; pitch+=20) {
                int currentId = iter % 250;
                int markerBorder = iter%2+1;
                iter ++;
                std::vector<cv::Point2f> groundTruthCorners;
                cv::Mat img = projectMarker(cv::aruco::DICT_6X6_250, currentId, cameraMatrix,
                                            deg2rad(pitch), deg2rad(yaw), distance, imgSize,
                                            markerBorder, groundTruthCorners);


                std::vector< std::vector<cv::Point2f> > corners;
                std::vector< int > ids;
                cv::aruco::DetectorParameters params;
                params.minDistanceToBorder = 1;
                params.doCornerRefinement = false;
                params.markerBorderBits = markerBorder;
                cv::aruco::detectMarkers(img, cv::aruco::DICT_6X6_250, corners, ids, params);

                if(ids.size()!=1 || (ids.size()==1 && ids[0]!=currentId)) {
                    if(ids.size() != 1)
                        ts->printf( cvtest::TS::LOG, "Incorrect number of detected markers" );
                    else
                        ts->printf( cvtest::TS::LOG, "Incorrect marker id" );
                    ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                    return;
                }
                for(int c=0; c<4; c++) {
                    double dist = cv::norm( groundTruthCorners[c] - corners[0][c] );
                    if(dist > 5) {
                        ts->printf( cvtest::TS::LOG, "Incorrect marker corners position" );
                        ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                        return;
                    }
                }

            }
        }
    }


}






TEST(Aruco_MarkerDetectionSimple, algorithmic) {
    CV_ArucoDetectionSimple test;
    test.safe_run();
}

TEST(Aruco_MarkerDetectionPerspective, algorithmic) {
    CV_ArucoDetectionPerspective test;
    test.safe_run();
}







