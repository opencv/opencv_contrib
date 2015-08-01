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
#include <opencv2/aruco.hpp>
#include <string>

using namespace std;
using namespace cv;


class CV_ArucoDetectionSimple : public cvtest::BaseTest
{
public:
    CV_ArucoDetectionSimple();
protected:
    void run(int);
};




CV_ArucoDetectionSimple::CV_ArucoDetectionSimple() {}


void CV_ArucoDetectionSimple::run(int) {

    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

    for(int i=0; i<20; i++) {

        vector< vector<Point2f> > groundTruthCorners;
        vector< int > groundTruthIds;

        const int markerSidePixels = 100;
        int imageSize = markerSidePixels*2 + 3*(markerSidePixels/2);

        Mat img = Mat(imageSize, imageSize, CV_8UC1, Scalar::all(255));
        for(int y=0; y<2; y++) {
            for(int x=0; x<2; x++) {
                Mat marker;
                int id = i*4 + y*2 + x;
                aruco::drawMarker(dictionary, id, markerSidePixels, marker);
                Point2f firstCorner = Point2f(markerSidePixels/2.f +
                                                      x*(1.5f*markerSidePixels),
                                                      markerSidePixels/2.f +
                                                      y*(1.5f*markerSidePixels) );
                Mat aux = img.colRange((int)firstCorner.x, (int)firstCorner.x+markerSidePixels)
                                 .rowRange((int)firstCorner.y, (int)firstCorner.y+markerSidePixels);
                marker.copyTo(aux);
                groundTruthIds.push_back(id);
                groundTruthCorners.push_back( vector<Point2f>() );
                groundTruthCorners.back().push_back(firstCorner);
                groundTruthCorners.back().push_back(firstCorner +
                                                    Point2f(markerSidePixels-1,0));
                groundTruthCorners.back().push_back(firstCorner +
                                                    Point2f(markerSidePixels-1,
                                                                markerSidePixels-1));
                groundTruthCorners.back().push_back(firstCorner +
                                                    Point2f(0,markerSidePixels-1));
            }
        }
        if(i%2==1) img.convertTo(img, CV_8UC3);

        vector< vector<Point2f> > corners;
        vector< int > ids;
        aruco::DetectorParameters params;
        params.doCornerRefinement = false;
        aruco::detectMarkers(img, dictionary, corners, ids, params);
        for (unsigned int m=0; m<groundTruthIds.size(); m++) {
            int idx = -1;
            for(unsigned int k=0; k<ids.size(); k++) {
                if(groundTruthIds[m] == ids[k]) {
                    idx = (int)k;
                    break;
                }
            }
            if(idx == -1) {
                ts->printf( cvtest::TS::LOG, "Marker not detected" );
                ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                return;
            }

            for(int c=0; c<4; c++) {
                double dist = norm( groundTruthCorners[m][c] - corners[idx][c] );
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


static double deg2rad(double deg) {
    return deg*PI/180.;
}


static void
getSyntheticRT(double yaw, double pitch, double distance, Mat &rvec, Mat &tvec) {

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


static Mat
projectMarker(aruco::Dictionary dictionary, int id, Mat cameraMatrix, double yaw,
              double pitch, double distance, Size imageSize, int markerBorder,
              vector<Point2f> &corners) {


    Mat markerImg;
    const int markerSizePixels = 100;
    aruco::drawMarker(dictionary, id, markerSizePixels, markerImg, markerBorder);

    Mat rvec, tvec;
    getSyntheticRT(yaw, pitch, distance, rvec, tvec);

    const float markerLength = 0.05f;
    vector<Point3f> markerObjPoints;
    markerObjPoints.push_back( Point3f(-markerLength/2.f, +markerLength/2.f, 0) );
    markerObjPoints.push_back( markerObjPoints[0] + Point3f(markerLength, 0, 0) );
    markerObjPoints.push_back( markerObjPoints[0] + Point3f(markerLength, -markerLength, 0) );
    markerObjPoints.push_back( markerObjPoints[0] + Point3f(0, -markerLength, 0) );

    Mat distCoeffs(5, 0, CV_64FC1, Scalar::all(0));
    projectPoints(markerObjPoints, rvec, tvec, cameraMatrix, distCoeffs, corners);


    vector<Point2f> originalCorners;
    originalCorners.push_back( Point2f(0, 0) );
    originalCorners.push_back( Point2f((float)markerSizePixels, 0) );
    originalCorners.push_back( Point2f((float)markerSizePixels, (float)markerSizePixels) );
    originalCorners.push_back( Point2f(0, (float)markerSizePixels) );

    Mat transformation = getPerspectiveTransform(originalCorners, corners);

    Mat img(imageSize, CV_8UC1, Scalar::all(255));
    Mat aux;
    const char borderValue = 127;
    warpPerspective(markerImg, aux, transformation, imageSize, INTER_NEAREST,
                        BORDER_CONSTANT, Scalar::all(borderValue));

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
    Mat cameraMatrix = Mat::eye(3,3, CV_64FC1);
    Size imgSize(500,500);
    cameraMatrix.at<double>(0,0) = cameraMatrix.at<double>(1,1) = 650;
    cameraMatrix.at<double>(0,2) = imgSize.width / 2;
    cameraMatrix.at<double>(1,2) = imgSize.height / 2;
    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    for(double distance = 0.1; distance <= 0.5; distance += 0.1) {
        for(int pitch = 0; pitch < 360; pitch+=20) {
            for(int yaw = 30; yaw <=90; yaw+=20) {
                int currentId = iter % 250;
                int markerBorder = iter%2+1;
                iter ++;
                vector<Point2f> groundTruthCorners;
                Mat img = projectMarker(dictionary, currentId, cameraMatrix,
                                            deg2rad(yaw), deg2rad(pitch), distance, imgSize,
                                            markerBorder, groundTruthCorners);


                vector< vector<Point2f> > corners;
                vector< int > ids;
                aruco::DetectorParameters params;
                params.minDistanceToBorder = 1;
                params.doCornerRefinement = false;
                params.markerBorderBits = markerBorder;
                aruco::detectMarkers(img, dictionary, corners, ids, params);

                if(ids.size()!=1 || (ids.size()==1 && ids[0]!=currentId)) {
                    if(ids.size() != 1)
                        ts->printf( cvtest::TS::LOG, "Incorrect number of detected markers" );
                    else
                        ts->printf( cvtest::TS::LOG, "Incorrect marker id" );
                    ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                    return;
                }
                for(int c=0; c<4; c++) {
                    double dist = norm( groundTruthCorners[c] - corners[0][c] );
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






TEST(CV_ArucoDetectionSimple, algorithmic) {
    CV_ArucoDetectionSimple test;
    test.safe_run();
}

TEST(CV_ArucoDetectionPerspective, algorithmic) {
    CV_ArucoDetectionPerspective test;
    test.safe_run();
}
