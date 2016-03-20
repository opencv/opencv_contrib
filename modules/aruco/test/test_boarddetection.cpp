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
#include <string>
#include <opencv2/aruco.hpp>

using namespace std;
using namespace cv;


static double deg2rad(double deg) { return deg * CV_PI / 180.; }

/**
 * @brief Get rvec and tvec from yaw, pitch and distance
 */
static void getSyntheticRT(double yaw, double pitch, double distance, Mat &rvec, Mat &tvec) {

    rvec = Mat(3, 1, CV_64FC1);
    tvec = Mat(3, 1, CV_64FC1);

    // Rvec
    // first put the Z axis aiming to -X (like the camera axis system)
    Mat rotZ(3, 1, CV_64FC1);
    rotZ.ptr< double >(0)[0] = 0;
    rotZ.ptr< double >(0)[1] = 0;
    rotZ.ptr< double >(0)[2] = -0.5 * CV_PI;

    Mat rotX(3, 1, CV_64FC1);
    rotX.ptr< double >(0)[0] = 0.5 * CV_PI;
    rotX.ptr< double >(0)[1] = 0;
    rotX.ptr< double >(0)[2] = 0;

    Mat camRvec, camTvec;
    composeRT(rotZ, Mat(3, 1, CV_64FC1, Scalar::all(0)), rotX, Mat(3, 1, CV_64FC1, Scalar::all(0)),
              camRvec, camTvec);

    // now pitch and yaw angles
    Mat rotPitch(3, 1, CV_64FC1);
    rotPitch.ptr< double >(0)[0] = 0;
    rotPitch.ptr< double >(0)[1] = pitch;
    rotPitch.ptr< double >(0)[2] = 0;

    Mat rotYaw(3, 1, CV_64FC1);
    rotYaw.ptr< double >(0)[0] = yaw;
    rotYaw.ptr< double >(0)[1] = 0;
    rotYaw.ptr< double >(0)[2] = 0;

    composeRT(rotPitch, Mat(3, 1, CV_64FC1, Scalar::all(0)), rotYaw,
              Mat(3, 1, CV_64FC1, Scalar::all(0)), rvec, tvec);

    // compose both rotations
    composeRT(camRvec, Mat(3, 1, CV_64FC1, Scalar::all(0)), rvec,
              Mat(3, 1, CV_64FC1, Scalar::all(0)), rvec, tvec);

    // Tvec, just move in z (camera) direction the specific distance
    tvec.ptr< double >(0)[0] = 0.;
    tvec.ptr< double >(0)[1] = 0.;
    tvec.ptr< double >(0)[2] = distance;
}

/**
 * @brief Project a synthetic marker
 */
static void projectMarker(Mat &img, Ptr<aruco::Dictionary> &dictionary, int id,
                          vector< Point3f > markerObjPoints, Mat cameraMatrix, Mat rvec, Mat tvec,
                          int markerBorder) {


    // canonical image
    Mat markerImg;
    const int markerSizePixels = 100;
    aruco::drawMarker(dictionary, id, markerSizePixels, markerImg, markerBorder);

    // projected corners
    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));
    vector< Point2f > corners;
    projectPoints(markerObjPoints, rvec, tvec, cameraMatrix, distCoeffs, corners);

    // get perspective transform
    vector< Point2f > originalCorners;
    originalCorners.push_back(Point2f(0, 0));
    originalCorners.push_back(Point2f((float)markerSizePixels, 0));
    originalCorners.push_back(Point2f((float)markerSizePixels, (float)markerSizePixels));
    originalCorners.push_back(Point2f(0, (float)markerSizePixels));
    Mat transformation = getPerspectiveTransform(originalCorners, corners);

    // apply transformation
    Mat aux;
    const char borderValue = 127;
    warpPerspective(markerImg, aux, transformation, img.size(), INTER_NEAREST, BORDER_CONSTANT,
                    Scalar::all(borderValue));

    // copy only not-border pixels
    for(int y = 0; y < aux.rows; y++) {
        for(int x = 0; x < aux.cols; x++) {
            if(aux.at< unsigned char >(y, x) == borderValue) continue;
            img.at< unsigned char >(y, x) = aux.at< unsigned char >(y, x);
        }
    }
}


/**
 * @brief Get a synthetic image of GridBoard in perspective
 */
static Mat projectBoard(Ptr<aruco::GridBoard> &board, Mat cameraMatrix, double yaw, double pitch,
                        double distance, Size imageSize, int markerBorder) {

    Mat rvec, tvec;
    getSyntheticRT(yaw, pitch, distance, rvec, tvec);

    Mat img = Mat(imageSize, CV_8UC1, Scalar::all(255));
    for(unsigned int m = 0; m < board->ids.size(); m++) {
        projectMarker(img, board->dictionary, board->ids[m], board->objPoints[m], cameraMatrix, rvec,
                      tvec, markerBorder);
    }

    return img;
}



/**
 * @brief Check pose estimation of aruco board
 */
class CV_ArucoBoardPose : public cvtest::BaseTest {
    public:
    CV_ArucoBoardPose();

    protected:
    void run(int);
};


CV_ArucoBoardPose::CV_ArucoBoardPose() {}


void CV_ArucoBoardPose::run(int) {

    int iter = 0;
    Mat cameraMatrix = Mat::eye(3, 3, CV_64FC1);
    Size imgSize(500, 500);
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    Ptr<aruco::GridBoard> gridboard = aruco::GridBoard::create(3, 3, 0.02f, 0.005f, dictionary);
    Ptr<aruco::Board> board = gridboard.staticCast<aruco::Board>();
    cameraMatrix.at< double >(0, 0) = cameraMatrix.at< double >(1, 1) = 650;
    cameraMatrix.at< double >(0, 2) = imgSize.width / 2;
    cameraMatrix.at< double >(1, 2) = imgSize.height / 2;
    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));

    // for different perspectives
    for(double distance = 0.2; distance <= 0.4; distance += 0.2) {
        for(int yaw = 0; yaw < 360; yaw += 100) {
            for(int pitch = 30; pitch <= 90; pitch += 50) {
                for(unsigned int i = 0; i < gridboard->ids.size(); i++)
                    gridboard->ids[i] = (iter + int(i)) % 250;
                int markerBorder = iter % 2 + 1;
                iter++;

                // create synthetic image
                Mat img = projectBoard(gridboard, cameraMatrix, deg2rad(pitch), deg2rad(yaw), distance,
                                       imgSize, markerBorder);


                vector< vector< Point2f > > corners;
                vector< int > ids;
                Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
                params->minDistanceToBorder = 3;
                params->markerBorderBits = markerBorder;
                aruco::detectMarkers(img, dictionary, corners, ids, params);

                if(ids.size() == 0) {
                    ts->printf(cvtest::TS::LOG, "Marker detection failed in Board test");
                    ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                    return;
                }

                // estimate pose
                Mat rvec, tvec;
                aruco::estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, rvec, tvec);

                // check result
                for(unsigned int i = 0; i < ids.size(); i++) {
                    int foundIdx = -1;
                    for(unsigned int j = 0; j < gridboard->ids.size(); j++) {
                        if(gridboard->ids[j] == ids[i]) {
                            foundIdx = int(j);
                            break;
                        }
                    }

                    if(foundIdx == -1) {
                        ts->printf(cvtest::TS::LOG, "Marker detected with wrong ID in Board test");
                        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                        return;
                    }

                    vector< Point2f > projectedCorners;
                    projectPoints(gridboard->objPoints[foundIdx], rvec, tvec, cameraMatrix, distCoeffs,
                                  projectedCorners);

                    for(int c = 0; c < 4; c++) {
                        double repError = norm(projectedCorners[c] - corners[i][c]);
                        if(repError > 5.) {
                            ts->printf(cvtest::TS::LOG, "Corner reprojection error too high");
                            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                            return;
                        }
                    }
                }
            }
        }
    }
}



/**
 * @brief Check refine strategy
 */
class CV_ArucoRefine : public cvtest::BaseTest {
    public:
    CV_ArucoRefine();

    protected:
    void run(int);
};


CV_ArucoRefine::CV_ArucoRefine() {}


void CV_ArucoRefine::run(int) {

    int iter = 0;
    Mat cameraMatrix = Mat::eye(3, 3, CV_64FC1);
    Size imgSize(500, 500);
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    Ptr<aruco::GridBoard> gridboard = aruco::GridBoard::create(3, 3, 0.02f, 0.005f, dictionary);
    Ptr<aruco::Board> board = gridboard.staticCast<aruco::Board>();
    cameraMatrix.at< double >(0, 0) = cameraMatrix.at< double >(1, 1) = 650;
    cameraMatrix.at< double >(0, 2) = imgSize.width / 2;
    cameraMatrix.at< double >(1, 2) = imgSize.height / 2;
    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));

    // for different perspectives
    for(double distance = 0.2; distance <= 0.4; distance += 0.2) {
        for(int yaw = 0; yaw < 360; yaw += 100) {
            for(int pitch = 30; pitch <= 90; pitch += 50) {
                for(unsigned int i = 0; i < gridboard->ids.size(); i++)
                    gridboard->ids[i] = (iter + int(i)) % 250;
                int markerBorder = iter % 2 + 1;
                iter++;

                // create synthetic image
                Mat img = projectBoard(gridboard, cameraMatrix, deg2rad(pitch), deg2rad(yaw), distance,
                                       imgSize, markerBorder);


                // detect markers
                vector< vector< Point2f > > corners, rejected;
                vector< int > ids;
                Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
                params->minDistanceToBorder = 3;
                params->doCornerRefinement = true;
                params->markerBorderBits = markerBorder;
                aruco::detectMarkers(img, dictionary, corners, ids, params, rejected);

                // remove a marker from detection
                int markersBeforeDelete = (int)ids.size();
                if(markersBeforeDelete < 2) continue;

                rejected.push_back(corners[0]);
                corners.erase(corners.begin(), corners.begin() + 1);
                ids.erase(ids.begin(), ids.begin() + 1);

                // try to refind the erased marker
                aruco::refineDetectedMarkers(img, board, corners, ids, rejected, cameraMatrix,
                                             distCoeffs, 10, 3., true, noArray(), params);

                // check result
                if((int)ids.size() < markersBeforeDelete) {
                    ts->printf(cvtest::TS::LOG, "Error in refine detected markers");
                    ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                    return;
                }
            }
        }
    }
}




TEST(CV_ArucoBoardPose, accuracy) {
    CV_ArucoBoardPose test;
    test.safe_run();
}

TEST(CV_ArucoRefine, accuracy) {
    CV_ArucoRefine test;
    test.safe_run();
}
