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

namespace opencv_test { namespace {

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
static void projectMarker(Mat &img, Ptr<aruco::Dictionary> dictionary, int id,
                          vector< Point3f > markerObjPoints, Mat cameraMatrix, Mat rvec, Mat tvec,
                          int markerBorder) {


    Mat markerImg;
    const int markerSizePixels = 100;
    aruco::drawMarker(dictionary, id, markerSizePixels, markerImg, markerBorder);

    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));
    vector< Point2f > corners;
    projectPoints(markerObjPoints, rvec, tvec, cameraMatrix, distCoeffs, corners);

    vector< Point2f > originalCorners;
    originalCorners.push_back(Point2f(0, 0));
    originalCorners.push_back(Point2f((float)markerSizePixels, 0));
    originalCorners.push_back(Point2f((float)markerSizePixels, (float)markerSizePixels));
    originalCorners.push_back(Point2f(0, (float)markerSizePixels));

    Mat transformation = getPerspectiveTransform(originalCorners, corners);

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
 * @brief Get a synthetic image of Chessboard in perspective
 */
static Mat projectChessboard(int squaresX, int squaresY, float squareSize, Size imageSize,
                             Mat cameraMatrix, Mat rvec, Mat tvec) {

    Mat img(imageSize, CV_8UC1, Scalar::all(255));
    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));

    for(int y = 0; y < squaresY; y++) {
        float startY = float(y) * squareSize;
        for(int x = 0; x < squaresX; x++) {
            if(y % 2 != x % 2) continue;
            float startX = float(x) * squareSize;

            vector< Point3f > squareCorners;
            squareCorners.push_back(Point3f(startX, startY, 0));
            squareCorners.push_back(squareCorners[0] + Point3f(squareSize, 0, 0));
            squareCorners.push_back(squareCorners[0] + Point3f(squareSize, squareSize, 0));
            squareCorners.push_back(squareCorners[0] + Point3f(0, squareSize, 0));

            vector< vector< Point2f > > projectedCorners;
            projectedCorners.push_back(vector< Point2f >());
            projectPoints(squareCorners, rvec, tvec, cameraMatrix, distCoeffs, projectedCorners[0]);

            vector< vector< Point > > projectedCornersInt;
            projectedCornersInt.push_back(vector< Point >());

            for(int k = 0; k < 4; k++)
                projectedCornersInt[0]
                    .push_back(Point((int)projectedCorners[0][k].x, (int)projectedCorners[0][k].y));

            fillPoly(img, projectedCornersInt, Scalar::all(0));
        }
    }

    return img;
}


/**
 * @brief Check pose estimation of charuco board
 */
static Mat projectCharucoBoard(Ptr<aruco::CharucoBoard> &board, Mat cameraMatrix, double yaw,
                               double pitch, double distance, Size imageSize, int markerBorder,
                               Mat &rvec, Mat &tvec) {

    getSyntheticRT(yaw, pitch, distance, rvec, tvec);

    // project markers
    Mat img = Mat(imageSize, CV_8UC1, Scalar::all(255));
    for(unsigned int m = 0; m < board->ids.size(); m++) {
        projectMarker(img, board->dictionary, board->ids[m], board->objPoints[m], cameraMatrix, rvec,
                      tvec, markerBorder);
    }

    // project chessboard
    Mat chessboard =
        projectChessboard(board->getChessboardSize().width, board->getChessboardSize().height,
                          board->getSquareLength(), imageSize, cameraMatrix, rvec, tvec);

    for(unsigned int i = 0; i < chessboard.total(); i++) {
        if(chessboard.ptr< unsigned char >()[i] == 0) {
            img.ptr< unsigned char >()[i] = 0;
        }
    }

    return img;
}




/**
 * @brief Check Charuco detection
 */
class CV_CharucoDetection : public cvtest::BaseTest {
    public:
    CV_CharucoDetection();

    protected:
    void run(int);
};


CV_CharucoDetection::CV_CharucoDetection() {}


void CV_CharucoDetection::run(int) {

    int iter = 0;
    Mat cameraMatrix = Mat::eye(3, 3, CV_64FC1);
    Size imgSize(500, 500);
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    Ptr<aruco::CharucoBoard> board = aruco::CharucoBoard::create(4, 4, 0.03f, 0.015f, dictionary);

    cameraMatrix.at< double >(0, 0) = cameraMatrix.at< double >(1, 1) = 650;
    cameraMatrix.at< double >(0, 2) = imgSize.width / 2;
    cameraMatrix.at< double >(1, 2) = imgSize.height / 2;

    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));

    // for different perspectives
    for(double distance = 0.2; distance <= 0.4; distance += 0.2) {
        for(int yaw = 0; yaw < 360; yaw += 100) {
            for(int pitch = 30; pitch <= 90; pitch += 50) {

                int markerBorder = iter % 2 + 1;
                iter++;

                // create synthetic image
                Mat rvec, tvec;
                Mat img = projectCharucoBoard(board, cameraMatrix, deg2rad(pitch), deg2rad(yaw),
                                              distance, imgSize, markerBorder, rvec, tvec);

                // detect markers
                vector< vector< Point2f > > corners;
                vector< int > ids;
                Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
                params->minDistanceToBorder = 3;
                params->markerBorderBits = markerBorder;
                aruco::detectMarkers(img, dictionary, corners, ids, params);

                if(ids.size() == 0) {
                    ts->printf(cvtest::TS::LOG, "Marker detection failed");
                    ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                    return;
                }

                // interpolate charuco corners
                vector< Point2f > charucoCorners;
                vector< int > charucoIds;

                if(iter % 2 == 0) {
                    aruco::interpolateCornersCharuco(corners, ids, img, board, charucoCorners,
                                                     charucoIds);
                } else {
                    aruco::interpolateCornersCharuco(corners, ids, img, board, charucoCorners,
                                                     charucoIds, cameraMatrix, distCoeffs);
                }

                // check results
                vector< Point2f > projectedCharucoCorners;
                projectPoints(board->chessboardCorners, rvec, tvec, cameraMatrix, distCoeffs,
                              projectedCharucoCorners);

                for(unsigned int i = 0; i < charucoIds.size(); i++) {

                    int currentId = charucoIds[i];

                    if(currentId >= (int)board->chessboardCorners.size()) {
                        ts->printf(cvtest::TS::LOG, "Invalid Charuco corner id");
                        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                        return;
                    }

                    double repError = cv::norm(charucoCorners[i] - projectedCharucoCorners[currentId]);  // TODO cvtest


                    if(repError > 5.) {
                        ts->printf(cvtest::TS::LOG, "Charuco corner reprojection error too high");
                        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                        return;
                    }
                }
            }
        }
    }
}



/**
 * @brief Check charuco pose estimation
 */
class CV_CharucoPoseEstimation : public cvtest::BaseTest {
    public:
    CV_CharucoPoseEstimation();

    protected:
    void run(int);
};


CV_CharucoPoseEstimation::CV_CharucoPoseEstimation() {}


void CV_CharucoPoseEstimation::run(int) {

    int iter = 0;
    Mat cameraMatrix = Mat::eye(3, 3, CV_64FC1);
    Size imgSize(500, 500);
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    Ptr<aruco::CharucoBoard> board = aruco::CharucoBoard::create(4, 4, 0.03f, 0.015f, dictionary);

    cameraMatrix.at< double >(0, 0) = cameraMatrix.at< double >(1, 1) = 650;
    cameraMatrix.at< double >(0, 2) = imgSize.width / 2;
    cameraMatrix.at< double >(1, 2) = imgSize.height / 2;

    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));
    // for different perspectives
    for(double distance = 0.2; distance <= 0.4; distance += 0.2) {
        for(int yaw = 0; yaw < 360; yaw += 100) {
            for(int pitch = 30; pitch <= 90; pitch += 50) {

                int markerBorder = iter % 2 + 1;
                iter++;

                // get synthetic image
                Mat rvec, tvec;
                Mat img = projectCharucoBoard(board, cameraMatrix, deg2rad(pitch), deg2rad(yaw),
                                              distance, imgSize, markerBorder, rvec, tvec);

                // detect markers
                vector< vector< Point2f > > corners;
                vector< int > ids;
                Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
                params->minDistanceToBorder = 3;
                params->markerBorderBits = markerBorder;
                aruco::detectMarkers(img, dictionary, corners, ids, params);

                if(ids.size() == 0) {
                    ts->printf(cvtest::TS::LOG, "Marker detection failed");
                    ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                    return;
                }

                // interpolate charuco corners
                vector< Point2f > charucoCorners;
                vector< int > charucoIds;

                if(iter % 2 == 0) {
                    aruco::interpolateCornersCharuco(corners, ids, img, board, charucoCorners,
                                                     charucoIds);
                } else {
                    aruco::interpolateCornersCharuco(corners, ids, img, board, charucoCorners,
                                                     charucoIds, cameraMatrix, distCoeffs);
                }

                if(charucoIds.size() == 0) continue;

                // estimate charuco pose
                aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix,
                                                distCoeffs, rvec, tvec);


                // check result
                vector< Point2f > projectedCharucoCorners;
                projectPoints(board->chessboardCorners, rvec, tvec, cameraMatrix, distCoeffs,
                              projectedCharucoCorners);

                for(unsigned int i = 0; i < charucoIds.size(); i++) {

                    int currentId = charucoIds[i];

                    if(currentId >= (int)board->chessboardCorners.size()) {
                        ts->printf(cvtest::TS::LOG, "Invalid Charuco corner id");
                        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                        return;
                    }

                    double repError = cv::norm(charucoCorners[i] - projectedCharucoCorners[currentId]);  // TODO cvtest


                    if(repError > 5.) {
                        ts->printf(cvtest::TS::LOG, "Charuco corner reprojection error too high");
                        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                        return;
                    }
                }
            }
        }
    }
}


/**
 * @brief Check diamond detection
 */
class CV_CharucoDiamondDetection : public cvtest::BaseTest {
    public:
    CV_CharucoDiamondDetection();

    protected:
    void run(int);
};


CV_CharucoDiamondDetection::CV_CharucoDiamondDetection() {}


void CV_CharucoDiamondDetection::run(int) {

    int iter = 0;
    Mat cameraMatrix = Mat::eye(3, 3, CV_64FC1);
    Size imgSize(500, 500);
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    float squareLength = 0.03f;
    float markerLength = 0.015f;
    Ptr<aruco::CharucoBoard> board =
        aruco::CharucoBoard::create(3, 3, squareLength, markerLength, dictionary);

    cameraMatrix.at< double >(0, 0) = cameraMatrix.at< double >(1, 1) = 650;
    cameraMatrix.at< double >(0, 2) = imgSize.width / 2;
    cameraMatrix.at< double >(1, 2) = imgSize.height / 2;

    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));
    // for different perspectives
    for(double distance = 0.3; distance <= 0.3; distance += 0.2) {
        for(int yaw = 0; yaw < 360; yaw += 100) {
            for(int pitch = 30; pitch <= 90; pitch += 30) {

                int markerBorder = iter % 2 + 1;
                for(int i = 0; i < 4; i++)
                    board->ids[i] = 4 * iter + i;
                iter++;

                // get synthetic image
                Mat rvec, tvec;
                Mat img = projectCharucoBoard(board, cameraMatrix, deg2rad(pitch), deg2rad(yaw),
                                              distance, imgSize, markerBorder, rvec, tvec);

                // detect markers
                vector< vector< Point2f > > corners;
                vector< int > ids;
                Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
                params->minDistanceToBorder = 0;
                params->markerBorderBits = markerBorder;
                aruco::detectMarkers(img, dictionary, corners, ids, params);

                if(ids.size() != 4) {
                    ts->printf(cvtest::TS::LOG, "Not enough markers for diamond detection");
                    ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                    return;
                }

                // detect diamonds
                vector< vector< Point2f > > diamondCorners;
                vector< Vec4i > diamondIds;
                aruco::detectCharucoDiamond(img, corners, ids, squareLength / markerLength,
                                            diamondCorners, diamondIds, cameraMatrix, distCoeffs);

                // check results
                if(diamondIds.size() != 1) {
                    ts->printf(cvtest::TS::LOG, "Diamond not detected correctly");
                    ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                    return;
                }

                for(int i = 0; i < 4; i++) {
                    if(diamondIds[0][i] != board->ids[i]) {
                        ts->printf(cvtest::TS::LOG, "Incorrect diamond ids");
                        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                        return;
                    }
                }


                vector< Point2f > projectedDiamondCorners;
                projectPoints(board->chessboardCorners, rvec, tvec, cameraMatrix, distCoeffs,
                              projectedDiamondCorners);

                vector< Point2f > projectedDiamondCornersReorder(4);
                projectedDiamondCornersReorder[0] = projectedDiamondCorners[2];
                projectedDiamondCornersReorder[1] = projectedDiamondCorners[3];
                projectedDiamondCornersReorder[2] = projectedDiamondCorners[1];
                projectedDiamondCornersReorder[3] = projectedDiamondCorners[0];


                for(unsigned int i = 0; i < 4; i++) {

                    double repError = cv::norm(diamondCorners[0][i] - projectedDiamondCornersReorder[i]);  // TODO cvtest

                    if(repError > 5.) {
                        ts->printf(cvtest::TS::LOG, "Diamond corner reprojection error too high");
                        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                        return;
                    }
                }

                // estimate diamond pose
                vector< Vec3d > estimatedRvec, estimatedTvec;
                aruco::estimatePoseSingleMarkers(diamondCorners, squareLength, cameraMatrix,
                                                 distCoeffs, estimatedRvec, estimatedTvec);

                // check result
                vector< Point2f > projectedDiamondCornersPose;
                vector< Vec3f > diamondObjPoints(4);
                diamondObjPoints[0] = Vec3f(-squareLength / 2.f, squareLength / 2.f, 0);
                diamondObjPoints[1] = Vec3f(squareLength / 2.f, squareLength / 2.f, 0);
                diamondObjPoints[2] = Vec3f(squareLength / 2.f, -squareLength / 2.f, 0);
                diamondObjPoints[3] = Vec3f(-squareLength / 2.f, -squareLength / 2.f, 0);
                projectPoints(diamondObjPoints, estimatedRvec[0], estimatedTvec[0], cameraMatrix,
                              distCoeffs, projectedDiamondCornersPose);

                for(unsigned int i = 0; i < 4; i++) {
                    double repError = cv::norm(projectedDiamondCornersReorder[i] - projectedDiamondCornersPose[i]);  // TODO cvtest

                    if(repError > 5.) {
                        ts->printf(cvtest::TS::LOG, "Charuco pose error too high");
                        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                        return;
                    }
                }
            }
        }
    }
}

/**
* @brief Check charuco board creation
*/
class CV_CharucoBoardCreation : public cvtest::BaseTest {
public:
    CV_CharucoBoardCreation();

protected:
    void run(int);
};

CV_CharucoBoardCreation::CV_CharucoBoardCreation() {}

void CV_CharucoBoardCreation::run(int)
{
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_5X5_250);
    int n = 6;

    float markerSizeFactor = 0.5f;

    for (float squareSize_mm = 5.0f; squareSize_mm < 35.0f; squareSize_mm += 0.1f)
    {
        Ptr<aruco::CharucoBoard> board_meters = aruco::CharucoBoard::create(
            n, n, squareSize_mm*1e-3f, squareSize_mm * markerSizeFactor * 1e-3f, dictionary);

        Ptr<aruco::CharucoBoard> board_millimeters = aruco::CharucoBoard::create(
            n, n, squareSize_mm, squareSize_mm * markerSizeFactor, dictionary);

        for (size_t i = 0; i < board_meters->nearestMarkerIdx.size(); i++)
        {
            if (board_meters->nearestMarkerIdx[i].size() != board_millimeters->nearestMarkerIdx[i].size() ||
                board_meters->nearestMarkerIdx[i][0] != board_millimeters->nearestMarkerIdx[i][0])
            {
                ts->printf(cvtest::TS::LOG,
                    cv::format("Charuco board topology is sensitive to scale with squareSize=%.1f\n",
                        squareSize_mm).c_str());
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                break;
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

TEST(CV_CharucoDiamondDetection, accuracy) {
    CV_CharucoDiamondDetection test;
    test.safe_run();
}

TEST(CV_CharucoBoardCreation, accuracy) {
    CV_CharucoBoardCreation test;
    test.safe_run();
}

}} // namespace
