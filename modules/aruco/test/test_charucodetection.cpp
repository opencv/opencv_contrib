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

TEST(Charuco, testCharucoCornersCollinear_true)
{
    int squaresX = 13;
    int squaresY = 28;
    float squareLength = 300;
    float markerLength = 150;
    int dictionaryId = 11;


    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();

    Ptr<aruco::Dictionary> dictionary =
            aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    Ptr<aruco::CharucoBoard> charucoBoard =
            aruco::CharucoBoard::create(squaresX, squaresY, squareLength, markerLength, dictionary);

    // consistency with C++98
    const int arrLine[9] = {192, 204, 216, 228, 240, 252, 264, 276, 288};
    vector<int> charucoIdsAxisLine(9, 0);

    for (int i = 0; i < 9; i++){
        charucoIdsAxisLine[i] = arrLine[i];
    }

    const int arrDiag[7] = {198, 209, 220, 231, 242, 253, 264};

    vector<int> charucoIdsDiagonalLine(7, 0);

    for (int i = 0; i < 7; i++){
        charucoIdsDiagonalLine[i] = arrDiag[i];
    }

    bool resultAxisLine = cv::aruco::testCharucoCornersCollinear(charucoBoard, charucoIdsAxisLine);

    bool resultDiagonalLine = cv::aruco::testCharucoCornersCollinear(charucoBoard, charucoIdsDiagonalLine);

    EXPECT_TRUE(resultAxisLine);

    EXPECT_TRUE(resultDiagonalLine);
}

TEST(Charuco, testCharucoCornersCollinear_false)
{
    int squaresX = 13;
    int squaresY = 28;
    float squareLength = 300;
    float markerLength = 150;
    int dictionaryId = 11;


    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();

    Ptr<aruco::Dictionary> dictionary =
            aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    Ptr<aruco::CharucoBoard> charucoBoard =
            aruco::CharucoBoard::create(squaresX, squaresY, squareLength, markerLength, dictionary);

    // consistency with C++98
    const int arr[63] = {192, 193, 194, 195, 196, 197, 198, 204, 205, 206, 207, 208,
                                209, 210, 216, 217, 218, 219, 220, 221, 222, 228, 229, 230,
                                231, 232, 233, 234, 240, 241, 242, 243, 244, 245, 246, 252,
                                253, 254, 255, 256, 257, 258, 264, 265, 266, 267, 268, 269,
                                270, 276, 277, 278, 279, 280, 281, 282, 288, 289, 290, 291,
                                292, 293, 294};

    vector<int> charucoIds(63, 0);
    for (int i = 0; i < 63; i++){
        charucoIds[i] = arr[i];
    }


    bool result = cv::aruco::testCharucoCornersCollinear(charucoBoard, charucoIds);

    EXPECT_FALSE(result);
}

// test that ChArUco board detection is subpixel accurate
TEST(Charuco, testBoardSubpixelCoords)
{
    cv::Size res{500, 500};
    cv::Mat K = (cv::Mat_<double>(3,3) <<
        0.5*res.width, 0, 0.5*res.width,
        0, 0.5*res.height, 0.5*res.height,
        0, 0, 1);

    // load board image with corners at round values
    cv::String testImagePath = cvtest::TS::ptr()->get_data_path() + "aruco/" + "trivial_board_detection.png";
    Mat img = imread(testImagePath);
    cv::Mat expected_corners = (cv::Mat_<float>(9,2) <<
        200, 300,
        250, 300,
        300, 300,
        200, 250,
        250, 250,
        300, 250,
        200, 200,
        250, 200,
        300, 200
    );

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    auto dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
    auto board = cv::aruco::CharucoBoard::create(4, 4, 1.f, .8f, dict);

    auto params = cv::aruco::DetectorParameters::create();
    params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_APRILTAG;

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners, rejected;

    cv::aruco::detectMarkers(gray, dict, corners, ids, params, rejected, K);

    ASSERT_EQ(ids.size(), size_t(8));

    cv::Mat c_ids, c_corners;
    cv::aruco::interpolateCornersCharuco(corners, ids, gray, board, c_corners, c_ids, K);
    cv::Mat corners_reshaped = c_corners.reshape(1);

    ASSERT_EQ(c_corners.rows, expected_corners.rows);
    EXPECT_NEAR(0, cvtest::norm(expected_corners, c_corners.reshape(1), NORM_INF), 1e-3);

    c_ids = cv::Mat();
    c_corners = cv::Mat();
    cv::aruco::interpolateCornersCharuco(corners, ids, gray, board, c_corners, c_ids);
    corners_reshaped = c_corners.reshape(1);

    ASSERT_EQ(c_corners.rows, expected_corners.rows);
    EXPECT_NEAR(0, cvtest::norm(expected_corners, c_corners.reshape(1), NORM_INF), 1e-3);
}

TEST(CV_ArucoTutorial, can_find_choriginal)
{
    string imgPath = cvtest::findDataFile("choriginal.jpg", false);
    Mat image = imread(imgPath);
    cv::Ptr<cv::aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();

    vector< int > ids;
    vector< vector< Point2f > > corners, rejected;
    const size_t N = 17ull;
    // corners of aruco markers with indices goldCornersIds
    const int goldCorners[N][8] = { {268,77,  290,80,  286,97,  263,94},  {360,90,  382,93,  379,111, 357,108},
                                    {211,106, 233,109, 228,127, 205,123}, {306,120, 328,124, 325,142, 302,138},
                                    {402,135, 425,139, 423,157, 400,154}, {247,152, 271,155, 267,174, 242,171},
                                    {347,167, 371,171, 369,191, 344,187}, {185,185, 209,189, 203,210, 178,206},
                                    {288,201, 313,206, 309,227, 284,223}, {393,218, 418,222, 416,245, 391,241},
                                    {223,240, 250,244, 244,268, 217,263}, {333,258, 359,262, 356,286, 329,282},
                                    {152,281, 179,285, 171,312, 143,307}, {267,300, 294,305, 289,331, 261,327},
                                    {383,319, 410,324, 408,351, 380,347}, {194,347, 223,352, 216,382, 186,377},
                                    {315,368, 345,373, 341,403, 310,398} };
    map<int, const int*> mapGoldCorners;
    for (int i = 0; i < static_cast<int>(N); i++)
        mapGoldCorners[i] = goldCorners[i];

    aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);

    ASSERT_EQ(N, ids.size());
    for (size_t i = 0; i < N; i++)
    {
        int arucoId = ids[i];
        ASSERT_EQ(4ull, corners[i].size());
        ASSERT_TRUE(mapGoldCorners.find(arucoId) != mapGoldCorners.end());
        for (int j = 0; j < 4; j++)
        {
            EXPECT_NEAR(static_cast<float>(mapGoldCorners[arucoId][j * 2]), corners[i][j].x, 1.f);
            EXPECT_NEAR(static_cast<float>(mapGoldCorners[arucoId][j * 2 + 1]), corners[i][j].y, 1.f);
        }
    }
}

TEST(CV_ArucoTutorial, can_find_chocclusion)
{
    string imgPath = cvtest::findDataFile("chocclusion_original.jpg", false);
    Mat image = imread(imgPath);
    cv::Ptr<cv::aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();

    vector< int > ids;
    vector< vector< Point2f > > corners, rejected;
    const size_t N = 13ull;
    // corners of aruco markers with indices goldCornersIds
    const int goldCorners[N][8] = { {301,57, 322,62, 317,79, 295,73}, {391,80, 413,85, 408,103, 386,97},
                                    {242,79, 264,85, 256,102, 234,96}, {334,103, 357,109, 352,126, 329,121},
                                    {428,129, 451,134, 448,152, 425,146}, {274,128, 296,134, 290,153, 266,147},
                                    {371,154, 394,160, 390,180, 366,174}, {208,155, 232,161, 223,181, 199,175},
                                    {309,182, 333,188, 327,209, 302,203}, {411,210, 436,216, 432,238, 407,231},
                                    {241,212, 267,219, 258,242, 232,235}, {167,244, 194,252, 183,277, 156,269},
                                    {202,314, 230,322, 220,349, 191,341} };
    map<int, const int*> mapGoldCorners;
    const int goldCornersIds[N] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15};
    for (int i = 0; i < static_cast<int>(N); i++)
        mapGoldCorners[goldCornersIds[i]] = goldCorners[i];

    aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);

    ASSERT_EQ(N, ids.size());
    for (size_t i = 0; i < N; i++)
    {
        int arucoId = ids[i];
        ASSERT_EQ(4ull, corners[i].size());
        ASSERT_TRUE(mapGoldCorners.find(arucoId) != mapGoldCorners.end());
        for (int j = 0; j < 4; j++)
        {
            EXPECT_NEAR(static_cast<float>(mapGoldCorners[arucoId][j * 2]), corners[i][j].x, 1.f);
            EXPECT_NEAR(static_cast<float>(mapGoldCorners[arucoId][j * 2 + 1]), corners[i][j].y, 1.f);
        }
    }
}

TEST(CV_ArucoTutorial, can_find_diamondmarkers)
{
    string imgPath = cvtest::findDataFile("diamondmarkers.png", false);
    Mat image = imread(imgPath);

    string dictPath = cvtest::findDataFile("tutorial_dict.yml", false);
    cv::Ptr<cv::aruco::Dictionary> dictionary;
    FileStorage fs(dictPath, FileStorage::READ);
    aruco::Dictionary::readDictionary(fs.root(), dictionary); // set marker from tutorial_dict.yml

    string detectorPath = cvtest::findDataFile("detector_params.yml", false);
    fs = FileStorage(detectorPath, FileStorage::READ);
    Ptr<aruco::DetectorParameters> detectorParams;
    aruco::DetectorParameters::readDetectorParameters(fs.root(), detectorParams);
    detectorParams->cornerRefinementMethod = 3;

    vector< int > ids;
    vector< vector< Point2f > > corners, rejected;
    const size_t N = 12ull;
    // corner indices of ArUco markers
    const int goldCornersIds[N] = { 4, 12, 11, 3, 12, 10, 12, 10, 10, 11, 2, 11 };
    map<int, int> counterGoldCornersIds;
    for (int i = 0; i < static_cast<int>(N); i++)
        counterGoldCornersIds[goldCornersIds[i]]++;

    aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);
    map<int, int> counterRes;
    for (size_t i = 0; i < N; i++)
    {
        int arucoId = ids[i];
        counterRes[arucoId]++;
    }

    ASSERT_EQ(N, ids.size());
    EXPECT_EQ(counterGoldCornersIds, counterRes); // check the number of ArUco markers
}

}} // namespace
