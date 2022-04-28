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
#include "test_aruco_utils.hpp"

namespace opencv_test { namespace {

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
            squareCorners.push_back(Point3f(startX, startY, 0) - Point3f(squaresX*squareSize/2.f, squaresY*squareSize/2.f, 0.f));
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
    for(unsigned int indexMarker = 0; indexMarker < board->ids.size(); indexMarker++) {
        projectMarker(img, board.staticCast<aruco::Board>(), indexMarker, cameraMatrix, rvec,
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
    Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
    params->minDistanceToBorder = 3;
    aruco::ArucoDetector detector(aruco::getPredefinedDictionary(aruco::DICT_6X6_250), params);
    Ptr<aruco::CharucoBoard> board = aruco::CharucoBoard::create(4, 4, 0.03f, 0.015f, detector.dictionary);

    cameraMatrix.at< double >(0, 0) = cameraMatrix.at< double >(1, 1) = 600;
    cameraMatrix.at< double >(0, 2) = imgSize.width / 2;
    cameraMatrix.at< double >(1, 2) = imgSize.height / 2;

    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));

    // for different perspectives
    for(double distance = 0.2; distance <= 0.4; distance += 0.2) {
        for(int yaw = -55; yaw <= 50; yaw += 25) {
            for(int pitch = -55; pitch <= 50; pitch += 25) {

                int markerBorder = iter % 2 + 1;
                iter++;

                // create synthetic image
                Mat rvec, tvec;
                Mat img = projectCharucoBoard(board, cameraMatrix, deg2rad(yaw), deg2rad(pitch),
                                              distance, imgSize, markerBorder, rvec, tvec);

                // detect markers
                vector< vector< Point2f > > corners;
                vector< int > ids;

                detector.params->markerBorderBits = markerBorder;
                detector.detectMarkers(img, corners, ids);

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

                // copy chessboardCorners
                vector<Point3f> copyChessboardCorners = board->chessboardCorners;
                // move copyChessboardCorners points
                for (size_t i = 0; i < copyChessboardCorners.size(); i++)
                    copyChessboardCorners[i] -= board->rightBottomBorder / 2.f;
                projectPoints(copyChessboardCorners, rvec, tvec, cameraMatrix, distCoeffs,
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
    Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
    params->minDistanceToBorder = 3;
    aruco::ArucoDetector detector(aruco::getPredefinedDictionary(aruco::DICT_6X6_250), params);
    Ptr<aruco::CharucoBoard> board = aruco::CharucoBoard::create(4, 4, 0.03f, 0.015f, detector.dictionary);

    cameraMatrix.at< double >(0, 0) = cameraMatrix.at< double >(1, 1) = 650;
    cameraMatrix.at< double >(0, 2) = imgSize.width / 2;
    cameraMatrix.at< double >(1, 2) = imgSize.height / 2;

    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));
    // for different perspectives
    for(double distance = 0.2; distance <= 0.3; distance += 0.1) {
        for(int yaw = -55; yaw <= 50; yaw += 25) {
            for(int pitch = -55; pitch <= 50; pitch += 25) {

                int markerBorder = iter % 2 + 1;
                iter++;

                // get synthetic image
                Mat rvec, tvec;
                Mat img = projectCharucoBoard(board, cameraMatrix, deg2rad(yaw), deg2rad(pitch),
                                              distance, imgSize, markerBorder, rvec, tvec);

                // detect markers
                vector< vector< Point2f > > corners;
                vector< int > ids;
                detector.params->markerBorderBits = markerBorder;
                detector.detectMarkers(img, corners, ids);

                ASSERT_EQ(ids.size(), board->ids.size());

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


                // check axes
                const float offset = (board->getSquareLength() - board->getMarkerLength()) / 2.f;
                vector<Point2f> axes = getAxis(cameraMatrix, distCoeffs, rvec, tvec, board->getSquareLength(), offset);
                vector<Point2f> topLeft = getMarkerById(board->ids[0], corners, ids);
                ASSERT_NEAR(topLeft[0].x, axes[1].x, 3.f);
                ASSERT_NEAR(topLeft[0].y, axes[1].y, 3.f);
                vector<Point2f> bottomLeft = getMarkerById(board->ids[2], corners, ids);
                ASSERT_NEAR(bottomLeft[0].x, axes[2].x, 3.f);
                ASSERT_NEAR(bottomLeft[0].y, axes[2].y, 3.f);

                // check estimate result
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
    Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
    params->minDistanceToBorder = 0;
    aruco::ArucoDetector detector(aruco::getPredefinedDictionary(aruco::DICT_6X6_250), params);
    float squareLength = 0.03f;
    float markerLength = 0.015f;
    Ptr<aruco::CharucoBoard> board =
        aruco::CharucoBoard::create(3, 3, squareLength, markerLength, detector.dictionary);

    cameraMatrix.at< double >(0, 0) = cameraMatrix.at< double >(1, 1) = 650;
    cameraMatrix.at< double >(0, 2) = imgSize.width / 2;
    cameraMatrix.at< double >(1, 2) = imgSize.height / 2;

    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));
    // for different perspectives
    for(double distance = 0.2; distance <= 0.3; distance += 0.1) {
        for(int yaw = -50; yaw <= 50; yaw += 25) {
            for(int pitch = -50; pitch <= 50; pitch += 25) {

                int markerBorder = iter % 2 + 1;
                for(int i = 0; i < 4; i++)
                    board->ids[i] = 4 * iter + i;
                iter++;

                // get synthetic image
                Mat rvec, tvec;
                Mat img = projectCharucoBoard(board, cameraMatrix, deg2rad(yaw), deg2rad(pitch),
                                              distance, imgSize, markerBorder, rvec, tvec);

                // detect markers
                vector< vector< Point2f > > corners;
                vector< int > ids;
                detector.params->markerBorderBits = markerBorder;
                detector.detectMarkers(img, corners, ids);

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

                // copy chessboardCorners
                vector<Point3f> copyChessboardCorners = board->chessboardCorners;
                // move copyChessboardCorners points
                for (size_t i = 0; i < copyChessboardCorners.size(); i++)
                    copyChessboardCorners[i] -= board->rightBottomBorder / 2.f;

                projectPoints(copyChessboardCorners, rvec, tvec, cameraMatrix, distCoeffs,
                              projectedDiamondCorners);

                vector< Point2f > projectedDiamondCornersReorder(4);
                projectedDiamondCornersReorder[0] = projectedDiamondCorners[0];
                projectedDiamondCornersReorder[1] = projectedDiamondCorners[1];
                projectedDiamondCornersReorder[2] = projectedDiamondCorners[3];
                projectedDiamondCornersReorder[3] = projectedDiamondCorners[2];


                for(unsigned int i = 0; i < 4; i++) {

                    double repError = cv::norm(diamondCorners[0][i] - projectedDiamondCornersReorder[i]);  // TODO cvtest

                    if(repError > 5.) {
                        ts->printf(cvtest::TS::LOG, "Diamond corner reprojection error too high");
                        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                        return;
                    }
                }

                Ptr<aruco::EstimateParameters> estimateParameters = aruco::EstimateParameters::create();
                estimateParameters->pattern = aruco::CW_top_left_corner;
                // estimate diamond pose
                vector< Vec3d > estimatedRvec, estimatedTvec;
                aruco::estimatePoseSingleMarkers(diamondCorners, squareLength, cameraMatrix, distCoeffs, estimatedRvec,
                                                 estimatedTvec, noArray(), estimateParameters);

                // check result
                vector< Point2f > projectedDiamondCornersPose;
                vector< Vec3f > diamondObjPoints(4);
                diamondObjPoints[0] = Vec3f(0.f, 0.f, 0);
                diamondObjPoints[1] = Vec3f(squareLength, 0.f, 0);
                diamondObjPoints[2] = Vec3f(squareLength, squareLength, 0);
                diamondObjPoints[3] = Vec3f(0.f, squareLength, 0);
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

    // set expected_corners values
    cv::Mat expected_corners = (cv::Mat_<float>(9,2) <<
        200, 200,
        250, 200,
        300, 200,
        200, 250,
        250, 250,
        300, 250,
        200, 300,
        250, 300,
        300, 300
    );

    cv::Mat gray;

    auto dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
    auto board = cv::aruco::CharucoBoard::create(4, 4, 1.f, .8f, dict);

    // generate ChArUco board
    board->draw(Size(res.width, res.height), gray, 150);
    cv::GaussianBlur(gray, gray, Size(5, 5), 1.0);

    auto params = cv::aruco::DetectorParameters::create();
    params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_APRILTAG;

    aruco::ArucoDetector detector(dict, params);

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners, rejected;

    detector.detectMarkers(gray, corners, ids, rejected);

    ASSERT_EQ(ids.size(), size_t(8));

    cv::Mat c_ids, c_corners;
    cv::aruco::interpolateCornersCharuco(corners, ids, gray, board, c_corners, c_ids, K);

    ASSERT_EQ(c_corners.rows, expected_corners.rows);
    EXPECT_NEAR(0, cvtest::norm(expected_corners, c_corners.reshape(1), NORM_INF), 1e-1);

    c_ids = cv::Mat();
    c_corners = cv::Mat();
    cv::aruco::interpolateCornersCharuco(corners, ids, gray, board, c_corners, c_ids);

    ASSERT_EQ(c_corners.rows, expected_corners.rows);
    EXPECT_NEAR(0, cvtest::norm(expected_corners, c_corners.reshape(1), NORM_INF), 1e-1);
}

TEST(CV_ArucoTutorial, can_find_choriginal)
{
    string imgPath = cvtest::findDataFile("choriginal.jpg", false);
    Mat image = imread(imgPath);
    aruco::ArucoDetector detector(aruco::getPredefinedDictionary(aruco::DICT_6X6_250));

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

    detector.detectMarkers(image, corners, ids, rejected);

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
    aruco::ArucoDetector detector(aruco::getPredefinedDictionary(aruco::DICT_6X6_250));

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

    detector.detectMarkers(image, corners, ids, rejected);

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
    Ptr<aruco::Dictionary> dictionary = makePtr<aruco::Dictionary>();
    FileStorage fs(dictPath, FileStorage::READ);
    dictionary->aruco::Dictionary::readDictionary(fs.root()); // set marker from tutorial_dict.yml

    string detectorPath = cvtest::findDataFile("detector_params.yml", false);
    fs = FileStorage(detectorPath, FileStorage::READ);
    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    detectorParams->readDetectorParameters(fs.root());
    detectorParams->cornerRefinementMethod = 3;

    aruco::ArucoDetector detector(dictionary, detectorParams);

    vector< int > ids;
    vector< vector< Point2f > > corners, rejected;
    const size_t N = 12ull;
    // corner indices of ArUco markers
    const int goldCornersIds[N] = { 4, 12, 11, 3, 12, 10, 12, 10, 10, 11, 2, 11 };
    map<int, int> counterGoldCornersIds;
    for (int i = 0; i < static_cast<int>(N); i++)
        counterGoldCornersIds[goldCornersIds[i]]++;

    detector.detectMarkers(image, corners, ids, rejected);
    map<int, int> counterRes;
    for (size_t i = 0; i < N; i++)
    {
        int arucoId = ids[i];
        counterRes[arucoId]++;
    }

    ASSERT_EQ(N, ids.size());
    EXPECT_EQ(counterGoldCornersIds, counterRes); // check the number of ArUco markers
}

TEST(Charuco, issue_14014)
{
    string imgPath = cvtest::findDataFile("aruco/recover.png");
    Mat img = imread(imgPath);

    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
    detectorParams->cornerRefinementMinAccuracy = 0.01;
    aruco::ArucoDetector detector(aruco::getPredefinedDictionary(aruco::DICT_7X7_250), detectorParams);
    Ptr<aruco::CharucoBoard> board = aruco::CharucoBoard::create(8, 5, 0.03455f, 0.02164f, detector.dictionary);

    vector<Mat> corners, rejectedPoints;
    vector<int> ids;

    detector.detectMarkers(img, corners, ids, rejectedPoints);

    ASSERT_EQ(corners.size(), 19ull);
    EXPECT_EQ(Size(4, 1), corners[0].size()); // check dimension of detected corners

    size_t numRejPoints = rejectedPoints.size();
    ASSERT_EQ(rejectedPoints.size(), 26ull); // optional check to track regressions
    EXPECT_EQ(Size(4, 1), rejectedPoints[0].size()); // check dimension of detected corners

    detector.refineDetectedMarkers(img, board, corners, ids, rejectedPoints);

    ASSERT_EQ(corners.size(), 20ull);
    EXPECT_EQ(Size(4, 1), corners[0].size()); // check dimension of rejected corners after successfully refine

    ASSERT_EQ(rejectedPoints.size() + 1, numRejPoints);
    EXPECT_EQ(Size(4, 1), rejectedPoints[0].size()); // check dimension of rejected corners after successfully refine
}

}} // namespace
