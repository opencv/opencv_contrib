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
#include <opencv2/aruco.hpp>
#include <string>

using namespace std;
using namespace cv;


/**
 * @brief Draw 2D synthetic markers and detect them
 */
class CV_ArucoDetectionSimple : public cvtest::BaseTest {
    public:
    CV_ArucoDetectionSimple();

    protected:
    void run(int);
};


CV_ArucoDetectionSimple::CV_ArucoDetectionSimple() {}


void CV_ArucoDetectionSimple::run(int) {

    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

    // 20 images
    for(int i = 0; i < 20; i++) {

        const int markerSidePixels = 100;
        int imageSize = markerSidePixels * 2 + 3 * (markerSidePixels / 2);

        // draw synthetic image and store marker corners and ids
        vector< vector< Point2f > > groundTruthCorners;
        vector< int > groundTruthIds;
        Mat img = Mat(imageSize, imageSize, CV_8UC1, Scalar::all(255));
        for(int y = 0; y < 2; y++) {
            for(int x = 0; x < 2; x++) {
                Mat marker;
                int id = i * 4 + y * 2 + x;
                aruco::drawMarker(dictionary, id, markerSidePixels, marker);
                Point2f firstCorner =
                    Point2f(markerSidePixels / 2.f + x * (1.5f * markerSidePixels),
                            markerSidePixels / 2.f + y * (1.5f * markerSidePixels));
                Mat aux = img.colRange((int)firstCorner.x, (int)firstCorner.x + markerSidePixels)
                              .rowRange((int)firstCorner.y, (int)firstCorner.y + markerSidePixels);
                marker.copyTo(aux);
                groundTruthIds.push_back(id);
                groundTruthCorners.push_back(vector< Point2f >());
                groundTruthCorners.back().push_back(firstCorner);
                groundTruthCorners.back().push_back(firstCorner + Point2f(markerSidePixels - 1, 0));
                groundTruthCorners.back().push_back(
                    firstCorner + Point2f(markerSidePixels - 1, markerSidePixels - 1));
                groundTruthCorners.back().push_back(firstCorner + Point2f(0, markerSidePixels - 1));
            }
        }
        if(i % 2 == 1) img.convertTo(img, CV_8UC3);

        // detect markers
        vector< vector< Point2f > > corners;
        vector< int > ids;
        Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
        aruco::detectMarkers(img, dictionary, corners, ids, params);

        // check detection results
        for(unsigned int m = 0; m < groundTruthIds.size(); m++) {
            int idx = -1;
            for(unsigned int k = 0; k < ids.size(); k++) {
                if(groundTruthIds[m] == ids[k]) {
                    idx = (int)k;
                    break;
                }
            }
            if(idx == -1) {
                ts->printf(cvtest::TS::LOG, "Marker not detected");
                ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                return;
            }

            for(int c = 0; c < 4; c++) {
                double dist = norm(groundTruthCorners[m][c] - corners[idx][c]);
                if(dist > 0.001) {
                    ts->printf(cvtest::TS::LOG, "Incorrect marker corners position");
                    ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                    return;
                }
            }
        }
    }
}


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
 * @brief Create a synthetic image of a marker with perspective
 */
static Mat projectMarker(Ptr<aruco::Dictionary> &dictionary, int id, Mat cameraMatrix, double yaw,
                         double pitch, double distance, Size imageSize, int markerBorder,
                         vector< Point2f > &corners) {

    // canonical image
    Mat markerImg;
    const int markerSizePixels = 100;
    aruco::drawMarker(dictionary, id, markerSizePixels, markerImg, markerBorder);

    // get rvec and tvec for the perspective
    Mat rvec, tvec;
    getSyntheticRT(yaw, pitch, distance, rvec, tvec);

    const float markerLength = 0.05f;
    vector< Point3f > markerObjPoints;
    markerObjPoints.push_back(Point3f(-markerLength / 2.f, +markerLength / 2.f, 0));
    markerObjPoints.push_back(markerObjPoints[0] + Point3f(markerLength, 0, 0));
    markerObjPoints.push_back(markerObjPoints[0] + Point3f(markerLength, -markerLength, 0));
    markerObjPoints.push_back(markerObjPoints[0] + Point3f(0, -markerLength, 0));

    // project markers and draw them
    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));
    projectPoints(markerObjPoints, rvec, tvec, cameraMatrix, distCoeffs, corners);

    vector< Point2f > originalCorners;
    originalCorners.push_back(Point2f(0, 0));
    originalCorners.push_back(Point2f((float)markerSizePixels, 0));
    originalCorners.push_back(Point2f((float)markerSizePixels, (float)markerSizePixels));
    originalCorners.push_back(Point2f(0, (float)markerSizePixels));

    Mat transformation = getPerspectiveTransform(originalCorners, corners);

    Mat img(imageSize, CV_8UC1, Scalar::all(255));
    Mat aux;
    const char borderValue = 127;
    warpPerspective(markerImg, aux, transformation, imageSize, INTER_NEAREST, BORDER_CONSTANT,
                    Scalar::all(borderValue));

    // copy only not-border pixels
    for(int y = 0; y < aux.rows; y++) {
        for(int x = 0; x < aux.cols; x++) {
            if(aux.at< unsigned char >(y, x) == borderValue) continue;
            img.at< unsigned char >(y, x) = aux.at< unsigned char >(y, x);
        }
    }

    return img;
}



/**
 * @brief Draws markers in perspective and detect them
 */
class CV_ArucoDetectionPerspective : public cvtest::BaseTest {
    public:
    CV_ArucoDetectionPerspective();

    protected:
    void run(int);
};


CV_ArucoDetectionPerspective::CV_ArucoDetectionPerspective() {}


void CV_ArucoDetectionPerspective::run(int) {

    int iter = 0;
    Mat cameraMatrix = Mat::eye(3, 3, CV_64FC1);
    Size imgSize(500, 500);
    cameraMatrix.at< double >(0, 0) = cameraMatrix.at< double >(1, 1) = 650;
    cameraMatrix.at< double >(0, 2) = imgSize.width / 2;
    cameraMatrix.at< double >(1, 2) = imgSize.height / 2;
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

    // detect from different positions
    for(double distance = 0.1; distance <= 0.5; distance += 0.2) {
        for(int pitch = 0; pitch < 360; pitch += 60) {
            for(int yaw = 30; yaw <= 90; yaw += 50) {
                int currentId = iter % 250;
                int markerBorder = iter % 2 + 1;
                iter++;
                vector< Point2f > groundTruthCorners;
                // create synthetic image
                Mat img =
                    projectMarker(dictionary, currentId, cameraMatrix, deg2rad(yaw), deg2rad(pitch),
                                  distance, imgSize, markerBorder, groundTruthCorners);

                // detect markers
                vector< vector< Point2f > > corners;
                vector< int > ids;
                Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
                params->minDistanceToBorder = 1;
                params->markerBorderBits = markerBorder;
                aruco::detectMarkers(img, dictionary, corners, ids, params);

                // check results
                if(ids.size() != 1 || (ids.size() == 1 && ids[0] != currentId)) {
                    if(ids.size() != 1)
                        ts->printf(cvtest::TS::LOG, "Incorrect number of detected markers");
                    else
                        ts->printf(cvtest::TS::LOG, "Incorrect marker id");
                    ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
                    return;
                }
                for(int c = 0; c < 4; c++) {
                    double dist = norm(groundTruthCorners[c] - corners[0][c]);
                    if(dist > 5) {
                        ts->printf(cvtest::TS::LOG, "Incorrect marker corners position");
                        ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                        return;
                    }
                }
            }
        }
    }
}


/**
 * @brief Check max and min size in marker detection parameters
 */
class CV_ArucoDetectionMarkerSize : public cvtest::BaseTest {
    public:
    CV_ArucoDetectionMarkerSize();

    protected:
    void run(int);
};


CV_ArucoDetectionMarkerSize::CV_ArucoDetectionMarkerSize() {}


void CV_ArucoDetectionMarkerSize::run(int) {

    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    int markerSide = 20;
    int imageSize = 200;

    // 10 cases
    for(int i = 0; i < 10; i++) {
        Mat marker;
        int id = 10 + i * 20;

        // create synthetic image
        Mat img = Mat(imageSize, imageSize, CV_8UC1, Scalar::all(255));
        aruco::drawMarker(dictionary, id, markerSide, marker);
        Mat aux = img.colRange(30, 30 + markerSide).rowRange(50, 50 + markerSide);
        marker.copyTo(aux);

        vector< vector< Point2f > > corners;
        vector< int > ids;
        Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();

        // set a invalid minMarkerPerimeterRate
        params->minMarkerPerimeterRate = min(4., (4. * markerSide) / float(imageSize) + 0.1);
        aruco::detectMarkers(img, dictionary, corners, ids, params);
        if(corners.size() != 0) {
            ts->printf(cvtest::TS::LOG, "Error in DetectorParameters::minMarkerPerimeterRate");
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            return;
        }

        // set an valid minMarkerPerimeterRate
        params->minMarkerPerimeterRate = max(0., (4. * markerSide) / float(imageSize) - 0.1);
        aruco::detectMarkers(img, dictionary, corners, ids, params);
        if(corners.size() != 1 || (corners.size() == 1 && ids[0] != id)) {
            ts->printf(cvtest::TS::LOG, "Error in DetectorParameters::minMarkerPerimeterRate");
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            return;
        }

        // set a invalid maxMarkerPerimeterRate
        params->maxMarkerPerimeterRate = min(4., (4. * markerSide) / float(imageSize) - 0.1);
        aruco::detectMarkers(img, dictionary, corners, ids, params);
        if(corners.size() != 0) {
            ts->printf(cvtest::TS::LOG, "Error in DetectorParameters::maxMarkerPerimeterRate");
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            return;
        }

        // set an valid maxMarkerPerimeterRate
        params->maxMarkerPerimeterRate = max(0., (4. * markerSide) / float(imageSize) + 0.1);
        aruco::detectMarkers(img, dictionary, corners, ids, params);
        if(corners.size() != 1 || (corners.size() == 1 && ids[0] != id)) {
            ts->printf(cvtest::TS::LOG, "Error in DetectorParameters::maxMarkerPerimeterRate");
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            return;
        }
    }
}


/**
 * @brief Check error correction in marker bits
 */
class CV_ArucoBitCorrection : public cvtest::BaseTest {
    public:
    CV_ArucoBitCorrection();

    protected:
    void run(int);
};


CV_ArucoBitCorrection::CV_ArucoBitCorrection() {}


void CV_ArucoBitCorrection::run(int) {

    Ptr<aruco::Dictionary> _dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::Dictionary &dictionary = *_dictionary;
    aruco::Dictionary dictionary2 = *_dictionary;
    int markerSide = 50;
    int imageSize = 150;
    Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();

    // 10 markers
    for(int l = 0; l < 10; l++) {
        Mat marker;
        int id = 10 + l * 20;

        Mat currentCodeBytes = dictionary.bytesList.rowRange(id, id + 1);

        // 5 valid cases
        for(int i = 0; i < 5; i++) {
            // how many bit errors (the error is low enough so it can be corrected)
            params->errorCorrectionRate = 0.2 + i * 0.1;
            int errors =
                (int)std::floor(dictionary.maxCorrectionBits * params->errorCorrectionRate - 1.);

            // create erroneous marker in currentCodeBits
            Mat currentCodeBits =
                aruco::Dictionary::getBitsFromByteList(currentCodeBytes, dictionary.markerSize);
            for(int e = 0; e < errors; e++) {
                currentCodeBits.ptr< unsigned char >()[2 * e] =
                    !currentCodeBits.ptr< unsigned char >()[2 * e];
            }

            // add erroneous marker to dictionary2 in order to create the erroneous marker image
            Mat currentCodeBytesError = aruco::Dictionary::getByteListFromBits(currentCodeBits);
            currentCodeBytesError.copyTo(dictionary2.bytesList.rowRange(id, id + 1));
            Mat img = Mat(imageSize, imageSize, CV_8UC1, Scalar::all(255));
            dictionary2.drawMarker(id, markerSide, marker);
            Mat aux = img.colRange(30, 30 + markerSide).rowRange(50, 50 + markerSide);
            marker.copyTo(aux);

            // try to detect using original dictionary
            vector< vector< Point2f > > corners;
            vector< int > ids;
            aruco::detectMarkers(img, _dictionary, corners, ids, params);
            if(corners.size() != 1 || (corners.size() == 1 && ids[0] != id)) {
                ts->printf(cvtest::TS::LOG, "Error in bit correction");
                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                return;
            }
        }

        // 5 invalid cases
        for(int i = 0; i < 5; i++) {
            // how many bit errors (the error is too high to be corrected)
            params->errorCorrectionRate = 0.2 + i * 0.1;
            int errors =
                (int)std::floor(dictionary.maxCorrectionBits * params->errorCorrectionRate + 1.);

            // create erroneous marker in currentCodeBits
            Mat currentCodeBits =
                aruco::Dictionary::getBitsFromByteList(currentCodeBytes, dictionary.markerSize);
            for(int e = 0; e < errors; e++) {
                currentCodeBits.ptr< unsigned char >()[2 * e] =
                    !currentCodeBits.ptr< unsigned char >()[2 * e];
            }

            // dictionary3 is only composed by the modified marker (in its original form)
            Ptr<aruco::Dictionary> _dictionary3 = makePtr<aruco::Dictionary>(
                    dictionary2.bytesList.rowRange(id, id + 1).clone(),
                    dictionary.markerSize,
                    dictionary.maxCorrectionBits);

            // add erroneous marker to dictionary2 in order to create the erroneous marker image
            Mat currentCodeBytesError = aruco::Dictionary::getByteListFromBits(currentCodeBits);
            currentCodeBytesError.copyTo(dictionary2.bytesList.rowRange(id, id + 1));
            Mat img = Mat(imageSize, imageSize, CV_8UC1, Scalar::all(255));
            dictionary2.drawMarker(id, markerSide, marker);
            Mat aux = img.colRange(30, 30 + markerSide).rowRange(50, 50 + markerSide);
            marker.copyTo(aux);

            // try to detect using dictionary3, it should fail
            vector< vector< Point2f > > corners;
            vector< int > ids;
            aruco::detectMarkers(img, _dictionary3, corners, ids, params);
            if(corners.size() != 0) {
                ts->printf(cvtest::TS::LOG, "Error in DetectorParameters::errorCorrectionRate");
                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                return;
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

TEST(CV_ArucoDetectionMarkerSize, algorithmic) {
    CV_ArucoDetectionMarkerSize test;
    test.safe_run();
}

TEST(CV_ArucoBitCorrection, algorithmic) {
    CV_ArucoBitCorrection test;
    test.safe_run();
}
