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

#ifndef __OPENCV_ARUCO_CPP__
#define __OPENCV_ARUCO_CPP__
#ifdef __cplusplus

#include "precomp.hpp"
#include "opencv2/aruco.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "dictionary.cpp"

namespace cv {
namespace aruco {

using namespace std;



/**
 * Detect square candidates in the input image
 */
void _detectCandidates(InputArray _image, OutputArrayOfArrays _candidates, int threshParam,
                       float minLength, OutputArray _thresholdedImage = noArray()) {

    cv::Mat image = _image.getMat();
    CV_Assert(image.cols != 0 && image.rows != 0 &&
              (image.channels() == 1 || image.channels() == 3));

    /// 1. CONVERT TO GRAY
    cv::Mat grey;
    if (image.type() == CV_8UC3)
        cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);
    else
        grey = image;

    /// 2. THRESHOLD
    CV_Assert(threshParam >= 3);
    cv::Mat thresh;
    cv::adaptiveThreshold(grey, thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV,
                          threshParam, 7);
    if (_thresholdedImage.needed())
        thresh.copyTo(_thresholdedImage);

    /// 3. DETECT RECTANGLES
    CV_Assert(minLength > 0);
    int minLengthPixels = minLength * std::max(thresh.cols, thresh.rows);
    cv::Mat contoursImg;
    thresh.copyTo(contoursImg);
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(contoursImg, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    std::vector<std::vector<Point2f> > candidates; // vector of marker candidates
    // now filter list of contours
    for (unsigned int i = 0; i < contours.size(); i++) {
        // check perimeter
        if (contours[i].size() < minLengthPixels)
            continue;

        // check is square and is convex
        vector<Point> approxCurve;
        cv::approxPolyDP(contours[i], approxCurve, double(contours[i].size()) * 0.05, true);
        if (approxCurve.size() != 4 || !cv::isContourConvex(approxCurve))
            continue;

        // check min distance between corners (minimum distance is 10,
        // so minimun square distance is 100)
        float minDistSq = 1e10;
        for (int j = 0; j < 4; j++) {
            float d = (float)(approxCurve[j].x - approxCurve[(j + 1) % 4].x) *
                          (approxCurve[j].x - approxCurve[(j + 1) % 4].x) +
                      (approxCurve[j].y - approxCurve[(j + 1) % 4].y) *
                          (approxCurve[j].y - approxCurve[(j + 1) % 4].y);
            minDistSq = std::min(minDistSq, d);
        }
        if (minDistSq < 100)
            continue;

        // check if it is too near to the image border
        bool tooNearBorder = false;
        int maxDistanceToBorder = 3;
        for (int j = 0; j < 4; j++) {
            if (approxCurve[j].x < maxDistanceToBorder || approxCurve[j].y < maxDistanceToBorder ||
                approxCurve[j].x > image.cols - 1 - maxDistanceToBorder ||
                approxCurve[j].y > image.rows - 1 - maxDistanceToBorder)
                tooNearBorder = true;
        }
        if (tooNearBorder)
            continue;

        // if it passes all the test, add to candidates vector
        std::vector<cv::Point2f> currentCandidate;
        currentCandidate.resize(4);
        for (int j = 0; j < 4; j++) {
            currentCandidate[j] = cv::Point2f(approxCurve[j].x, approxCurve[j].y);
        }
        candidates.push_back(currentCandidate);
    }

    /// 4. SORT CORNERS
    for (unsigned int i = 0; i < candidates.size(); i++) {
        double dx1 = candidates[i][1].x - candidates[i][0].x;
        double dy1 = candidates[i][1].y - candidates[i][0].y;
        double dx2 = candidates[i][2].x - candidates[i][0].x;
        double dy2 = candidates[i][2].y - candidates[i][0].y;
        double crossProduct = (dx1 * dy2) - (dy1 * dx2); // clockwise direction

        if (crossProduct < 0.0)
            swap(candidates[i][1], candidates[i][3]);
    }

    /// 5. FILTER OUT NEAR CANDIDATE PAIRS
    std::vector<std::pair<int, int> > nearCandidates;
    for (unsigned int i = 0; i < candidates.size(); i++) {
        for (unsigned int j = i + 1; j < candidates.size(); j++) {
            float distSq = 0;
            for (int c = 0; c < 4; c++)
                distSq += (candidates[i][c].x - candidates[j][c].x) *
                              (candidates[i][c].x - candidates[j][c].x) +
                          (candidates[i][c].y - candidates[j][c].y) *
                              (candidates[i][c].y - candidates[j][c].y);
            distSq /= 4.;
            // if mean square distance is lower than 100, remove the smaller one of the two markers
            // (minimum mean distance = 10)
            if (distSq < 100)
                nearCandidates.push_back(std::pair<int, int>(i, j));
        }
    }

    /// 6. MARK SMALLER CANDIDATES IN NEAR PAIRS TO REMOVE
    std::vector<bool> toRemove(candidates.size(), false);
    for (unsigned int i = 0; i < nearCandidates.size(); i++) {
        float perimeterSq1 = 0, perimeterSq2 = 0;
        for (unsigned int c = 0; c < 4; c++) {
            // check which one is the smaller and remove it
            perimeterSq1 += (candidates[nearCandidates[i].first][c].x -
                             candidates[nearCandidates[i].first][(c + 1) % 4].x) *
                                (candidates[nearCandidates[i].first][c].x -
                                 candidates[nearCandidates[i].first][(c + 1) % 4].x) +
                            (candidates[nearCandidates[i].first][c].y -
                             candidates[nearCandidates[i].first][(c + 1) % 4].y) *
                                (candidates[nearCandidates[i].first][c].y -
                                 candidates[nearCandidates[i].first][(c + 1) % 4].y);
            perimeterSq2 += (candidates[nearCandidates[i].second][c].x -
                             candidates[nearCandidates[i].second][(c + 1) % 4].x) *
                                (candidates[nearCandidates[i].second][c].x -
                                 candidates[nearCandidates[i].second][(c + 1) % 4].x) +
                            (candidates[nearCandidates[i].second][c].y -
                             candidates[nearCandidates[i].second][(c + 1) % 4].y) *
                                (candidates[nearCandidates[i].second][c].y -
                                 candidates[nearCandidates[i].second][(c + 1) % 4].y);
            if (perimeterSq1 > perimeterSq2)
                toRemove[nearCandidates[i].second] = true;
            else
                toRemove[nearCandidates[i].first] = true;
        }
    }

    /// 7. REMOVE EXTRA CANDIDATES
    int totalRemaining = 0;
    for (unsigned int i = 0; i < toRemove.size(); i++)
        if (!toRemove[i])
            totalRemaining++;
    _candidates.create(totalRemaining, 1, CV_32FC2);
    for (unsigned int i = 0, currIdx = 0; i < candidates.size(); i++) {
        if (toRemove[i])
            continue;
        _candidates.create(4, 1, CV_32FC2, currIdx, true);
        Mat m = _candidates.getMat(currIdx);
        for (int j = 0; j < 4; j++)
            m.ptr<cv::Vec2f>(0)[j] = candidates[i][j];
        currIdx++;
    }
}



/**
 * Identify square candidates according to a marker dictionary
 */
void _identifyCandidates(InputArray _image, InputArrayOfArrays _candidates,
                         DictionaryData dictionary, OutputArrayOfArrays _accepted,
                         OutputArray _ids, OutputArrayOfArrays _rejected = noArray()) {

    int ncandidates = _candidates.total();

    std::vector<cv::Mat> accepted;
    std::vector<cv::Mat> rejected;
    std::vector<int> ids;

    CV_Assert(_image.getMat().cols != 0 && _image.getMat().rows != 0 &&
              (_image.getMat().channels() == 1 || _image.getMat().channels() == 3));

    cv::Mat grey;
    if (_image.getMat().type() == CV_8UC3)
        cv::cvtColor(_image.getMat(), grey, cv::COLOR_BGR2GRAY);
    else
        grey = _image.getMat();

    // try to identify each candidate
    for (int i = 0; i < ncandidates; i++) {
        int currId;
        cv::Mat currentCandidate = _candidates.getMat(i);
        if (dictionary.identify(grey, currentCandidate, currId)) {
            accepted.push_back(currentCandidate);
            ids.push_back(currId);
        } else
            rejected.push_back(_candidates.getMat(i));
    }

    // parse output
    _accepted.create((int)accepted.size(), 1, CV_32FC2);
    for (unsigned int i = 0; i < accepted.size(); i++) {
        _accepted.create(4, 1, CV_32FC2, i, true);
        Mat m = _accepted.getMat(i);
        accepted[i].copyTo(m);
    }

    _ids.create((int)ids.size(), 1, CV_32SC1);
    for (unsigned int i = 0; i < ids.size(); i++)
        _ids.getMat().ptr<int>(0)[i] = ids[i];

    if (_rejected.needed()) {
        _rejected.create((int)rejected.size(), 1, CV_32FC2);
        for (unsigned int i = 0; i < rejected.size(); i++) {
            _rejected.create(4, 1, CV_32FC2, i, true);
            Mat m = _rejected.getMat(i);
            rejected[i].copyTo(m);
        }
    }
}



/**
  * Return object points for the system centered in a single marker, given the marker length
  */
void _getSingleMarkerObjectPoints(float markerLength, OutputArray _objPoints) {

    CV_Assert(markerLength > 0);

    _objPoints.create(4, 1, CV_32FC3);
    cv::Mat objPoints = _objPoints.getMat();
    objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength / 2., markerLength / 2., 0);
    objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength / 2., markerLength / 2., 0);
    objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength / 2., -markerLength / 2., 0);
    objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength / 2., -markerLength / 2., 0);
}



/**
  */
const DictionaryData &_getDictionaryData(DICTIONARY name) {
    switch (name) {
    case DICT_ARUCO:
        return DICT_ARUCO_DATA;
    }
    return DictionaryData();
}



/**
  */
void detectMarkers(InputArray _image, DICTIONARY dictionary, OutputArrayOfArrays _corners,
                   OutputArray _ids, OutputArrayOfArrays _rejectedImgPoints, int threshParam,
                   float minLength) {

    cv::Mat grey;
    if (_image.getMat().type() == CV_8UC3)
        cv::cvtColor(_image.getMat(), grey, cv::COLOR_BGR2GRAY);
    else
        grey = _image.getMat();

    /// STEP 1: Detect marker candidates
    std::vector<std::vector<cv::Point2f> > candidates;
    _detectCandidates(grey, candidates, threshParam, minLength);

    /// STEP 2: Check candidate codification (identify markers)
    DictionaryData dictionaryData = _getDictionaryData(dictionary);
    _identifyCandidates(grey, candidates, dictionaryData, _corners, _ids, _rejectedImgPoints);

    for (int i = 0; i < _corners.total(); i++) {
        /// STEP 3: Corner refinement
        cv::cornerSubPix(grey, _corners.getMat(i), cvSize(5, 5), cvSize(-1, -1),
                         cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.1));
    }
}



/**
  */
void estimatePoseSingleMarkers(InputArrayOfArrays _corners, float markersize,
                               InputArray _cameraMatrix, InputArray _distCoeffs,
                               OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs) {

    cv::Mat markerObjPoints;
    _getSingleMarkerObjectPoints(markersize, markerObjPoints);
    int nMarkers = _corners.total();
    _rvecs.create(nMarkers, 1, CV_32FC1);
    _tvecs.create(nMarkers, 1, CV_32FC1);

    // for each marker, calculate its pose
    for (int i = 0; i < nMarkers; i++) {
        _rvecs.create(3, 1, CV_64FC1, i, true);
        _tvecs.create(3, 1, CV_64FC1, i, true);
        cv::solvePnP(markerObjPoints, _corners.getMat(i), _cameraMatrix, _distCoeffs,
                     _rvecs.getMat(i), _tvecs.getMat(i));
    }
}



/**
  * Given a board configuration and a set of detected markers, returns the corresponding
  * image points and object points to call solvePnP
  */
void _getBoardObjectAndImagePoints(Board board, InputArray _detectedIds,
                                   InputArrayOfArrays _detectedCorners,
                                   OutputArray _imgPoints, OutputArray _objPoints) {

    int nDetectedMarkets = _detectedIds.total();

    std::vector<cv::Point3f> objPnts;
    objPnts.reserve(nDetectedMarkets);

    std::vector<cv::Point2f> imgPnts;
    imgPnts.reserve(nDetectedMarkets);

    // look for detected markers that belong to the board and get their information
    for (int i = 0; i < nDetectedMarkets; i++) {
        int currentId = _detectedIds.getMat().ptr<int>(0)[i];
        for (int j = 0; j < board.ids.size(); j++) {
            if (currentId == board.ids[j]) {
                for (int p = 0; p < 4; p++) {
                    objPnts.push_back(board.objPoints[j][p]);
                    imgPnts.push_back(_detectedCorners.getMat(i).ptr<cv::Point2f>(0)[p]);
                }
            }
        }
    }

    // create output
    _objPoints.create((int)objPnts.size(), 1, CV_32FC3);
    for (int i = 0; i < objPnts.size(); i++)
        _objPoints.getMat().ptr<cv::Point3f>(0)[i] = objPnts[i];

    _imgPoints.create((int)objPnts.size(), 1, CV_32FC2);
    for (int i = 0; i < imgPnts.size(); i++)
        _imgPoints.getMat().ptr<cv::Point2f>(0)[i] = imgPnts[i];
}



/**
  */
void estimatePoseBoard(InputArrayOfArrays _corners, InputArray _ids, Board board,
                       InputArray _cameraMatrix, InputArray _distCoeffs, OutputArray _rvec,
                       OutputArray _tvec) {

    cv::Mat objPoints, imgPoints;
    _getBoardObjectAndImagePoints(board, _ids, _corners, imgPoints, objPoints);

    _rvec.create(3, 1, CV_64FC1);
    _tvec.create(3, 1, CV_64FC1);
    cv::solvePnP(objPoints, imgPoints, _cameraMatrix, _distCoeffs, _rvec, _tvec);
}



/**
 */
Board createPlanarBoard(int width, int height, float markerSize, float markerSeparation,
                        DICTIONARY dictionary) {
    Board res;
    int totalMarkers = width * height;
    res.ids.resize(totalMarkers);
    res.objPoints.reserve(totalMarkers);

    // fill ids with first identifiers
    for (int i = 0; i < totalMarkers; i++)
        res.ids[i] = i;

    // calculate Board objPoints
    float maxY = height * markerSize + (height - 1) * markerSeparation;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::vector<cv::Point3f> corners;
            corners.resize(4);
            corners[0] = cv::Point3f(x * (markerSize + markerSeparation),
                                     maxY - y * (markerSize + markerSeparation), 0);
            corners[1] = corners[0] + cv::Point3f(markerSize, 0, 0);
            corners[2] = corners[0] + cv::Point3f(markerSize, -markerSize, 0);
            corners[3] = corners[0] + cv::Point3f(0, -markerSize, 0);
            res.objPoints.push_back(corners);
        }
    }

    res.dictionary = dictionary;
    return res;
}



/**
 */
void drawDetectedMarkers(InputArray _in, OutputArray _out, InputArrayOfArrays _corners,
                         InputArray _ids, cv::Scalar borderColor) {

    // calculate colors
    cv::Scalar textColor, cornerColor;
    textColor = cornerColor = borderColor;
    swap(textColor.val[0], textColor.val[1]);     // text color just sawp G and R
    swap(cornerColor.val[1], cornerColor.val[2]); // corner color just sawp G and B

    _out.create(_in.size(), _in.type());
    cv::Mat outImg = _out.getMat();
    _in.getMat().copyTo(outImg);

    int nMarkers = _corners.total();
    for (int i = 0; i < nMarkers; i++) {
        cv::Mat currentMarker = _corners.getMat(i);

        // draw marker sides
        for (int j = 0; j < 4; j++) {
            cv::Point2f p0, p1;
            p0 = currentMarker.ptr<cv::Point2f>(0)[j];
            p1 = currentMarker.ptr<cv::Point2f>(0)[(j + 1) % 4];
            cv::line(outImg, p0, p1, borderColor, 2);
        }
        // draw first corner mark
        cv::rectangle(outImg, currentMarker.ptr<cv::Point2f>(0)[0] - Point2f(3, 3),
                      currentMarker.ptr<cv::Point2f>(0)[0] + Point2f(3, 3), cornerColor, 2,
                      cv::LINE_AA);
        // draw ID
        if (_ids.total() != 0) {
            Point2f cent(0, 0);
            for (int p = 0; p < 4; p++)
                cent += currentMarker.ptr<cv::Point2f>(0)[p];
            cent = cent / 4.;
            stringstream s;
            s << "id=" << _ids.getMat().ptr<int>(0)[i];
            putText(outImg, s.str(), cent, cv::FONT_HERSHEY_SIMPLEX, 0.5, textColor, 2);
        }
    }
}



/**
 */
void drawAxis(InputArray _in, OutputArray _out, InputArray _cameraMatrix, InputArray _distCoeffs,
              InputArray _rvec, InputArray _tvec, float length) {

    _out.create(_in.size(), _in.type());
    cv::Mat outImg = _out.getMat();
    _in.getMat().copyTo(outImg);

    std::vector<cv::Point3f> axisPoints;
    axisPoints.push_back(cv::Point3f(0, 0, 0));
    axisPoints.push_back(cv::Point3f(length, 0, 0));
    axisPoints.push_back(cv::Point3f(0, length, 0));
    axisPoints.push_back(cv::Point3f(0, 0, length));
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(axisPoints, _rvec, _tvec, _cameraMatrix, _distCoeffs, imagePoints);

    cv::line(outImg, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 3);
    cv::line(outImg, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0), 3);
    cv::line(outImg, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 3);
}



/**
 */
void drawMarker(DICTIONARY dictionary, int id, int sidePixels, OutputArray _img) {
    DictionaryData dictionaryData = _getDictionaryData(dictionary);
    dictionaryData.drawMarker(id, sidePixels, _img);
}



/**
 */
void drawPlanarBoard(Board board, cv::Size outSize, OutputArray _img) {

    DictionaryData dictData = _getDictionaryData(board.dictionary);

    _img.create(outSize, CV_8UC1);
    cv::Mat out = _img.getMat();
    out.setTo(cv::Scalar::all(255));
    cv::Mat outNoMargins = out.colRange(2, out.cols - 2).rowRange(2, out.rows - 2);

    // calculate max and min values in XY plane
    CV_Assert(board.objPoints.size() > 0);
    float minX, maxX, minY, maxY;
    minX = maxX = board.objPoints[0][0].x;
    minY = maxY = board.objPoints[0][0].y;

    for (int i = 0; i < board.objPoints.size(); i++) {
        for (int j = 0; j < 4; j++) {
            minX = min(minX, board.objPoints[i][j].x);
            maxX = max(maxX, board.objPoints[i][j].x);
            minY = min(minY, board.objPoints[i][j].y);
            maxY = max(maxY, board.objPoints[i][j].y);
        }
    }

    float sizeX, sizeY;
    sizeX = maxX - minX;
    sizeY = maxY - minY;

    float xReduction = sizeX / float(outNoMargins.cols);
    float yReduction = sizeY / float(outNoMargins.rows);

    // determine the zone where the markers are placed
    cv::Mat markerZone;
    if (xReduction > yReduction) {
        int nRows = sizeY / xReduction;
        int rowsMargins = (outNoMargins.rows - nRows) / 2;
        markerZone = outNoMargins.rowRange(rowsMargins, outNoMargins.rows - rowsMargins);
    } else {
        int nCols = sizeX / yReduction;
        int colsMargins = (outNoMargins.cols - nCols) / 2;
        markerZone = outNoMargins.colRange(colsMargins, outNoMargins.cols - colsMargins);
    }

    // now paint each marker
    for (int m = 0; m < board.objPoints.size(); m++) {

        // transform corners to markerZone coordinates
        std::vector<cv::Point2f> outCorners;
        outCorners.resize(4);
        for (int j = 0; j < 4; j++) {
            cv::Point2f p0, p1, pf;
            p0 = cv::Point2f(board.objPoints[m][j].x, board.objPoints[m][j].y);
            // remove negativity
            p1.x = p0.x - minX;
            p1.y = p0.y - minY;
            pf.x = p1.x * float(markerZone.cols - 1) / sizeX;
            pf.y = float(markerZone.rows - 1) - p1.y * float(markerZone.rows - 1) / sizeY;
            outCorners[j] = pf;
        }

        // get tiny marker
        int tinyMarkerSize = 10 * dictData.markerSize + 2;
        cv::Mat tinyMarker;
        dictData.drawMarker(board.ids[m], tinyMarkerSize, tinyMarker);

        // interpolate tiny marker to marker position in markerZone
        cv::Mat inCorners(4, 1, CV_32FC2);
        inCorners.ptr<cv::Point2f>(0)[0] = Point2f(0, 0);
        inCorners.ptr<cv::Point2f>(0)[1] = Point2f(tinyMarker.cols, 0);
        inCorners.ptr<cv::Point2f>(0)[2] = Point2f(tinyMarker.cols, tinyMarker.rows);
        inCorners.ptr<cv::Point2f>(0)[3] = Point2f(0, tinyMarker.rows);

        // remove perspective
        cv::Mat transformation = cv::getPerspectiveTransform(inCorners, outCorners);
        cv::Mat aux;
        const char borderValue = 127;
        cv::warpPerspective(tinyMarker, aux, transformation, markerZone.size(), cv::INTER_NEAREST,
                            cv::BORDER_CONSTANT, cv::Scalar::all(borderValue));

        // copy only not-border pixels
        for (int y = 0; y < aux.rows; y++) {
            for (int x = 0; x < aux.cols; x++) {
                if (aux.at<unsigned char>(y, x) == borderValue)
                    continue;
                markerZone.at<unsigned char>(y, x) = aux.at<unsigned char>(y, x);
            }
        }
    }
    cv::Mat img = _img.getMat();
    out.copyTo(img);
}



/**
  */
double calibrateCameraAruco(std::vector<std::vector<std::vector<cv::Point2f> > > corners,
                            std::vector<std::vector<int> > ids, Board board, Size imageSize,
                            InputOutputArray _cameraMatrix, InputOutputArray _distCoeffs,
                            OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs, int flags,
                            TermCriteria criteria) {

    // for each frame, get properly processed imagePoints and objectPoints for the calibrateCamera
    // function
    std::vector<cv::Mat> processedObjectPoints, processedImagePoints;
    int nFrames = corners.size();
    for (int frame = 0; frame < nFrames; frame++) {
        cv::Mat currentImgPoints, currentObjPoints;
        _getBoardObjectAndImagePoints(board, ids[frame], corners[frame], currentImgPoints,
                                      currentObjPoints);
        if (currentImgPoints.total() > 0 && currentObjPoints.total() > 0) {
            processedImagePoints.push_back(currentImgPoints);
            processedObjectPoints.push_back(currentObjPoints);
        }
    }

    return cv::calibrateCamera(processedObjectPoints, processedImagePoints, imageSize,
                               _cameraMatrix, _distCoeffs, _rvecs, _tvecs, flags, criteria);
}


}
}

#endif // cplusplus
#endif // __OPENCV_ARUCO_CPP__
