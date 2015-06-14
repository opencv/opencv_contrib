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
  *
  */
DetectorParameters::DetectorParameters() : adaptiveThreshWinSize(21),
                                           adaptiveThreshConstant(7),
                                           minMarkerPerimeterRate(0.03),
                                           maxMarkerPerimeterRate(4.),
                                           polygonalApproxAccuracyRate(0.05),
                                           minCornerDistance(10),
                                           minDistanceToBorder(3),
                                           minMarkerDistance(10),
                                           cornerRefinementWinSize(5),
                                           cornerRefinementMaxIterations(30),
                                           cornerRefinementMinAccuracy(0.1),
                                           markerBorderBits(1),
                                           perspectiveRemovePixelPerCell(8),
                                           perspectiveRemoveIgnoredMarginPerCell(0.13),
                                           maxErroneousBitsInBorderRate(0.04) {

}


/**
  * @brief Convert input image to gray if it is a 3 channels image
  */
void _convertToGrey(InputArray _in, OutputArray _out) {

    CV_Assert(_in.getMat().channels() == 1 || _in.getMat().channels() == 3);

    _out.create(_in.getMat().size(), CV_8UC1);
    if (_in.getMat().type() == CV_8UC3)
        cv::cvtColor(_in.getMat(), _out.getMat(), cv::COLOR_BGR2GRAY);
    else
       _in.getMat().copyTo(_out);
}


/**
  * @brief Threshold input image using adaptive thresholding
  */
void _threshold(InputArray _in, OutputArray _out, int winSize, double constant) {

    CV_Assert(winSize >= 3);

    cv::adaptiveThreshold(_in, _out, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV,
                          winSize, constant);
}


/**
  * @brief Given a tresholded image, find the contours, calculate their polygonal approximation
  * and take those that accomplish some conditions
  */
void _findMarkerContours(InputArray _in, std::vector<std::vector<Point2f> > &candidates,
                         double minPerimeterRate, int maxPerimeterRate, double accuracyRate,
                         double minCornerDistance, int minDistanceToBorder) {

    CV_Assert(minPerimeterRate > 0 && maxPerimeterRate > 0 && accuracyRate > 0 &&
              minCornerDistance > 0 && minDistanceToBorder >= 0);

    int minPerimeterPixels = minPerimeterRate * std::max(_in.getMat().cols, _in.getMat().rows);
    int maxPerimeterPixels = maxPerimeterRate * std::max(_in.getMat().cols, _in.getMat().rows);
    cv::Mat contoursImg;
    _in.getMat().copyTo(contoursImg);
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(contoursImg, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    // now filter list of contours
    for (unsigned int i = 0; i < contours.size(); i++) {
        // check perimeter
        if (contours[i].size() < minPerimeterPixels || contours[i].size() > maxPerimeterPixels)
            continue;

        // check is square and is convex
        vector<Point> approxCurve;
        cv::approxPolyDP(contours[i], approxCurve, double(contours[i].size()) * accuracyRate, true);
        if (approxCurve.size() != 4 || !cv::isContourConvex(approxCurve))
            continue;

        // check min distance between corners (minimum distance is 10,
        // so minimun square distance is 100)
        double minDistSq = std::max(contoursImg.cols, contoursImg.rows) *
                           std::max(contoursImg.cols, contoursImg.rows);
        for (int j = 0; j < 4; j++) {
            double d = (double)(approxCurve[j].x - approxCurve[(j + 1) % 4].x) *
                       (double)(approxCurve[j].x - approxCurve[(j + 1) % 4].x) +
                       (double)(approxCurve[j].y - approxCurve[(j + 1) % 4].y) *
                       (double)(approxCurve[j].y - approxCurve[(j + 1) % 4].y);
            minDistSq = std::min(minDistSq, d);
        }
        if (minDistSq < minCornerDistance * minCornerDistance)
            continue;

        // check if it is too near to the image border
        bool tooNearBorder = false;
        for (int j = 0; j < 4; j++) {
            if (approxCurve[j].x < minDistanceToBorder || approxCurve[j].y < minDistanceToBorder ||
                approxCurve[j].x > contoursImg.cols - 1 - minDistanceToBorder ||
                approxCurve[j].y > contoursImg.rows - 1 - minDistanceToBorder)
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
}


/**
  * @brief Assure order of candidate corners is clockwise direction
  */
void _reorderCandidatesCorners(std::vector<std::vector<Point2f> > &candidates) {

    for (unsigned int i = 0; i < candidates.size(); i++) {
        double dx1 = candidates[i][1].x - candidates[i][0].x;
        double dy1 = candidates[i][1].y - candidates[i][0].y;
        double dx2 = candidates[i][2].x - candidates[i][0].x;
        double dy2 = candidates[i][2].y - candidates[i][0].y;
        double crossProduct = (dx1 * dy2) - (dy1 * dx2); // clockwise direction

        if (crossProduct < 0.0)
            swap(candidates[i][1], candidates[i][3]);
    }
}


/**
  * @brief Check candidates that are too close to each other and remove the smaller one
  */
void _filterTooCloseCandidates(const std::vector<std::vector<Point2f> > &candidatesIn,
                               std::vector<std::vector<Point2f> > &candidatesOut,
                               double minMarkerDistance) {
    CV_Assert(minMarkerDistance > 0);

    std::vector<std::pair<int, int> > nearCandidates;
    for (unsigned int i = 0; i < candidatesIn.size(); i++) {
        for (unsigned int j = i + 1; j < candidatesIn.size(); j++) {
            double distSq = 0;
            for (int c = 0; c < 4; c++)
                distSq += (candidatesIn[i][c].x - candidatesIn[j][c].x) *
                              (candidatesIn[i][c].x - candidatesIn[j][c].x) +
                          (candidatesIn[i][c].y - candidatesIn[j][c].y) *
                              (candidatesIn[i][c].y - candidatesIn[j][c].y);
            distSq /= 4.;
            // if mean square distance is lower than 100, remove the smaller one of the two markers
            // (minimum mean distance = 10)
            if (distSq < minMarkerDistance * minMarkerDistance)
                nearCandidates.push_back(std::pair<int, int>(i, j));
        }
    }

    // mark smaller one in pairs to remove
    std::vector<bool> toRemove(candidatesIn.size(), false);
    for (unsigned int i = 0; i < nearCandidates.size(); i++) {
        double perimeterSq1 = 0, perimeterSq2 = 0;
        for (unsigned int c = 0; c < 4; c++) {
            // check which one is the smaller and remove it
            perimeterSq1 += (candidatesIn[nearCandidates[i].first][c].x -
                             candidatesIn[nearCandidates[i].first][(c + 1) % 4].x) *
                                (candidatesIn[nearCandidates[i].first][c].x -
                                 candidatesIn[nearCandidates[i].first][(c + 1) % 4].x) +
                            (candidatesIn[nearCandidates[i].first][c].y -
                             candidatesIn[nearCandidates[i].first][(c + 1) % 4].y) *
                                (candidatesIn[nearCandidates[i].first][c].y -
                                 candidatesIn[nearCandidates[i].first][(c + 1) % 4].y);
            perimeterSq2 += (candidatesIn[nearCandidates[i].second][c].x -
                             candidatesIn[nearCandidates[i].second][(c + 1) % 4].x) *
                                (candidatesIn[nearCandidates[i].second][c].x -
                                 candidatesIn[nearCandidates[i].second][(c + 1) % 4].x) +
                            (candidatesIn[nearCandidates[i].second][c].y -
                             candidatesIn[nearCandidates[i].second][(c + 1) % 4].y) *
                                (candidatesIn[nearCandidates[i].second][c].y -
                                 candidatesIn[nearCandidates[i].second][(c + 1) % 4].y);
            if (perimeterSq1 > perimeterSq2)
                toRemove[nearCandidates[i].second] = true;
            else
                toRemove[nearCandidates[i].first] = true;
        }
    }

    // remove extra candidates
    candidatesOut.clear();
    int totalRemaining = 0;
    for (unsigned int i = 0; i < toRemove.size(); i++)
        if (!toRemove[i])
            totalRemaining++;
    candidatesOut.resize(totalRemaining);
    for (unsigned int i = 0, currIdx = 0; i < candidatesIn.size(); i++) {
        if (toRemove[i])
            continue;
        candidatesOut[currIdx] = candidatesIn[i];
        currIdx++;
    }
}


/**
 * @brief Detect square candidates in the input image
 */
void _detectCandidates(InputArray _image, OutputArrayOfArrays _candidates,
                       DetectorParameters params, OutputArray _thresholdedImage = noArray()) {

    cv::Mat image = _image.getMat();
    CV_Assert(image.total() != 0);

    /// 1. CONVERT TO GRAY
    cv::Mat grey;
    _convertToGrey(image, grey);

    /// 2. THRESHOLD
    CV_Assert(params.adaptiveThreshWinSize >= 3);
    cv::Mat thresh;
    _threshold(grey, thresh, params.adaptiveThreshWinSize, params.adaptiveThreshConstant);
    if (_thresholdedImage.needed())
        thresh.copyTo(_thresholdedImage);

    /// 3. DETECT RECTANGLES
    std::vector<std::vector<Point2f> > candidates;
    _findMarkerContours(thresh, candidates, params.minMarkerPerimeterRate,
                        params.maxMarkerPerimeterRate, params.polygonalApproxAccuracyRate,
                        params.minCornerDistance, params.minDistanceToBorder);

    /// 4. SORT CORNERS
    _reorderCandidatesCorners(candidates);

    /// 5. FILTER OUT NEAR CANDIDATE PAIRS
    std::vector<std::vector<Point2f> > candidatesOut;
    _filterTooCloseCandidates(candidates, candidatesOut, params.minMarkerDistance);

    // parse output
    _candidates.create(candidatesOut.size(), 1, CV_32FC2);
    for (unsigned int i = 0; i < candidatesOut.size(); i++) {
        _candidates.create(4, 1, CV_32FC2, i, true);
        Mat m = _candidates.getMat(i);
        for (int j = 0; j < 4; j++)
            m.ptr<cv::Vec2f>(0)[j] = candidatesOut[i][j];
    }
}


/**
  * @brief Given an input image and a candidate corners, extract the bits of the candidate, including
  * the border
  */
cv::Mat _extractBits(InputArray _image, InputArray _corners, int markerSize, int markerBorderBits,
                     int cellSize, double cellMarginRate) {

    CV_Assert(_image.getMat().channels() == 1);
    CV_Assert(_corners.total() == 4);
    CV_Assert(markerBorderBits > 0 && cellSize > 0 && cellMarginRate > 0);

    int cellMarginPixels = cellMarginRate*cellSize;
    int markerSizeWithBorders = markerSize + 2*markerBorderBits;
    cv::Mat resultImg; // marker image after removing perspective
    int resultImgSize = markerSizeWithBorders * cellSize;
    cv::Mat resultImgCorners(4, 1, CV_32FC2);
    resultImgCorners.ptr<cv::Point2f>(0)[0] = Point2f(0, 0);
    resultImgCorners.ptr<cv::Point2f>(0)[1] = Point2f(resultImgSize - 1, 0);
    resultImgCorners.ptr<cv::Point2f>(0)[2] = Point2f(resultImgSize - 1, resultImgSize - 1);
    resultImgCorners.ptr<cv::Point2f>(0)[3] = Point2f(0, resultImgSize - 1);

    // remove perspective
    cv::Mat transformation = cv::getPerspectiveTransform(_corners, resultImgCorners);
    cv::warpPerspective(_image, resultImg, transformation,
                        cv::Size(resultImgSize, resultImgSize), cv::INTER_NEAREST);

    // now extract code
    cv::Mat bits(markerSizeWithBorders, markerSizeWithBorders, CV_8UC1, cv::Scalar::all(0));
    cv::threshold(resultImg, resultImg, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    for (unsigned int y = 0; y < markerSizeWithBorders; y++) {
        for (unsigned int x = 0; x < markerSizeWithBorders; x++) {
            int Xstart = x * (cellSize)+cellMarginPixels;
            int Ystart = y * (cellSize)+cellMarginPixels;
            cv::Mat square =
                resultImg(cv::Rect(Xstart, Ystart, cellSize - 2*cellMarginPixels,
                                   cellSize - 2*cellMarginPixels));
            int nZ = countNonZero(square);
            if (nZ > square.total() / 2)
                bits.at<unsigned char>(y, x) = 1;
        }
    }

    return bits;
}


/**
  * @brief Return number of erroneous bits in border, i.e. number of white bits in border.
  */
int _getBorderErrors(const cv::Mat &bits, int markerSize, int borderSize) {

    int sizeWithBorders = markerSize + 2*borderSize;

    CV_Assert(markerSize > 0 && bits.cols == sizeWithBorders && bits.rows == sizeWithBorders);

    int totalErrors = 0;
    for (int y = 0; y < sizeWithBorders; y++) {
        for (int k = 0; k < borderSize; k++) {
            if (bits.ptr<unsigned char>(y)[k] != 0)
                totalErrors++;
            if (bits.ptr<unsigned char>(y)[sizeWithBorders - 1 - k] != 0)
                totalErrors++;
        }
    }
    for (int x = borderSize; x < sizeWithBorders - borderSize; x++) {
        for (int k = 0; k < borderSize; k++) {
            if (bits.ptr<unsigned char>(k)[x] != 0)
                totalErrors++;
            if (bits.ptr<unsigned char>(sizeWithBorders - 1 - k)[x] != 0)
                totalErrors++;
        }
    }
    return totalErrors;
}


/**
 * @brief Tries to identify one candidate given the dictionary
 */
bool identifyOneCandidate(DictionaryData dictionary, InputArray _image, InputOutputArray _corners,
                          int &idx, DetectorParameters params) {

    CV_Assert(_corners.total() == 4);
    CV_Assert(_image.getMat().cols != 0 && _image.getMat().rows);
    CV_Assert(params.markerBorderBits > 0);

    // get bits
    cv::Mat candidateBits = _extractBits(_image, _corners, dictionary.markerSize,
                                         params.markerBorderBits,
                                         params.perspectiveRemovePixelPerCell,
                                         params.perspectiveRemoveIgnoredMarginPerCell);
    int maximumErrorsInBorder = dictionary.markerSize * dictionary.markerSize *
                                params.maxErroneousBitsInBorderRate;
    int borderErrors = _getBorderErrors(candidateBits, dictionary.markerSize,
                                        params.markerBorderBits);
    if (borderErrors > maximumErrorsInBorder)
        return false; // border is wrong
    cv::Mat onlyBits =
        candidateBits.rowRange(params.markerBorderBits,
                               candidateBits.rows - params.markerBorderBits)
                     .colRange(params.markerBorderBits,
                               candidateBits.rows - params.markerBorderBits);

    int rotation;
    if (!dictionary.identify(onlyBits, idx, rotation))
        return false;
    else {
        if (rotation != 0) {
            cv::Mat copyPoints = _corners.getMat().clone();
            for (int j = 0; j < 4; j++)
                _corners.getMat().ptr<cv::Point2f>(0)[j] =
                    copyPoints.ptr<cv::Point2f>(0)[(j + 4 - rotation) % 4];
        }
        return true;
    }
}



/**
 * @brief Identify square candidates according to a marker dictionary
 */
void _identifyCandidates(InputArray _image, InputArrayOfArrays _candidates,
                         const DictionaryData &dictionary, OutputArrayOfArrays _accepted,
                         OutputArray _ids, DetectorParameters params,
                         OutputArrayOfArrays _rejected = noArray()) {

    int ncandidates = _candidates.total();

    std::vector<cv::Mat> accepted;
    std::vector<cv::Mat> rejected;
    std::vector<int> ids;

    CV_Assert(_image.getMat().total() != 0);

    cv::Mat grey;
    _convertToGrey(_image.getMat(), grey);

    // try to identify each candidate
    for (int i = 0; i < ncandidates; i++) {
        int currId;
        cv::Mat currentCandidate = _candidates.getMat(i);
        if (identifyOneCandidate(dictionary, grey, currentCandidate, currId, params)) {
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
  * @brief Return object points for the system centered in a single marker, given the marker length
  */
void _getSingleMarkerObjectPoints(double markerLength, OutputArray _objPoints) {

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
                   OutputArray _ids, DetectorParameters params,
                   OutputArrayOfArrays _rejectedImgPoints) {

    CV_Assert(_image.getMat().total() != 0);

    cv::Mat grey;
    _convertToGrey(_image.getMat(), grey);

    /// STEP 1: Detect marker candidates
    std::vector<std::vector<cv::Point2f> > candidates;
    _detectCandidates(grey, candidates, params);

    /// STEP 2: Check candidate codification (identify markers)
    DictionaryData dictionaryData = _getDictionaryData(dictionary);
    _identifyCandidates(grey, candidates, dictionaryData, _corners, _ids, params,
                        _rejectedImgPoints);

    /// STEP 3: Corner refinement
    CV_Assert(params.cornerRefinementWinSize > 0 && params.cornerRefinementMaxIterations > 0 &&
              params.cornerRefinementMinAccuracy > 0);

    for (int i = 0; i < _corners.total(); i++) {
        cv::cornerSubPix(grey, _corners.getMat(i),
                         cvSize(params.cornerRefinementWinSize, params.cornerRefinementWinSize),
                         cvSize(-1, -1),
                         cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
                                        params.cornerRefinementMaxIterations,
                                        params.cornerRefinementMinAccuracy));
    }
}



/**
  */
void estimatePoseSingleMarkers(InputArrayOfArrays _corners, double markerLength,
                               InputArray _cameraMatrix, InputArray _distCoeffs,
                               OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs) {

    CV_Assert(markerLength > 0);

    cv::Mat markerObjPoints;
    _getSingleMarkerObjectPoints(markerLength, markerObjPoints);
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
  * @brief Given a board configuration and a set of detected markers, returns the corresponding
  * image points and object points to call solvePnP
  */
void _getBoardObjectAndImagePoints(const Board &board, InputArray _detectedIds,
                                   InputArrayOfArrays _detectedCorners,
                                   OutputArray _imgPoints, OutputArray _objPoints) {

    CV_Assert(board.ids.size()==board.objPoints.size());
    CV_Assert(_detectedIds.total() == _detectedCorners.total() );

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
int estimatePoseBoard(InputArrayOfArrays _corners, InputArray _ids, const Board &board,
                      InputArray _cameraMatrix, InputArray _distCoeffs, OutputArray _rvec,
                      OutputArray _tvec) {

    CV_Assert(_corners.total() == _ids.total() );

    cv::Mat objPoints, imgPoints;
    _getBoardObjectAndImagePoints(board, _ids, _corners, imgPoints, objPoints);

    CV_Assert(imgPoints.total() == objPoints.total());

    if (objPoints.total() == 0) // 0 of the detected markers in board
        return 0;

    _rvec.create(3, 1, CV_64FC1);
    _tvec.create(3, 1, CV_64FC1);
    cv::solvePnP(objPoints, imgPoints, _cameraMatrix, _distCoeffs, _rvec, _tvec);

    return objPoints.total()/4;
}




/**
 */
void GridBoard::draw(cv::Size outSize, OutputArray _img, int marginSize, int borderBits) {
    cv::aruco::drawPlanarBoard((*this), outSize, _img, marginSize, borderBits);
}



/**
 */
GridBoard GridBoard::create(int markersX, int markersY, double markerLength,
                            double markerSeparation, DICTIONARY _dictionary) {

    GridBoard res;

    CV_Assert(markersX>0 && markersY>0 && markerLength>0 && markerSeparation>0);

    res._markersX = markersX;
    res._markersY = markersY;
    res._markerLength = markerLength;
    res._markerSeparation = markerSeparation;
    res.dictionary = _dictionary;

    int totalMarkers = markersX * markersY;
    res.ids.resize(totalMarkers);
    res.objPoints.reserve(totalMarkers);

    // fill ids with first identifiers
    for (int i = 0; i < totalMarkers; i++)
        res.ids[i] = i;

    // calculate Board objPoints
    double maxY = markersY * markerLength + (markersY - 1) * markerSeparation;
    for (int y = 0; y < markersY; y++) {
        for (int x = 0; x < markersX; x++) {
            std::vector<cv::Point3f> corners;
            corners.resize(4);
            corners[0] = cv::Point3f(x * (markerLength + markerSeparation),
                                     maxY - y * (markerLength + markerSeparation), 0);
            corners[1] = corners[0] + cv::Point3f(markerLength, 0, 0);
            corners[2] = corners[0] + cv::Point3f(markerLength, -markerLength, 0);
            corners[3] = corners[0] + cv::Point3f(0, -markerLength, 0);
            res.objPoints.push_back(corners);
        }
    }

    return res;
}



/**
 */
void drawDetectedMarkers(InputArray _in, OutputArray _out, InputArrayOfArrays _corners,
                         InputArray _ids, cv::Scalar borderColor) {


    CV_Assert(_in.getMat().cols != 0 && _in.getMat().rows != 0 &&
              (_in.getMat().channels() == 1 || _in.getMat().channels() == 3));
    CV_Assert((_corners.total() == _ids.total()) || _ids.total()==0 );

    // calculate colors
    cv::Scalar textColor, cornerColor;
    textColor = cornerColor = borderColor;
    swap(textColor.val[0], textColor.val[1]);     // text color just sawp G and R
    swap(cornerColor.val[1], cornerColor.val[2]); // corner color just sawp G and B

    _out.create(_in.size(), _in.type());
    cv::Mat outImg = _out.getMat();
    if (_in.getMat().channels()==3) _in.getMat().copyTo(outImg);
    else cv::cvtColor(_in.getMat(), outImg, cv::COLOR_GRAY2BGR);

    int nMarkers = _corners.total();
    for (int i = 0; i < nMarkers; i++) {
        cv::Mat currentMarker = _corners.getMat(i);
        CV_Assert(currentMarker.total()==4 && currentMarker.type()==CV_32FC2);

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
              InputArray _rvec, InputArray _tvec, double length) {

    CV_Assert(_in.getMat().cols != 0 && _in.getMat().rows != 0 &&
              (_in.getMat().channels() == 1 || _in.getMat().channels() == 3));
    CV_Assert(length > 0);

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
void drawMarker(DICTIONARY dictionary, int id, int sidePixels, OutputArray _img, int borderBits) {
    DictionaryData dictionaryData = _getDictionaryData(dictionary);
    dictionaryData.drawMarker(id, sidePixels, _img, borderBits);
}



/**
 */
void drawPlanarBoard(const Board &board, cv::Size outSize, OutputArray _img, int marginSize,
                     int borderBits) {

    CV_Assert(outSize.area() > 0);
    CV_Assert(marginSize >= 0);

    DictionaryData dictData = _getDictionaryData(board.dictionary);

    _img.create(outSize, CV_8UC1);
    cv::Mat out = _img.getMat();
    out.setTo(cv::Scalar::all(255));
    cv::Mat outNoMargins = out.colRange(marginSize, out.cols - marginSize).
                               rowRange(marginSize, out.rows - marginSize);

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

    double sizeX, sizeY;
    sizeX = maxX - minX;
    sizeY = maxY - minY;

    double xReduction = sizeX / double(outNoMargins.cols);
    double yReduction = sizeY / double(outNoMargins.rows);

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
            pf.x = p1.x * double(markerZone.cols - 1) / sizeX;
            pf.y = double(markerZone.rows - 1) - p1.y * double(markerZone.rows - 1) / sizeY;
            outCorners[j] = pf;
        }

        // get tiny marker
        int tinyMarkerSize = 10 * dictData.markerSize + 2;
        cv::Mat tinyMarker;
        dictData.drawMarker(board.ids[m], tinyMarkerSize, tinyMarker, borderBits);

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

}



/**
  */
double calibrateCameraAruco(const std::vector<std::vector<std::vector<cv::Point2f> > > &corners,
                            const std::vector<std::vector<int> > &ids, const Board &board,
                            Size imageSize, InputOutputArray _cameraMatrix,
                            InputOutputArray _distCoeffs, OutputArrayOfArrays _rvecs,
                            OutputArrayOfArrays _tvecs, int flags, TermCriteria criteria) {

    CV_Assert(corners.size()==ids.size());

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
