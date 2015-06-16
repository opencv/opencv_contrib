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

#ifndef __OPENCV_CHARUCO_CPP__
#define __OPENCV_CHARUCO_CPP__
#ifdef __cplusplus

#include "precomp.hpp"
#include "opencv2/aruco/charuco.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>


namespace cv {
namespace aruco {

using namespace std;



/**
 */
void CharucoBoard::draw(cv::Size outSize, OutputArray _img, int marginSize, int borderBits) {

    CV_Assert(outSize.area() > 0);
    CV_Assert(marginSize >= 0);

    _img.create(outSize, CV_8UC1);
    _img.setTo(255);
    cv::Mat out = _img.getMat();
    cv::Mat noMarginsImg = out.colRange(marginSize, out.cols - marginSize)
                              .rowRange(marginSize, out.rows - marginSize);

    double totalLengthX, totalLengthY;
    totalLengthX = _squareLength * _squaresX;
    totalLengthY = _squareLength * _squaresY;

    double xReduction = totalLengthX / double(noMarginsImg.cols);
    double yReduction = totalLengthY / double(noMarginsImg.rows);

    // determine the zone where the chessboard is placed
    cv::Mat chessboardZoneImg;
    if (xReduction > yReduction) {
        int nRows = totalLengthY / xReduction;
        int rowsMargins = (noMarginsImg.rows - nRows) / 2;
        chessboardZoneImg = noMarginsImg.rowRange(rowsMargins, noMarginsImg.rows - rowsMargins);
    } else {
        int nCols = totalLengthX / yReduction;
        int colsMargins = (noMarginsImg.cols - nCols) / 2;
        chessboardZoneImg = noMarginsImg.colRange(colsMargins, noMarginsImg.cols - colsMargins);
    }

    // determine the margins to draw only the markers
    double squareSizePixels = double(chessboardZoneImg.cols) / double(_squaresX);
    double diffSquareMarkerLength = (_squareLength - _markerLength) / 2;
    double diffSquareMarkerLengthPixels = diffSquareMarkerLength * squareSizePixels / _squareLength;

    // draw markers
    cv::Mat markersImg;
    cv::aruco::drawPlanarBoard((*this), chessboardZoneImg.size(), markersImg,
                               diffSquareMarkerLengthPixels, borderBits);

    markersImg.copyTo(chessboardZoneImg);

    // now draw black boards
    for (int y = 0; y < _squaresY; y++) {
        for (int x = 0; x < _squaresX; x++) {

            if (y % 2 != x % 2)
                continue; // white corner, dont do anything

            double startX, startY;
            startX = squareSizePixels * double(x);
            startY = double(chessboardZoneImg.rows) - squareSizePixels * double(y+1);

            cv::Mat squareZone = chessboardZoneImg.rowRange(startY, startY + squareSizePixels)
                                                  .colRange(startX, startX + squareSizePixels);

            squareZone.setTo(0);

        }
    }

}



/**
 */
CharucoBoard CharucoBoard::create(int squaresX, int squaresY, double squareLength,
                                  double markerLength, DICTIONARY dictionary) {

    CV_Assert(squaresX > 1 && squaresY > 1 && markerLength > 0 && squareLength > markerLength);
    CharucoBoard res;

    res._squaresX = squaresX;
    res._squaresY = squaresY;
    res._squareLength = squareLength;
    res._markerLength = markerLength;
    res.dictionary = dictionary;

    double diffSquareMarkerLength = (squareLength - markerLength) / 2;

    // calculate Board objPoints
    for (int y = squaresY-1; y >= 0; y--) {
        for (int x = 0; x < squaresX; x++) {

            if (y % 2 == x % 2)
                continue; // black corner, no marker here

            std::vector<cv::Point3f> corners;
            corners.resize(4);
            corners[0] = cv::Point3f(x * squareLength + diffSquareMarkerLength,
                                     y * squareLength + diffSquareMarkerLength + markerLength, 0);
            corners[1] = corners[0] + cv::Point3f(markerLength, 0, 0);
            corners[2] = corners[0] + cv::Point3f(markerLength, -markerLength, 0);
            corners[3] = corners[0] + cv::Point3f(0, -markerLength, 0);
            res.objPoints.push_back(corners);
            // first ids in dictionary
            int nextId = res.ids.size()+1;
            res.ids.push_back(nextId);
        }
    }

    // now fill chessboardCorners
    for (int y = 0; y < squaresY-1; y++) {
        for (int x = 0; x < squaresX-1; x++) {
            cv::Point3f corner;
            corner.x = (x+1)*squareLength;
            corner.y = (y+1)*squareLength;
            corner.z = 0;
            res.chessboardCorners.push_back(corner);
        }
    }

    return res;
}



/**
  * @brief From all projected chessboard corners, select those inside the image and apply subpixel
  * refinement. Correspondent object points are also returned. Returns number of valid corners.
  */
unsigned int _selectAndRefineChessboardCorners(InputOutputArray _allImgCorners, InputArray _image,
                                               const CharucoBoard &board,
                                               OutputArray _selectedImgCorners,
                                               OutputArray _selectedObjCorners) {

    // filter points outside image
    int minDistToBorder = 2;
    std::vector<cv::Point2f> filteredChessboardImgPoints;
    std::vector<cv::Point3f> filteredChessboardObjPoints;
    cv::Rect innerRect (minDistToBorder, minDistToBorder,
                        _image.getMat().cols - 2 * minDistToBorder,
                        _image.getMat().rows - 2 * minDistToBorder);
    for (unsigned int i=0; i<_allImgCorners.getMat().total(); i++) {
        if (innerRect.contains(_allImgCorners.getMat().ptr<cv::Point2f>(0)[i])) {
            filteredChessboardImgPoints.push_back(_allImgCorners.getMat().ptr<cv::Point2f>(0)[i]);
            filteredChessboardObjPoints.push_back(board.chessboardCorners[i]);
        }
    }

    // if none valid, return 0
    if (filteredChessboardImgPoints.size() == 0) return 0;

    // corner refinement
    cv::Mat grey;
    if (_image.getMat().type() == CV_8UC3)
        cv::cvtColor(_image.getMat(), grey, cv::COLOR_BGR2GRAY);
    else
       _image.getMat().copyTo(grey);
    DetectorParameters params; // use default params for corner refinement
    cv::cornerSubPix(grey, filteredChessboardImgPoints,
                     cvSize(params.cornerRefinementWinSize, params.cornerRefinementWinSize),
                     cvSize(-1, -1),
                     cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
                                    params.cornerRefinementMaxIterations,
                                    params.cornerRefinementMinAccuracy));

    // update refinement in input _allImgCorners
    unsigned int idx = 0;
    for (unsigned int i=0; i<_allImgCorners.getMat().total(); i++) {
        if (innerRect.contains(_allImgCorners.getMat().ptr<cv::Point2f>(0)[i])) {
            _allImgCorners.getMat().ptr<cv::Point2f>(0)[i] = filteredChessboardImgPoints[idx];
            idx++;
        }
    }

    // parse output
    _selectedImgCorners.create((int)filteredChessboardImgPoints.size(), 1, CV_32FC2);
    for (unsigned int i = 0; i < filteredChessboardImgPoints.size(); i++) {
        _selectedImgCorners.getMat().ptr<cv::Point2f>(0)[i] = filteredChessboardImgPoints[i];
    }

    _selectedObjCorners.create((int)filteredChessboardObjPoints.size(), 1, CV_32FC3);
    for (unsigned int i = 0; i < filteredChessboardObjPoints.size(); i++) {
        _selectedObjCorners.getMat().ptr<cv::Point3f>(0)[i] = filteredChessboardObjPoints[i];
    }

    return filteredChessboardImgPoints.size();
}



/**
  */
bool estimatePoseCharucoBoard(InputArrayOfArrays _corners, InputArray _ids, InputArray _image,
                              const CharucoBoard &board, InputArray _cameraMatrix,
                              InputArray _distCoeffs, OutputArray _rvec, OutputArray _tvec,
                              OutputArray _chessboardCorners) {

    CV_Assert(_image.getMat().channels() == 1 || _image.getMat().channels() == 3);
    CV_Assert(_corners.total() == _ids.total() && _ids.total() > 0);

    // approximated pose estimation
    cv::Mat approximatedRvec, approximatedTvec;
    int detectedBoardMarkers;
    detectedBoardMarkers = cv::aruco::estimatePoseBoard(_corners, _ids, board, _cameraMatrix,
                                                        _distCoeffs, approximatedRvec,
                                                        approximatedTvec);

    if(detectedBoardMarkers == 0)
        return false;


    // project chessboard corners
    cv::Mat allChessboardImgPoints, filteredChessboardImgPoints, filteredChessboardObjPoints;
    cv::projectPoints(board.chessboardCorners, approximatedRvec, approximatedTvec, _cameraMatrix,
                      _distCoeffs, allChessboardImgPoints);


    unsigned int nRefinedCorners;
    nRefinedCorners = _selectAndRefineChessboardCorners(allChessboardImgPoints, _image, board,
                                                        filteredChessboardImgPoints,
                                                        filteredChessboardObjPoints);

    if (nRefinedCorners < 4) return false;


    // final pose estimation
    cv::solvePnP(filteredChessboardObjPoints, filteredChessboardImgPoints, _cameraMatrix,
                 _distCoeffs, _rvec, _tvec);

    if (_chessboardCorners.needed())
        allChessboardImgPoints.copyTo(_chessboardCorners);


    return true;
}



/**
  */
double calibrateCameraCharuco(const std::vector<std::vector<std::vector<Point2f> > > &corners,
                              const std::vector<std::vector<int> > & ids,
                              InputArrayOfArrays _images, const CharucoBoard &board,
                              InputOutputArray _cameraMatrix, InputOutputArray _distCoeffs,
                              OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs,
                              OutputArrayOfArrays _chessboardCorners, int flags,
                              TermCriteria criteria) {

    CV_Assert(_images.total() > 0);
    CV_Assert(corners.size() == ids.size());
    CV_Assert(corners.size() == _images.total());

    cv::Size imageSize = _images.getMat(0).size();

    // calibrate using aruco markers
    cv::Mat approximatedCameraMatrix, approximatedDistCoeffs;
    std::vector<cv::Mat> approximatedRvecs, approximatedTvecs;
    cv::aruco::calibrateCameraAruco(corners, ids, board, imageSize, approximatedCameraMatrix,
                                    approximatedDistCoeffs, approximatedRvecs, approximatedTvecs,
                                    flags, criteria);


    std::vector<cv::Mat> allImgCorners, allObjCorners;



    int nFrames = corners.size();

    if (_chessboardCorners.needed())
        _chessboardCorners.create(nFrames, 1, CV_32FC2);

    for (int frame = 0; frame < nFrames; frame++) {

        cv::Mat currentAllImgCorners;
        cv::Mat currentFilteredImgCorners, currentFilteredObjCorners;
        cv::projectPoints(board.chessboardCorners, approximatedRvecs[frame],
                          approximatedTvecs[frame], approximatedCameraMatrix,
                          approximatedDistCoeffs, currentAllImgCorners);

        _selectAndRefineChessboardCorners(currentAllImgCorners, _images.getMat(frame), board,
                                          currentFilteredImgCorners, currentFilteredObjCorners);

        if (_chessboardCorners.needed()) {
            _chessboardCorners.create(currentAllImgCorners.total(), 1, CV_32FC2, frame, true);
            cv::Mat cornersMat = _chessboardCorners.getMat(frame);
            currentAllImgCorners.copyTo(cornersMat);
        }

        // if none valid, just skip this frame for calibration
        if(currentFilteredImgCorners.total() == 0) continue;

        allImgCorners.push_back(currentFilteredImgCorners);
        allObjCorners.push_back(currentFilteredObjCorners);

    }

    return cv::calibrateCamera(allObjCorners, allImgCorners, imageSize, _cameraMatrix, _distCoeffs,
                               _rvecs, _tvecs, flags, criteria);
}



}
}

#endif // cplusplus
#endif // __OPENCV_CHARUCO_CPP__
