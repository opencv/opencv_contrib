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
#include "opencv2/charuco.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>


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

            if(y % 2 != x % 2)
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
CharucoBoard CharucoBoard::create(int squaresX, int squaresY, float squareLength,
                                  float markerLength, DICTIONARY dictionary) {

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

            if(y % 2 == x % 2)
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

    return res;
}



}
}

#endif // cplusplus
#endif // __OPENCV_CHARUCO_CPP__
