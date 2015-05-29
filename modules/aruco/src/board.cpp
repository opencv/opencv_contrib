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

#ifndef __OPENCV_ARUCO_BOARD_CPP__
#define __OPENCV_ARUCO_BOARD_CPP__
#ifdef __cplusplus

#include "precomp.hpp"
#include "opencv2/aruco.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "predefined_dictionaries.cpp"

#include <iostream>


namespace cv{ namespace aruco{

using namespace std;



/**
 */
void Board::drawBoard(InputOutputArray img) {
    /// TODO
}

/**
  */
void Board::getObjectAndImagePoints(InputArray detectedIds, InputArrayOfArrays detectedImagePoints, OutputArray imagePoints, OutputArray objectPoints) {
    std::vector<cv::Point3f> objPnts;
    objPnts.reserve(detectedIds.total());

    std::vector<cv::Point2f> imgPnts;
    imgPnts.reserve(detectedIds.total());

    for(int i=0; i<detectedIds.getMat().total(); i++) {
        int currentId = detectedIds.getMat().ptr<int>(0)[i];
        for(int j=0; j<ids.size(); j++) {
            if(currentId == ids[j]) {
                for(int p=0; p<4; p++) {
                    objPnts.push_back( objPoints[j][p] );
                    imgPnts.push_back( detectedImagePoints.getMat(i).ptr<cv::Point2f>(0)[p] );
                }
            }
        }
    }
    objectPoints.create((int)objPnts.size(), 1, CV_32FC3);
    for(int i=0; i<objPnts.size(); i++) objectPoints.getMat().ptr<cv::Point3f>(0)[i] = objPnts[i];

    imagePoints.create((int)objPnts.size(), 1, CV_32FC2);
    for(int i=0; i<imgPnts.size(); i++) imagePoints.getMat().ptr<cv::Point2f>(0)[i] = imgPnts[i];
}


/**
 */
Board Board::createPlanarBoard(int width, int height, float markerSize, float markerSeparation) {
    Board res;
    int totalMarkers = width*height;
    res.ids.resize(totalMarkers);
    res.objPoints.reserve(totalMarkers);
    for(int i=0; i<totalMarkers; i++) res.ids[i] = i;

    float maxY = height*markerSize + (height-1)*markerSeparation;
    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            std::vector<cv::Point3f> corners;
            corners.resize(4);
            corners[0] = cv::Point3f(x*(markerSize+markerSeparation), maxY-y*(markerSize+markerSeparation) , 0);
            corners[1] = corners[0]+cv::Point3f(markerSize,0,0);
            corners[2] = corners[0]+cv::Point3f(markerSize,-markerSize,0);
            corners[3] = corners[0]+cv::Point3f(0,-markerSize,0);
            res.objPoints.push_back(corners);
        }
    }
    return res;
}





}}

#endif // cplusplus
#endif // __OPENCV_ARUCO_BOARD_CPP__

