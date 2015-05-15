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

#include <vector>

namespace cv{ namespace aruco{

using namespace std;





/**
 * @brief detect marker candidates
 * 
 * @param image input image
 * @param candidates return candidate corners positions
 * @param threshParam window size for adaptative thresholding
 * @param minLenght minimum size of candidates contour lenght. It is indicated as a ratio 
 *                  respect to the largest image dimension
 * @param thresholdedImage if set, returns the thresholded image for debugging purposes.
 * @return void
 */
void _detectArucoCandidates(InputArray image, OutputArrayOfArrays candidates, int threshParam,
                           int minLenght, OutputArray thresholdedImage=noArray()) {
  /// TODO
}




/**
 * @brief identify a vector of marker candidates based on the dictionary codification
 * 
 * @param image input image
 * @param candidates candidate corners positions
 * @param dictionary
 * @param accepted returns vector of accepted marker corners
 * @param ids returns vector of accepted markers ids
 * @param rejected ... if set, return vector of rejected markers
 * @return void
 */
void _identifyArucoCandidates(InputArray image, InputArrayOfArrays candidates, 
                             Dictionary dictionary, OutputArrayOfArrays accepted, OutputArray ids, 
                             OutputArrayOfArrays rejected=noArray()) {
    for(int i=0; i<candidates.size; i++) {
        int currId;
        if( dictionary.identify(image,candidates[i],currId) ) {
            accepted.push_back(candidates[i]);
            ids.push_back(currId);
        }
        else rejected.push_back(candidates[i]);
    }
}



/**
 * @brief Given the marker size, it returns the vector of object points for pose estimation
 * 
 * @param markerSize size of marker in meters
 * @param objPnts vector of 4 3d points
 * @return void
 */
void getSingleMarkerObjectPoints(float markerSize, OutputArray objPnts) {
  /// TODO
}




/**
 */
void detectArucoSingle(InputArray image, InputArray cameraMatrix, InputArray distCoeffs,
                       float markersize, Dictionary dictionary, OutputArrayOfArrays imgPoints,
                       OutputArray ids, OutputArray Rvec, OutputArray Tvec, 
                       int threshParam, float minLenght) {

    // STEP 1: Detect marker candidates
    std::vector<std::vector<cv::Point2f> > candidates;
    detectArucoCandidates(image,candidates,threshParam,minLenght);

    // STEP 2: Check candidate codification (identify markers)
    std::vector<std::vector<cv::Point2f> > accepted;
    std::vector< int > ids;
    identifyArucoCandidates(candidates, dictionary, accepted, ids);
	
    
    for(int i=0; i<accepted.size; i++) {
        // STEP 3: Corner refinement
        cv::cornerSubpix...

        // STEP 4: Pose Estimation
        
    }

}



/**
 */
void detectArucoBoard(InputArray image, InputArray cameraMatrix, InputArray distCoeffs, 
                      Board board, OutputArrayOfArrays imgPoints, OutputArray ids,
                      OutputArray rvec, OutputArray tvec, int threshParam, float minLenght) {

    // STEP 1: Detect marker candidates
    std::vector<std::vector<cv::Point2f> > candidates;
    _detectArucoCandidates(image,candidates,threshParam1,threshParam2,minLenght,maxLenght);

    // STEP 2: Check candidate codification (identify markers)
    std::vector<std::vector<cv::Point2f> > accepted;
    std::vector< int > ids;
    _identifyArucoCandidates(candidates, dictionary, accepted, ids);

    // STEP 3: Corner refinement
    for(int i=0; i<accepted.size; i++) {
        
        cv::cornerSubpix...

    }

    // STEP 4: Pose Estimation

}


/**
 */
void drawArucoDetectedMarkers(InputArray image, InputArrayOfArrays markers, InputArray ids) {
    /// TODO
}



/**
 */
bool Dictionary::identify(InputArray image, InputArray imgPoints, int &idx) {
    /// TODO
}



/**
 */
void Dictionary::printMarker(InputOutputArray img, int id) {
    /// TODO
}


/**
 */
void Board::printBoard(InputOutputArray img) {
    /// TODO
}


/**
 */
static Board Board::createPlanarBoard(int width, int height, float markerSize, 
				float markerSeparation, Dictionary dictionary) {
    /// TODO
}


}}

#endif // cplusplus
#endif // __OPENCV_ARUCO_CPP__

