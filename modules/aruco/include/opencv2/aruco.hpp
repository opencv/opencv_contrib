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


#ifndef __OPENCV_ARUCO_HPP__
#define __OPENCV_ARUCO_HPP__

#include <opencv2/core.hpp>
#include <vector>

#include "dictionary.hpp"
#include "board.hpp"


/**
 * @defgroup aruco ArUco Marker Detection
 * This module is dedicated to square fiducial marker (also known as Augmented Reality Markers) detection.
 * These markers are useful for easy, fast and robust camera pose estimation.
 * The implementation is based on the ArUco Library TODO reference
 * The main functionalities are:
 * - Detection of markers in a image
 * - Pose estimation from a single marker or from a board/set of markers
 * - Detection of ChArUco board for high subpixel accuracy
 * - Camera calibration from both, ArUco boards and ChArUco boards.
 * - Detection of ChArUco-based markers
 * The samples directory includes easy examples of how to use the module:
 * - marker_detector.cpp - simple marker detection
 * - marker_detector_pose.cpp - marker detection and pose estimation of each marker individually
 * - board_detector.cpp - marker detection and pose estimation of a marker board
 * - charuco_marker_detector.cpp - detection of charuco-based markers and pose estimation.
 * - charuco_calibration.cpp - camera calibration util based on charuco boards
 * - charuco_calibration_online.cpp - online, automatic and incremental calibration using charuco boards
*/

namespace cv { namespace aruco {


//! @addtogroup aruco
//! @{
  
  

/**
 * @brief Basic marker detection
 *
 * @param image input image
 * @param dictionary indicates the type of markers that will be searched
 * @param imgPoints vector of detected marker corners. For each marker, its four corners are provided,
 * (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the dimensions of this array is Nx4.   
 * The order of the corners is clockwise.                
 * @param ids vector of identifiers of the detected markers. The identifier is of type int (e.g. std::vector<int>).
 * For N detected markers, the size of ids is also N. 
 * The identifiers have the same order than the markers in the imgPoints array.
 * @param threshParam window size for adaptative thresholding. A larger param can slow down the detection process.
 * A low param can produce false negatives during the detection.
 * @param minLenght minimum size of candidates contour lenght. It is indicated as a ratio
 * respect to the largest dimension of the input image. Markers whose perimeter is lower than the correspoding value
 * wont be detected. A low value can slow down the detection process
 * 
 * Performs marker detection in the input image. Only markers included in the specific dictionary are searched.
 * For each detected marker, it returns the 2D position of its corner in the image and its corresponding identifier.
 * Note that this function does not perform pose estimation. 
 * @see estimatePoseSingleMarkers,  estimatePoseBoard
 * 
 */
CV_EXPORTS void detectMarkers(InputArray image, Dictionary dictionary, OutputArrayOfArrays imgPoints,
                       OutputArray ids, int threshParam=21,float minLenght=0.03);

/**
 * @brief Overload detectMarkers function for predefined dictionaries
 **/
CV_EXPORTS void detectMarkers(InputArray image, PREDEFINED_DICTIONARIES dictionary, OutputArrayOfArrays imgPoints,
                       OutputArray ids, int threshParam=21,float minLenght=0.03);





/**
 * @brief Pose estimation for single markers
 *
 * @param imgPoints vector of already detected markers corners. For each marker, its four corners are provided,
 * (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the dimensions of this array should be Nx4. 
 * The order of the corners should be clockwise.
 * @see detectMarkers
 * @param markerSize the lenght of the markers' side. The returning translation vectors will be in the same unit.
 * Normally, unit is meters.
 * @param cameraMatrix input 3x3 floating-point camera matrix 
 * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
 * @param distCoeffs vector of distortion coefficients
 * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
 * @param rvecs array of output rotation vectors (@see Rodrigues) (e.g. std::vector<cv::Mat>>).
 * Each element in rvecs corresponds to the specific marker in imgPoints.
 * @param tvecs array of output translation vectors (e.g. std::vector<cv::Mat>>).
 * Each element in tvecs corresponds to the specific marker in imgPoints.
 * 
 * This function receives the detected markers and returns their pose estimation respect to the camera individually.
 * So for each marker, one rotation and translation vector is returned.
 * The returned transformation is the one that transforms points from each marker coordinate system to the camera
 * coordinate system. 
 * The marker corrdinate system is centered on the middle of the marker, with the Z axis perpendicular
 * to the marker plane. 
 * The coordinates of the four corners of the marker in its own coordinate system are:
 * (-markerSize/2, markerSize/2, 0), (markerSize/2, markerSize/2, 0), 
 * (markerSize/2, -markerSize/2, 0), (-markerSize/2, -markerSize/2, 0)
 */
CV_EXPORTS void estimatePoseSingleMarkers(InputArrayOfArrays imgPoints, float markerSize, InputArray cameraMatrix,
                                          InputArray distCoeffs, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs);





/**
 * @brief Pose estimation for a board of markers
 *
 * @param imgPoints vector of already detected markers corners. For each marker, its four corners are provided,
 * (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the dimensions of this array should be Nx4.
 * The order of the corners should be clockwise.* 
 * @param board layout of markers in the board. The layout is composed by the marker identifiers and the positions 
 * of each marker corner in the board reference system.
 * @param cameraMatrix input 3x3 floating-point camera matrix 
 * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
 * @param distCoeffs vector of distortion coefficients
 * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
 * @param rvec Output vector (e.g. cv::Mat) corresponding to the rotation vector of the board (@see Rodrigues).
 * @param tvec Output vector (e.g. cv::Mat) corresponding to the translation vector of the board.
 * 
 * This function receives the detected markers and returns the pose of a marker board composed by those markers.
 * A Board of marker has a single world coordinate system which is defined by the board layout.
 * The returned transformation is the one that transforms points from the board coordinate system to the 
 * camera coordinate system.
 * Input markers that are not included in the board layout are ignored.
 */
CV_EXPORTS void estimatePoseBoard(InputArrayOfArrays imgPoints, InputArray ids, Board board,
                                          InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec, OutputArray tvec);




/**
 * @brief Draw detected markers in image
 * 
 * @param in input image
 * @param out output image. It will be a copy of in but the markers will be painted on.
 * @param markersCorners positions of marker corners on input image. (e.g std::vector<std::vector<cv::Point2f> > )
 * For N detected markers, the dimensions of this array should be Nx4.
 * The order of the corners should be clockwise.
 * @param ids vector of identifiers for markers in markersCorners . Optional, if not provided, ids are not painted.
 * 
 * Given an array of detected marker corners and its corresponding ids, this functions draws the markers in the image.
 * The marker borders are painted and the markers identifiers if provided.
 */
CV_EXPORTS void drawDetectedMarkers(InputArray in,  OutputArray out, InputArrayOfArrays markersCorners, InputArray ids=noArray());




/**
 * @brief Draw colored axis in image
 *
 * @param image
 * @param cameraMatrix
 * @param distCoeffs
 * @param rvec
 * @param tvec
 * @param lenght axis lenght in meters
 * 
 * 
 */
CV_EXPORTS void drawAxis(InputOutputArray image, InputArray cameraMatrix, InputArray distCoeffs, InputArray rvec, InputArray tvec, float lenght);



/**
 * @brief Draw a canonical marker image
 *
 * @param dict
 * @param id
 * @param sidePixels
 * @param img
 * @return void
 */
CV_EXPORTS void drawMarker(Dictionary dict, int id, int sidePixels, OutputArray img);
CV_EXPORTS void drawMarker(PREDEFINED_DICTIONARIES dict, int id, int sidePixels, OutputArray img);



/**
 * @brief Draw a planar board
 *
 * @param board
 * @param dict
 * @param outSize
 * @param img
 * @return void
 */
CV_EXPORTS void drawPlanarBoard(Board board, Dictionary dict, cv::Size outSize, OutputArray img);
CV_EXPORTS void drawPlanarBoard(Board board, PREDEFINED_DICTIONARIES dict, cv::Size outSize, OutputArray img);


//CV_EXPORTS void calibrateCamera(InputArrayOfArrays images, Board board, cv::Mat& cameraMatrix, cv::Mat& distCoeffs, OutputArrayOfArrays imgPoints, OutputArrayOfArrays ids, OutputArray tvecs, OutputArray tvecs, int threshParam, int minLenght)




//! @}

}} 


#endif
