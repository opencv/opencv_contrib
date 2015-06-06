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
  
  

enum DICTIONARY { DICT_ARUCO=0 };


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
 * @param rejectedImgPoints contains the imgPoints of those squares whose inner code has not a correct codification
 * Useful for debugging purposes.
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
CV_EXPORTS void detectMarkers(InputArray image, DICTIONARY dictionary, OutputArrayOfArrays imgPoints,
                       OutputArray ids, OutputArrayOfArrays rejectedImgPoints=cv::noArray(), int threshParam=21,float minLenght=0.03);





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
 * @brief Board of markers
 *
 * A board is a set of markers in the 3D space with a common cordinate system.
 * The common form of a board of marker is a planar (2D) board, however any 3D layout can be employed.
 * A Board object is composed by:
 * - The object points of the marker corners, i.e. their coordinates respect to the board coordinate system.
 * - The dictionary which indicates the type of markers of the board
 * - The identifier of all the markers in the board.
 */
struct CV_EXPORTS Board {

    // array of object points of all the marker corners in the board
    // each marker include its 4 corners, i.e. for M markers, the size is Mx4
    std::vector< std::vector<cv::Point3f> > objPoints;

    // the dictionary of markers employed for this board
    DICTIONARY dictionary;

    // vector of the identifiers of the markers in the board (same size than objPoints)
    // The identifiers refers to the board dictionary
    std::vector< int > ids;

};



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
 * @brief Create a planar Board object
 *
 * @param width number of markers in X direction
 * @param height number of markers in Y direction
 * @param markerSize marker side length (normally in meters)
 * @param markerSeparation separation between two markers (same unit than markerSize)
 * @param dictionary dictionary of markers indicating the type of markers. The first width*height markers
 * in the dictionary are used.
 * @return the output Board object
 *
 * This functions creates a planar board object given the number of markers in each direction and
 * the marker size and marker separation.
 */
CV_EXPORTS Board createPlanarBoard(int width, int height, float markerSize, float markerSeparation, DICTIONARY dictionary);





/**
 * @brief Draw detected markers in image
 * 
 * @param in input image
 * @param out output image. It will be a copy of in but the markers will be painted on.
 * @param markersCorners positions of marker corners on input image. (e.g std::vector<std::vector<cv::Point2f> > )
 * For N detected markers, the dimensions of this array should be Nx4.
 * The order of the corners should be clockwise.
 * @param ids vector of identifiers for markers in markersCorners . Optional, if not provided, ids are not painted.
 * @param color color of marker borders. Rest of colors (text color and first corner color) are calculated based on this one.
 * 
 * Given an array of detected marker corners and its corresponding ids, this functions draws the markers in the image.
 * The marker borders are painted and the markers identifiers if provided. Useful for debugging purposes.
 */
CV_EXPORTS void drawDetectedMarkers(InputArray in,  OutputArray out, InputArrayOfArrays markersCorners, InputArray ids=noArray(), cv::Scalar borderColor=cv::Scalar(0,255,0));




/**
 * @brief Draw coordinate system axis from pose estimation
 *
 * @param in input image
 * @param out output image. It will be a copy of in but the axis will be painted on.
 * @param cameraMatrix input 3x3 floating-point camera matrix 
 * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
 * @param distCoeffs vector of distortion coefficients
 * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
 * @param rvec rotation vector of the coordinate system that will be drawn. (@see Rodrigues).
 * @param tvec translation vector of the coordinate system that will be drawn. (@see Rodrigues).
 * @param lenght lenght of the painted axis in the same unit than tvec (usually in meters)
 * 
 * Given the pose estimation of a marker or board, this function draws the axis of the world coordinate system
 * i.e. the system centered on the marker/board. Useful for debugging purposes.
 */
CV_EXPORTS void drawAxis(InputArray in, OutputArray out, InputArray cameraMatrix, InputArray distCoeffs, InputArray rvec, InputArray tvec, float lenght);





/**
 * @brief Draw a canonical marker image
 *
 * @param dictionary dictionary of markers indicating the type of markers
 * @param id identifier of the marker that will be returned. It has to be a valid id in the specified dictionary.
 * @param sidePixels size of the image in pixels
 * @param img output image with the marker
 * 
 * This function returns a marker image in its canonical form (i.e. ready to be printed)
 */
CV_EXPORTS void drawMarker(DICTIONARY dictionary, int id, int sidePixels, OutputArray img);





/**
 * @brief Draw a planar board
 *
 * @param board layout of the board that will be drawn. The board should be planar, z coordinate is ignored 
 * @param dictionary dictionary of markers indicating the type of markers
 * @param outSize size of the output image in pixels.
 * @param img output image with the board. The size of this image will be outSize and the board will be
 * on the center, keeping the board proportions.
 * 
 * This function return the image of a planar board, ready to be printed. It assumes the Board layout specified
 * is planar by ignoring the z coordinates of the object points.
 */
CV_EXPORTS void drawPlanarBoard(Board board, cv::Size outSize, OutputArray img);







//CV_EXPORTS void calibrateCamera(InputArrayOfArrays images, Board board, cv::Mat& cameraMatrix, cv::Mat& distCoeffs, OutputArrayOfArrays imgPoints, OutputArrayOfArrays ids, OutputArray tvecs, OutputArray tvecs, int threshParam, int minLenght)




//! @}

}} 


#endif
