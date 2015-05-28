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


namespace cv { namespace aruco {


/**
 * @brief Board of markers containing the layout of the markers
 * 
 */
class CV_EXPORTS Board {

public:
    
    std::vector< std::vector<cv::Point3f> > objPoints; // each marker include its 4 corners, i.e. for M marker, Mx4
    std::vector< int > ids; // ids of each marker in the board (same size than objPoints)

    
    /**
     * @brief Draw board image considering (x,y) coordinates (ignoring Z)
     * 
     * @param image 
     * @return void
     */
    void drawBoard(InputOutputArray image);

    

    /**
     * @brief get list of object points for the detected markers
     *
     * @param detectedIds
     * @param objectPoints
     * @return void
     */
    void getObjectAndImagePoints(InputArray detectedIds, InputArrayOfArrays detectedImagePoints, OutputArray imagePoints, OutputArray objectPoints);

    
    /**
     * @brief Fast creation of a planar Board object
     * 
     * @param width number of markers in X
     * @param height number of markers in Y
     * @param markerSize marker side distance (in meters)
     * @param markerSeparation consecutive marker separation (in meters)
     * @param dictionary
     * @return cv::aruco::Board
     */
    static Board createPlanarBoard(int width, int height, float markerSize, float markerSeparation);
    
};




/**
 * @brief Detect single markers of the specific dictionary in an image
 *
 * @param image input image
 * @param dictionary markers are identified based on this dictionary
 * @param imgPoints returns detected markers corner positions
 * @param ids returns detected markers ids
 * @param threshParam window size for adaptative thresholding
 * @param minLenght minimum size of candidates contour lenght. It is indicated as a ratio
 *                  respect to the largest image dimension
 * @return void
 */
CV_EXPORTS void detectMarkers(InputArray image, Dictionary dictionary, OutputArrayOfArrays imgPoints,
                       OutputArray ids, int threshParam=21,float minLenght=0.03);
CV_EXPORTS void detectMarkers(InputArray image, PREDEFINED_DICTIONARIES dictionary, OutputArrayOfArrays imgPoints,
                       OutputArray ids, int threshParam=21,float minLenght=0.03);





/**
 * @brief Estimate single poses of list of markers
 *
 * @param imgPoints List of markers corners
 * @param markersize size of marker side in meters
 * @param cameraMatrix
 * @param distCoeffs
 * @param rvecs returns rvec for each marker
 * @param tvecs same as Tvec
 * @return void
 */
CV_EXPORTS void estimatePoseSingleMarkers(InputArrayOfArrays imgPoints, float markersize, InputArray cameraMatrix,
                                          InputArray distCoeffs, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs);


/**
 * @brief Estimate pose of marker board
 *
 * @param imgPoints List of markers corners
 * @param board board layout
 * @param cameraMatrix
 * @param distCoeffs
 * @param rvec board rotation vector
 * @param tvec board translation vector
 * @return void
 */
CV_EXPORTS void estimatePoseBoard(InputArrayOfArrays imgPoints, InputArray ids, Board board,
                                          InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec, OutputArray tvec);




/**
 * @brief Draw detected markers in image
 * 
 * @param image 
 * @param markers maker corner positions
 * @param ids marker ids
 * @return void
 */
CV_EXPORTS void drawDetectedMarkers(InputOutputArray image, InputArrayOfArrays markers, InputArray ids=noArray(), bool drawId=true);


/**
 * @brief Draw colored axis in image
 *
 * @param image
 * @param cameraMatrix
 * @param distCoeffs
 * @param rvec
 * @param tvec
 * @param lenght axis lenght in meters
 * @return void
 */
CV_EXPORTS void drawAxis(InputOutputArray image, InputArray cameraMatrix, InputArray distCoeffs, InputArray rvec, InputArray tvec, float lenght);




//CV_EXPORTS void calibrateCamera(InputArrayOfArrays images, Board board, cv::Mat& cameraMatrix, cv::Mat& distCoeffs, OutputArrayOfArrays imgPoints, OutputArrayOfArrays ids, OutputArray tvecs, OutputArray tvecs, int threshParam, int minLenght)



}} 


#endif
