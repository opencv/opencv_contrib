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


#ifndef __OPENCV_ARUCO_BOARD_HPP__
#define __OPENCV_ARUCO_BOARD_HPP__

#include <opencv2/core.hpp>
#include <vector>

#include "dictionary.hpp"


namespace cv { namespace aruco {

    
//! @addtogroup aruco
//! @{
    

/**
 * @brief Board of markers
 * 
 * A board is a set of markers in the 3D space with a common cordinate system.
 * The common form of a board of marker is a planar (2D) board, however any 3D layout can be employed.
 * A Board object is composed by:
 * - The object points of the marker corners, i.e. their coordinates respect to the board coordinate system.
 * - The identifier of all the markers in the board.
 */
class CV_EXPORTS Board {

public:
    
    // array of object points of all the marker corners in the board
    // each marker include its 4 corners, i.e. for M markers, the size is Mx4
    std::vector< std::vector<cv::Point3f> > objPoints; 
    
    // vector of the identifiers of the markers in the board (same size than objPoints)
    // The identifiers refers to the board dictionary
    std::vector< int > ids;
    
    /// @TODO dictionary here


    /**
     * @brief Return list of marker points (image and object points) contained in the board
     *
     * @param detectedIds list of identifiers of detected markers (e.g. std::vector<int>)
     * @param detectedImagePoints list of image points of the markers in the detectedIds list
     * (e.g. std::vector< std::vector<Point2f> >) (@see detectMarkers).
     * @param imagePoints output list of image points. This vector contains the same points than 
     * detectedImagePoints excluding those markers that are not included in the board
     * @param objectPoints output list of object points for the same markers in imagePoints.
     * (e.g. std::vector< std::vector<Point3f> > )
     * 
     * Given a list of detected marker identifiers and its image points, this function filters the detected markers
     * and returns the arrays of image points and object points that are contained in the board
     * (i.e.) the points of those markers that are not included in the board are removed.
     */
    void getObjectAndImagePoints(InputArray detectedIds, InputArrayOfArrays detectedImagePoints, OutputArray imagePoints, OutputArray objectPoints);

    
   /**
    * @brief Draw the board
    *
    * @param dictionary dictionary of markers indicating the type of markers
    * @param outSize size of the output image in pixels.
    * @param img output image with the board. The size of this image will be outSize and the board will be
    * on the center, keeping the board proportions.
    * 
    * This function return the image of the planar board, ready to be printed. It assumes the Board layout
    * is planar by ignoring the z coordinates of the object points.
    */
    void drawBoard(Dictionary dictionary, cv::Size outSize, OutputArray img);
    
    
    
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
    static Board createPlanarBoard(int width, int height, float markerSize, float markerSeparation);    
    
    
};



//! @}


}} 


#endif
