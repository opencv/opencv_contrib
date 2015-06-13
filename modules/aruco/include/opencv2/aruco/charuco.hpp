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

#ifndef __OPENCV_CHARUCO_HPP__
#define __OPENCV_CHARUCO_HPP__

#include <opencv2/core.hpp>
#include <vector>
#include "aruco.hpp"


namespace cv {
namespace aruco {

//! @addtogroup aruco
//! @{


/**
 * @defgroup charuco ChArUco detection based on ArUco markers and chessboards.
 * This module is dedicated to square fiducial marker (also known as Augmented Reality Markers)
 * The ChArUco board combines the versatility of the ArUco markers with the high corner precision
 * of chessboards.
 * The two main ChArUco tools are:
 * - ChArUco boards for versatil calibration with high precision.
 * - ChArUco markers for accurate pose estimation.
*/


/**
 * @brief ChArUco board
 * Specific class for ChArUco boards. A ChArUco board is a planar board where the markers are placed
 * inside the white squares of a chessboard. The benefits of ChArUco boards is that they provide
 * both, ArUco markers versatility and chessboard corner precision, which is important for
 * calibration and pose estimation.
 * This class also allows the easy creation and drawing of ChArUco boards.
 */
class CV_EXPORTS CharucoBoard : public Board {

public:


    /**
     * @brief Draw a ChArUco board
     *
     * @param outSize size of the output image in pixels.
     * @param img output image with the board. The size of this image will be outSize
     * and the board will be on the center, keeping the board proportions.
     * @param marginSize minimum margins (in pixels) of the board in the output image
     * @param borderBits width of the marker borders.
     *
     * This function return the image of the ChArUco board, ready to be printed.
     */
    CV_EXPORTS void draw(cv::Size outSize, OutputArray img, int marginSize = 0, int borderBits = 1);


    /**
     * @brief Create a CharucoBoard object
     *
     * @param squaresX number of chessboard squares in X direction
     * @param squaresY number of chessboard squares in Y direction
     * @param squareLength chessboard square side length (normally in meters)
     * @param markerLenght marker side length (same unit than squareLength)
     * @param dictionary dictionary of markers indicating the type of markers.
     * The first markers in the dictionary are used to fill the white chessboard squares.
     * @return the output CharucoBoard object
     *
     * This functions creates a CharucoBoard object given the number of squares in each direction
     * and the size of the markers and chessboard squares.
     */
    CV_EXPORTS static CharucoBoard create(int squaresX, int squaresY, double squareLength,
                                          double markerLength, DICTIONARY dictionary);

    /**
      *
      */
    cv::Size getChessboardSize() {
        return cv::Size(_squaresX, _squaresY);
    }

    /**
      *
      */
    int getSquareLength() {
        return _squareLength;
    }

    /**
      *
      */
    double getMarkerLength() {
        return _markerLength;
    }

private:

    // number of markers in X and Y directions
    int _squaresX, _squaresY;

    // size of chessboard squares side (normally in meters)
    double _squareLength;

    // marker side lenght (normally in meters)
    double _markerLength;

};



/**
 * @brief Pose estimation for a board of markers
 *
 * @param corners vector of already detected markers corners. For each marker, its four corners
 * are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the
 * dimensions of this array should be Nx4. The order of the corners should be clockwise.
 * @param ids list of identifiers for each marker in corners
 * @param board layout of ChArUco board.
 * @param cameraMatrix input 3x3 floating-point camera matrix
 * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
 * @param distCoeffs vector of distortion coefficients
 * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
 * @param rvec Output vector (e.g. cv::Mat) corresponding to the rotation vector of the board
 * (@sa Rodrigues).
 * @param tvec Output vector (e.g. cv::Mat) corresponding to the translation vector of the board.
 *
 * This function receives the detected markers and returns the pose of a ChArUco board composed
 * by those markers.
 * A ChArUco board, as any marker board, has a single world coordinate system which is defined by
 * the board layout. The returned transformation is the one that transforms points from the board
 * coordinate system to the camera coordinate system.
 * Input markers that are not included in the board layout are ignored.
 */
/// @TODO
//CV_EXPORTS void estimatePoseCharucoBoard(InputArrayOfArrays corners, InputArray ids,
//                                         const CharucoBoard &board, InputArray cameraMatrix,
//                                         InputArray distCoeffs, OutputArray rvec, OutputArray tvec,
//                                         OutputArray chessboardCorners = noArray());



//! @}


}
}

#endif
