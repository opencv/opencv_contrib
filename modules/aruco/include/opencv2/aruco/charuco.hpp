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
#include "../aruco.hpp"


namespace cv {
namespace aruco {

//! @addtogroup aruco
//! @{


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

    // vector of chessboard 3D corners precalculated
    std::vector< cv::Point3f > chessboardCorners;

    // for each charuco corner, nearest marker id and nearest marker corner id of each marker
    std::vector< std::vector<int> > nearestMarkerIds;
    std::vector< std::vector<int> > nearestMarkerCorners;

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
    void draw(cv::Size outSize, OutputArray img, int marginSize = 0, int borderBits = 1);


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
    static CharucoBoard create(int squaresX, int squaresY, double squareLength,
                               double markerLength, DICTIONARY dictionary);

    /**
      *
      */
    cv::Size getChessboardSize() const {
        return cv::Size(_squaresX, _squaresY);
    }

    /**
      *
      */
    double getSquareLength() const {
        return _squareLength;
    }

    /**
      *
      */
    double getMarkerLength() const {
        return _markerLength;
    }

private:

    void _getNearestMarkerCorners();

    // number of markers in X and Y directions
    int _squaresX, _squaresY;

    // size of chessboard squares side (normally in meters)
    double _squareLength;

    // marker side lenght (normally in meters)
    double _markerLength;



};





/**
 * @brief Interpolate position of ChArUco board corners
 * @param markerCorners vector of already detected markers corners. For each marker, its four
 * corners are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the
 * dimensions of this array should be Nx4. The order of the corners should be clockwise.
 * @param markerIds list of identifiers for each marker in corners
 * @param image input image necesary for corner refinement. Note that markers are not detected and
 * should be sent in corners and ids parameters.
 * @param board layout of ChArUco board.
 * @param cameraMatrix input 3x3 floating-point camera matrix
 * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
 * @param distCoeffs vector of distortion coefficients
 * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
 * @param charucoCorners interpolated chessboard corners
 * @param charucoIds interpolated chessboard corners identifiers
 *
 * This function receives the detected markers and returns the 2D position of the chessboard corners
 * from a ChArUco board. The process is based in an approximated pose estimation using
 * the detected Aruco markers.
 * Only those the visible corners are returned. For each corner, its corresponding identifier is
 * also returned in charucoIds.
 * The function returns the number of interpolated corners.
 */
CV_EXPORTS int interpolateCornersCharucoApproxCalib(InputArrayOfArrays markerCorners,
                                                    InputArray markerIds,
                                                    InputArray image, const CharucoBoard &board,
                                                    InputArray cameraMatrix, InputArray distCoeffs,
                                                    OutputArray charucoCorners,
                                                    OutputArray charucoIds );




/**
 * @brief Interpolate position of ChArUco board corners
 * @param markerCorners vector of already detected markers corners. For each marker, its four
 * corners are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the
 * dimensions of this array should be Nx4. The order of the corners should be clockwise.
 * @param markerIds list of identifiers for each marker in corners
 * @param image input image necesary for corner refinement. Note that markers are not detected and
 * should be sent in corners and ids parameters.
 * @param board layout of ChArUco board.
 * @param charucoCorners interpolated chessboard corners
 * @param charucoIds interpolated chessboard corners identifiers
 *
 * This function receives the detected markers and returns the 2D position of the chessboard corners
 * from a ChArUco board. The process is based in an global homography process.
 * Only those the visible corners are returned. For each corner, its corresponding identifier is
 * also returned in charucoIds.
 * The function returns the number of interpolated corners.
 */
CV_EXPORTS int interpolateCornersCharucoGlobalHom(InputArrayOfArrays markerCorners,
                                                  InputArray markerIds,
                                                  InputArray image, const CharucoBoard &board,
                                                  OutputArray charucoCorners,
                                                  OutputArray charucoIds );




/**
 * @brief Interpolate position of ChArUco board corners
 * @param markerCorners vector of already detected markers corners. For each marker, its four
 * corners are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the
 * dimensions of this array should be Nx4. The order of the corners should be clockwise.
 * @param markerIds list of identifiers for each marker in corners
 * @param image input image necesary for corner refinement. Note that markers are not detected and
 * should be sent in corners and ids parameters.
 * @param board layout of ChArUco board.
 * @param charucoCorners interpolated chessboard corners
 * @param charucoIds interpolated chessboard corners identifiers
 *
 * This function receives the detected markers and returns the 2D position of the chessboard corners
 * from a ChArUco board. The process is based in an local homography process (one homography
 * per marker).
 * Only those the visible corners are returned. For each corner, its corresponding identifier is
 * also returned in charucoIds.
 * The function returns the number of interpolated corners.
 */
CV_EXPORTS int interpolateCornersCharucoLocalHom(InputArrayOfArrays markerCorners,
                                                 InputArray markerIds,
                                                 InputArray image, const CharucoBoard &board,
                                                 OutputArray charucoCorners,
                                                 OutputArray charucoIds );





/**
 * @brief Pose estimation for a ChArUco board given some of their corners
 * @param charucoCorners vector of detected charuco corners
 * @param charucoIds list of identifiers for each corner in charucoCorners
 * @param board layout of ChArUco board.
 * @param cameraMatrix input 3x3 floating-point camera matrix
 * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
 * @param distCoeffs vector of distortion coefficients
 * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
 * @param rvec Output vector (e.g. cv::Mat) corresponding to the rotation vector of the board
 * (@sa Rodrigues).
 * @param tvec Output vector (e.g. cv::Mat) corresponding to the translation vector of the board.
 *
 * This function estimates a Charuco board pose from some detected corners.
 * The function checks if the input corners are enough and valid to perform pose estimation.
 * If pose estimation is valid, returns true, else returns false.
 */
CV_EXPORTS bool estimatePoseCharucoBoard(InputArray charucoCorners, InputArray charucoIds,
                                         CharucoBoard &board, InputArray cameraMatrix,
                                         InputArray distCoeffs, OutputArray rvec, OutputArray tvec);




/**
 * @brief Draws a set of Charuco corners
 * @param in input image
 * @param out output image
 * @param charucoCorners vector of detected charuco corners
 * @param charucoIds list of identifiers for each corner in charucoCorners
 * @param cornerColor color of the square surrounding each corner
 *
 * This function draws a set of detected Charuco corners. If identifiers vector is provided, it also
 * draws the id of each corner.
 */
CV_EXPORTS void drawDetectedCornersCharuco(InputArray in, OutputArray out,
                                           InputArray charucoCorners,
                                           InputArray charucoIds = noArray(),
                                           cv::Scalar cornerColor = cv::Scalar(255, 0, 0));



/**
 * @brief Calibrate a camera using Charuco corners
 *
 * @param charucoCorners vector of detected charuco corners per frame
 * @param charucoIds list of identifiers for each corner in charucoCorners per frame
 * @param board Marker Board layout
 * @param imageSize input image size
 * @param cameraMatrix Output 3x3 floating-point camera matrix
 * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ . If CV\_CALIB\_USE\_INTRINSIC\_GUESS
 * and/or CV_CALIB_FIX_ASPECT_RATIO are specified, some or all of fx, fy, cx, cy must be
 * initialized before calling the function.
 * @param distCoeffs Output vector of distortion coefficients
 * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
 * @param rvecs Output vector of rotation vectors (see Rodrigues ) estimated for each board view
 * (e.g. std::vector<cv::Mat>>). That is, each k-th rotation vector together with the corresponding
 * k-th translation vector (see the next output parameter description) brings the board pattern
 * from the model coordinate space (in which object points are specified) to the world coordinate
 * space, that is, a real position of the board pattern in the k-th pattern view (k=0.. *M* -1).
 * @param tvecs Output vector of translation vectors estimated for each pattern view.
 * @param flags flags Different flags  for the calibration process (@sa calibrateCamera)
 * @param criteria Termination criteria for the iterative optimization algorithm.
 *
 * This function calibrates a camera using a set of corners of a  Charuco Board. The function
 * receives a list of detected corners and its identifiers from several views of the Board.
 * The function returns the final re-projection error.
 */
CV_EXPORTS double calibrateCameraCharuco(InputArrayOfArrays charucoCorners,
                                         InputArrayOfArrays charucoIds, const CharucoBoard &board,
                                         cv::Size imageSize, InputOutputArray cameraMatrix,
                                         InputOutputArray distCoeffs,
                                         OutputArrayOfArrays rvecs = noArray(),
                                         OutputArrayOfArrays tvecs = noArray(),
                                         int flags = 0,
                                         TermCriteria criteria = TermCriteria(TermCriteria::COUNT +
                                                                              TermCriteria::EPS,
                                                                              30, DBL_EPSILON));






//! @}


}
}

#endif
