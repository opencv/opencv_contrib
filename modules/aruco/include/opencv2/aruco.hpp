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
 * This module is dedicated to square fiducial marker (also known as Augmented Reality Markers)
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
 * - charuco_calibration_online.cpp - online, automatic calibration using charuco boards
*/

namespace cv {
namespace aruco {

//! @addtogroup aruco
//! @{



/**
 * @brief Predefined markers dictionaries/sets
 * - DICT_ARUCO: standard ArUco Library Markers. 1024 markers, 5x5 bits, 0 minimum distance
 */
enum DICTIONARY { DICT_ARUCO = 0 };



/**
 * @brief Parameters for the detectMarker process:
 * - adaptiveThreshWinSize: window size for adaptive thresholding before finding contours.
 *   (default 21)
 * - adaptiveThreshConstant: constant for adaptive thresholding before finding contours (default 7)
 * - minMarkerPerimeterRate: determine minimum perimeter for marker contour to be detected. This
 *   is defined as a rate respect to the maximum dimension of the input image (default 0.03).
 * - maxMarkerPerimeterRate:  determine maximum perimeter for marker contour to be detected. This
 *   is defined as a rate respect to the maximum dimension of the input image (default 4.0).
 * - polygonalApproxAccuracyRate: minimum accuracy during the polygonal approximation process to
 *   determine which contours are squares.
 * - minCornerDistance: minimum distance between corners for detected markers (in pixels)
 *   (default 10)
 * - minDistanceToBorder: minimum distance of any corner to the image border for detected markers
 *   (in pixels) (default 3)
 * - minMarkerDistance: minimum mean distance beetween two marker corners to be considered
 *   similar, so that the smaller one is removed (in pixels) (default 10).
 * - cornerRefinementWinSize: window size for the corner refinement process (in pixels) (default 5).
 * - cornerRefinementMaxIterations: maximum number of iterations for stop criteria of the corner
 *   refinement process (default 30).
 * - cornerRefinementMinAccuracy: minimum error for the stop cristeria of the corner refinement
 *   process (default: 0.1)
 * - markerBorderBits: number of bits of the marker border, i.e. marker border width (default 1).
 * - perpectiveRemovePixelPerCell: number of bits (per dimension) for each cell of the marker
 *   when removing the perspective (default 8).
 * - perspectiveRemoveIgnoredMarginPerCell: width of the margin of pixels on each cell not
 *   considered for the determination of the cell bit. Represents the rate respect to the total
 *   size of the cell, i.e. perpectiveRemovePixelPerCell (default 0.13)
 * - maxErroneousBitsInBorderRate: maximum number of accepted erroneous bits in the border (i.e.
 *   number of allowed white bits in the border). Represented as a rate respect to the total
 *   number of bits per marker (default 0.04).
 */
struct DetectorParameters {

    CV_EXPORTS DetectorParameters();

    int adaptiveThreshWinSize;
    double adaptiveThreshConstant;
    double minMarkerPerimeterRate;
    double maxMarkerPerimeterRate;
    double polygonalApproxAccuracyRate;
    double minCornerDistance;
    int minDistanceToBorder;
    double minMarkerDistance;
    int cornerRefinementWinSize;
    int cornerRefinementMaxIterations;
    double cornerRefinementMinAccuracy;
    int markerBorderBits;
    int perspectiveRemovePixelPerCell;
    double perspectiveRemoveIgnoredMarginPerCell;
    double maxErroneousBitsInBorderRate;
};



/**
 * @brief Basic marker detection
 *
 * @param image input image
 * @param dictionary indicates the type of markers that will be searched
 * @param corners vector of detected marker corners. For each marker, its four corners
 * are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers,
 * the dimensions of this array is Nx4. The order of the corners is clockwise.
 * @param ids vector of identifiers of the detected markers. The identifier is of type int
 * (e.g. std::vector<int>). For N detected markers, the size of ids is also N.
 * The identifiers have the same order than the markers in the imgPoints array.
 * @param parameters marker detection parameters
 * @param rejectedImgPoints contains the imgPoints of those squares whose inner code has not a
 * correct codification. Useful for debugging purposes.
 *
 * Performs marker detection in the input image. Only markers included in the specific dictionary
 * are searched. For each detected marker, it returns the 2D position of its corner in the image
 * and its corresponding identifier.
 * Note that this function does not perform pose estimation.
 * @sa estimatePoseSingleMarkers,  estimatePoseBoard
 *
 */
CV_EXPORTS void detectMarkers(InputArray image, DICTIONARY dictionary,
                              OutputArrayOfArrays corners, OutputArray ids,
                              DetectorParameters parameters=DetectorParameters(),
                              OutputArrayOfArrays rejectedImgPoints = cv::noArray());



/**
 * @brief Pose estimation for single markers
 *
 * @param corners vector of already detected markers corners. For each marker, its four corners
 * are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers,
 * the dimensions of this array should be Nx4. The order of the corners should be clockwise.
 * @sa detectMarkers
 * @param markerLength the length of the markers' side. The returning translation vectors will
 * be in the same unit. Normally, unit is meters.
 * @param cameraMatrix input 3x3 floating-point camera matrix
 * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
 * @param distCoeffs vector of distortion coefficients
 * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
 * @param rvecs array of output rotation vectors (@sa Rodrigues) (e.g. std::vector<cv::Mat>>).
 * Each element in rvecs corresponds to the specific marker in imgPoints.
 * @param tvecs array of output translation vectors (e.g. std::vector<cv::Mat>>).
 * Each element in tvecs corresponds to the specific marker in imgPoints.
 *
 * This function receives the detected markers and returns their pose estimation respect to
 * the camera individually. So for each marker, one rotation and translation vector is returned.
 * The returned transformation is the one that transforms points from each marker coordinate system
 * to the camera coordinate system.
 * The marker corrdinate system is centered on the middle of the marker, with the Z axis
 * perpendicular to the marker plane.
 * The coordinates of the four corners of the marker in its own coordinate system are:
 * (-markerLength/2, markerLength/2, 0), (markerLength/2, markerLength/2, 0),
 * (markerLength/2, -markerLength/2, 0), (-markerLength/2, -markerLength/2, 0)
 */
CV_EXPORTS void estimatePoseSingleMarkers(InputArrayOfArrays corners, double markerLength,
                                          InputArray cameraMatrix, InputArray distCoeffs,
                                          OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs);



/**
 * @brief Board of markers
 *
 * A board is a set of markers in the 3D space with a common cordinate system.
 * The common form of a board of marker is a planar (2D) board, however any 3D layout can be used.
 * A Board object is composed by:
 * - The object points of the marker corners, i.e. their coordinates respect to the board system.
 * - The dictionary which indicates the type of markers of the board
 * - The identifier of all the markers in the board.
 */
class CV_EXPORTS Board {

public:

    // array of object points of all the marker corners in the board
    // each marker include its 4 corners, i.e. for M markers, the size is Mx4
    std::vector<std::vector<cv::Point3f> > objPoints;

    // the dictionary of markers employed for this board
    DICTIONARY dictionary;

    // vector of the identifiers of the markers in the board (same size than objPoints)
    // The identifiers refers to the board dictionary
    std::vector<int> ids;
};



/**
 * @brief Planar board with grid arrangement of markers
 * More common type of board. All markers are placed in the same plane in a grid arrangment.
 * The board can be drawn using drawPlanarBoard() function (@sa drawPlanarBoard)
 */
class CV_EXPORTS GridBoard : public Board {

public:


    /**
     * @brief Draw a GridBoard
     *
     * @param outSize size of the output image in pixels.
     * @param img output image with the board. The size of this image will be outSize
     * and the board will be on the center, keeping the board proportions.
     * @param marginSize minimum margins (in pixels) of the board in the output image
     * @param borderBits width of the marker borders.
     *
     * This function return the image of the GridBoard, ready to be printed.
     */
    CV_EXPORTS void draw(cv::Size outSize, OutputArray img, int marginSize = 0, int borderBits = 1);


    /**
     * @brief Create a GridBoard object
     *
     * @param markersX number of markers in X direction
     * @param markersY number of markers in Y direction
     * @param markerLenght marker side length (normally in meters)
     * @param markerSeparation separation between two markers (same unit than markerLenght)
     * @param dictionary dictionary of markers indicating the type of markers.
     * The first markersX*markersY markers in the dictionary are used.
     * @return the output GridBoard object
     *
     * This functions creates a GridBoard object given the number of markers in each direction and
     * the marker size and marker separation.
     */
    CV_EXPORTS static GridBoard create(int markersX, int markersY, double markerLength,
                                       double markerSeparation, DICTIONARY dictionary);

    /**
      *
      */
    cv::Size getGridSize() {
        return cv::Size(_markersX, _markersY);
    }

    /**
      *
      */
    double getMarkerLength() {
        return _markerLength;
    }

    /**
      *
      */
    int getMarkerSeparation() {
        return _markerSeparation;
    }


private:
    // number of markers in X and Y directions
    int _markersX, _markersY;

    // marker side lenght (normally in meters)
    double _markerLength;

    // separation between markers in the grid
    double _markerSeparation;

};



/**
 * @brief Pose estimation for a board of markers
 *
 * @param corners vector of already detected markers corners. For each marker, its four corners
 * are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the
 * dimensions of this array should be Nx4. The order of the corners should be clockwise.
 * @param ids list of identifiers for each marker in corners
 * @param board layout of markers in the board. The layout is composed by the marker identifiers
 * and the positions of each marker corner in the board reference system.
 * @param cameraMatrix input 3x3 floating-point camera matrix
 * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
 * @param distCoeffs vector of distortion coefficients
 * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
 * @param rvec Output vector (e.g. cv::Mat) corresponding to the rotation vector of the board
 * (@sa Rodrigues).
 * @param tvec Output vector (e.g. cv::Mat) corresponding to the translation vector of the board.
 *
 * This function receives the detected markers and returns the pose of a marker board composed
 * by those markers.
 * A Board of marker has a single world coordinate system which is defined by the board layout.
 * The returned transformation is the one that transforms points from the board coordinate system
 * to the camera coordinate system.
 * Input markers that are not included in the board layout are ignored.
 * The function returns the number of markers from the input employed for the board pose estimation.
 * Note that returning a 0 means the pose has not been estimated.
 */
CV_EXPORTS int estimatePoseBoard(InputArrayOfArrays corners, InputArray ids, const Board &board,
                                 InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec,
                                 OutputArray tvec);



/**
 * @brief Draw detected markers in image
 *
 * @param in input image
 * @param out output image. It will be a copy of in but the markers will be painted on.
 * @param corners positions of marker corners on input image.
 * (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the dimensions of
 * this array should be Nx4. The order of the corners should be clockwise.
 * @param ids vector of identifiers for markers in markersCorners .
 * Optional, if not provided, ids are not painted.
 * @param color color of marker borders. Rest of colors (text color and first corner color)
 * are calculated based on this one.
 *
 * Given an array of detected marker corners and its corresponding ids, this functions draws
 * the markers in the image. The marker borders are painted and the markers identifiers if provided.
 * Useful for debugging purposes.
 */
CV_EXPORTS void drawDetectedMarkers(InputArray in, OutputArray out,
                                    InputArrayOfArrays corners, InputArray ids = noArray(),
                                    cv::Scalar borderColor = cv::Scalar(0, 255, 0));



/**
 * @brief Draw coordinate system axis from pose estimation
 *
 * @param in input image
 * @param out output image. It will be a copy of in but the axis will be painted on.
 * @param cameraMatrix input 3x3 floating-point camera matrix
 * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
 * @param distCoeffs vector of distortion coefficients
 * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
 * @param rvec rotation vector of the coordinate system that will be drawn. (@sa Rodrigues).
 * @param tvec translation vector of the coordinate system that will be drawn.
 * @param length length of the painted axis in the same unit than tvec (usually in meters)
 *
 * Given the pose estimation of a marker or board, this function draws the axis of the world
 * coordinate system, i.e. the system centered on the marker/board. Useful for debugging purposes.
 */
CV_EXPORTS void drawAxis(InputArray in, OutputArray out, InputArray cameraMatrix,
                         InputArray distCoeffs, InputArray rvec, InputArray tvec, double length);



/**
 * @brief Draw a canonical marker image
 *
 * @param dictionary dictionary of markers indicating the type of markers
 * @param id identifier of the marker that will be returned. It has to be a valid id
 * in the specified dictionary.
 * @param sidePixels size of the image in pixels
 * @param img output image with the marker
 * @param borderBits width of the marker border.
 *
 * This function returns a marker image in its canonical form (i.e. ready to be printed)
 */
CV_EXPORTS void drawMarker(DICTIONARY dictionary, int id, int sidePixels, OutputArray img,
                           int borderBits = 1);



/**
 * @brief Draw a planar board
 *
 * @param board layout of the board that will be drawn. The board should be planar,
 * z coordinate is ignored
 * @param outSize size of the output image in pixels.
 * @param img output image with the board. The size of this image will be outSize
 * and the board will be on the center, keeping the board proportions.
 * @param marginSize minimum margins (in pixels) of the board in the output image
 * @param borderBits width of the marker borders.
 *
 * This function return the image of a planar board, ready to be printed. It assumes
 * the Board layout specified is planar by ignoring the z coordinates of the object points.
 */
CV_EXPORTS void drawPlanarBoard(const Board &board, cv::Size outSize, OutputArray img,
                                int marginSize = 0, int borderBits = 1);



/**
 * @brief Calibrate a camera using aruco markers
 *
 * @param corners vector of detected marker corners in each frame.
 * The corners should have the same format returned by detectMarkers (@sa detectMarkers).
 * @param ids list of identifiers for each marker in corners
 * @param board Marker Board layout
 * @param imageSize Size of the image used only to initialize the intrinsic camera matrix.
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
 * This function calibrates a camera using an Aruco Board. The function receives a list of
 * detected markers from several views of the Board. The process is similar to the chessboard
 * calibration in calibrateCamera(). The function returns the final re-projection error.
 */
CV_EXPORTS double calibrateCameraAruco(const
                                       std::vector<std::vector<std::vector<Point2f> > > &corners,
                                       const std::vector<std::vector<int> > & ids,
                                       const Board &board, Size imageSize,
                                       InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                       OutputArrayOfArrays rvecs = noArray(),
                                       OutputArrayOfArrays tvecs = noArray(), int flags = 0,
                                       TermCriteria criteria = TermCriteria(TermCriteria::COUNT +
                                                                            TermCriteria::EPS,
                                                                            30, DBL_EPSILON));



//! @}


}
}

#endif
