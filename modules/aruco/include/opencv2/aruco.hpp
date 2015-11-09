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
#include "opencv2/aruco/dictionary.hpp"

/**
 * @defgroup aruco ArUco Marker Detection
 * This module is dedicated to square fiducial markers (also known as Augmented Reality Markers)
 * These markers are useful for easy, fast and robust camera pose estimation.ç
 *
 * The main functionalities are:
 * - Detection of markers in a image
 * - Pose estimation from a single marker or from a board/set of markers
 * - Detection of ChArUco board for high subpixel accuracy
 * - Camera calibration from both, ArUco boards and ChArUco boards.
 * - Detection of ChArUco diamond markers
 * The samples directory includes easy examples of how to use the module.
 *
 * The implementation is based on the ArUco Library by R. Muñoz-Salinas and S. Garrido-Jurado.
 *
 * @sa S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas, and M. J. Marín-Jiménez. 2014.
 * "Automatic generation and detection of highly reliable fiducial markers under occlusion".
 * Pattern Recogn. 47, 6 (June 2014), 2280-2292. DOI=10.1016/j.patcog.2014.01.005
 *
 * @sa http://www.uco.es/investiga/grupos/ava/node/26
 *
 * This module has been originally developed by Sergio Garrido-Jurado as a project
 * for Google Summer of Code 2015 (GSoC 15).
 *
 *
*/

namespace cv {
namespace aruco {

//! @addtogroup aruco
//! @{



/**
 * @brief Parameters for the detectMarker process:
 * - adaptiveThreshWinSizeMin: minimum window size for adaptive thresholding before finding
 *   contours (default 3).
 * - adaptiveThreshWinSizeMax: maximum window size for adaptive thresholding before finding
 *   contours (default 23).
 * - adaptiveThreshWinSizeStep: increments from adaptiveThreshWinSizeMin to adaptiveThreshWinSizeMax
 *   during the thresholding (default 10).
 * - adaptiveThreshConstant: constant for adaptive thresholding before finding contours (default 7)
 * - minMarkerPerimeterRate: determine minimum perimeter for marker contour to be detected. This
 *   is defined as a rate respect to the maximum dimension of the input image (default 0.03).
 * - maxMarkerPerimeterRate:  determine maximum perimeter for marker contour to be detected. This
 *   is defined as a rate respect to the maximum dimension of the input image (default 4.0).
 * - polygonalApproxAccuracyRate: minimum accuracy during the polygonal approximation process to
 *   determine which contours are squares.
 * - minCornerDistanceRate: minimum distance between corners for detected markers relative to its
 *   perimeter (default 0.05)
 * - minDistanceToBorder: minimum distance of any corner to the image border for detected markers
 *   (in pixels) (default 3)
 * - minMarkerDistanceRate: minimum mean distance beetween two marker corners to be considered
 *   similar, so that the smaller one is removed. The rate is relative to the smaller perimeter
 *   of the two markers (default 0.05).
 * - doCornerRefinement: do subpixel refinement or not
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
 *   number of bits per marker (default 0.35).
 * - minOtsuStdDev: minimun standard deviation in pixels values during the decodification step to
 *   apply Otsu thresholding (otherwise, all the bits are set to 0 or 1 depending on mean higher
 *   than 128 or not) (default 5.0)
 * - errorCorrectionRate error correction rate respect to the maximun error correction capability
 *   for each dictionary. (default 0.6).
 */
struct CV_EXPORTS DetectorParameters {

    DetectorParameters();

    int adaptiveThreshWinSizeMin;
    int adaptiveThreshWinSizeMax;
    int adaptiveThreshWinSizeStep;
    double adaptiveThreshConstant;
    double minMarkerPerimeterRate;
    double maxMarkerPerimeterRate;
    double polygonalApproxAccuracyRate;
    double minCornerDistanceRate;
    int minDistanceToBorder;
    double minMarkerDistanceRate;
    bool doCornerRefinement;
    int cornerRefinementWinSize;
    int cornerRefinementMaxIterations;
    double cornerRefinementMinAccuracy;
    int markerBorderBits;
    int perspectiveRemovePixelPerCell;
    double perspectiveRemoveIgnoredMarginPerCell;
    double maxErroneousBitsInBorderRate;
    double minOtsuStdDev;
    double errorCorrectionRate;
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
CV_EXPORTS void detectMarkers(InputArray image, Dictionary dictionary, OutputArrayOfArrays corners,
                              OutputArray ids, DetectorParameters parameters = DetectorParameters(),
                              OutputArrayOfArrays rejectedImgPoints = noArray());



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
 * @param rvecs array of output rotation vectors (@sa Rodrigues) (e.g. std::vector<cv::Vec3d>>).
 * Each element in rvecs corresponds to the specific marker in imgPoints.
 * @param tvecs array of output translation vectors (e.g. std::vector<cv::Vec3d>>).
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
CV_EXPORTS void estimatePoseSingleMarkers(InputArrayOfArrays corners, float markerLength,
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
    std::vector< std::vector< Point3f > > objPoints;

    // the dictionary of markers employed for this board
    Dictionary dictionary;

    // vector of the identifiers of the markers in the board (same size than objPoints)
    // The identifiers refers to the board dictionary
    std::vector< int > ids;
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
    void draw(Size outSize, OutputArray img, int marginSize = 0, int borderBits = 1);


    /**
     * @brief Create a GridBoard object
     *
     * @param markersX number of markers in X direction
     * @param markersY number of markers in Y direction
     * @param markerLength marker side length (normally in meters)
     * @param markerSeparation separation between two markers (same unit than markerLenght)
     * @param dictionary dictionary of markers indicating the type of markers.
     * The first markersX*markersY markers in the dictionary are used.
     * @return the output GridBoard object
     *
     * This functions creates a GridBoard object given the number of markers in each direction and
     * the marker size and marker separation.
     */
    static GridBoard create(int markersX, int markersY, float markerLength, float markerSeparation,
                            Dictionary dictionary);

    /**
      *
      */
    Size getGridSize() const { return Size(_markersX, _markersY); }

    /**
      *
      */
    float getMarkerLength() const { return _markerLength; }

    /**
      *
      */
    float getMarkerSeparation() const { return _markerSeparation; }


    private:
    // number of markers in X and Y directions
    int _markersX, _markersY;

    // marker side lenght (normally in meters)
    float _markerLength;

    // separation between markers in the grid
    float _markerSeparation;
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
 * @brief Refind not detected markers based on the already detected and the board layout
 *
 * @param image input image
 * @param board layout of markers in the board.
 * @param detectedCorners vector of already detected marker corners.
 * @param detectedIds vector of already detected marker identifiers.
 * @param rejectedCorners vector of rejected candidates during the marker detection process.
 * @param cameraMatrix optional input 3x3 floating-point camera matrix
 * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
 * @param distCoeffs optional vector of distortion coefficients
 * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
 * @param minRepDistance minimum distance between the corners of the rejected candidate and the
 * reprojected marker in order to consider it as a correspondence.
 * @param errorCorrectionRate rate of allowed erroneous bits respect to the error correction
 * capability of the used dictionary. -1 ignores the error correction step.
 * @param checkAllOrders Consider the four posible corner orders in the rejectedCorners array.
 * If it set to false, only the provided corner order is considered (default true).
 * @param recoveredIdxs Optional array to returns the indexes of the recovered candidates in the
 * original rejectedCorners array.
 * @param parameters marker detection parameters
 *
 * This function tries to find markers that were not detected in the basic detecMarkers function.
 * First, based on the current detected marker and the board layout, the function interpolates
 * the position of the missing markers. Then it tries to find correspondence between the reprojected
 * markers and the rejected candidates based on the minRepDistance and errorCorrectionRate
 * parameters.
 * If camera parameters and distortion coefficients are provided, missing markers are reprojected
 * using projectPoint function. If not, missing marker projections are interpolated using global
 * homography, and all the marker corners in the board must have the same Z coordinate.
 */
CV_EXPORTS void refineDetectedMarkers(
    InputArray image, const Board &board, InputOutputArrayOfArrays detectedCorners,
    InputOutputArray detectedIds, InputOutputArray rejectedCorners,
    InputArray cameraMatrix = noArray(), InputArray distCoeffs = noArray(),
    float minRepDistance = 10.f, float errorCorrectionRate = 3.f, bool checkAllOrders = true,
    OutputArray recoveredIdxs = noArray(), DetectorParameters parameters = DetectorParameters());



/**
 * @brief Draw detected markers in image
 *
 * @param image input/output image. It must have 1 or 3 channels. The number of channels is not
 * altered.
 * @param corners positions of marker corners on input image.
 * (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the dimensions of
 * this array should be Nx4. The order of the corners should be clockwise.
 * @param ids vector of identifiers for markers in markersCorners .
 * Optional, if not provided, ids are not painted.
 * @param borderColor color of marker borders. Rest of colors (text color and first corner color)
 * are calculated based on this one to improve visualization.
 *
 * Given an array of detected marker corners and its corresponding ids, this functions draws
 * the markers in the image. The marker borders are painted and the markers identifiers if provided.
 * Useful for debugging purposes.
 */
CV_EXPORTS void drawDetectedMarkers(InputOutputArray image, InputArrayOfArrays corners,
                                    InputArray ids = noArray(),
                                    Scalar borderColor = Scalar(0, 255, 0));



/**
 * @brief Draw coordinate system axis from pose estimation
 *
 * @param image input/output image. It must have 1 or 3 channels. The number of channels is not
 * altered.
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
CV_EXPORTS void drawAxis(InputOutputArray image, InputArray cameraMatrix, InputArray distCoeffs,
                         InputArray rvec, InputArray tvec, float length);



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
CV_EXPORTS void drawMarker(Dictionary dictionary, int id, int sidePixels, OutputArray img,
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
CV_EXPORTS void drawPlanarBoard(const Board &board, Size outSize, OutputArray img,
                                int marginSize = 0, int borderBits = 1);



/**
 * @brief Calibrate a camera using aruco markers
 *
 * @param corners vector of detected marker corners in all frames.
 * The corners should have the same format returned by detectMarkers (@sa detectMarkers).
 * @param ids list of identifiers for each marker in corners
 * @param counter number of markers in each frame so that corners and ids can be split
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
CV_EXPORTS double calibrateCameraAruco(
    InputArrayOfArrays corners, InputArray ids, InputArray counter, const Board &board,
    Size imageSize, InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
    OutputArrayOfArrays rvecs = noArray(), OutputArrayOfArrays tvecs = noArray(), int flags = 0,
    TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON));



//! @}
}
}

#endif
