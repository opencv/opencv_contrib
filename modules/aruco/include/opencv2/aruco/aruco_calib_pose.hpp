// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef __OPENCV_ARUCO_CALIB_POSE_HPP__
#define __OPENCV_ARUCO_CALIB_POSE_HPP__
#include <opencv2/aruco/board.hpp>
#include <opencv2/calib3d.hpp>

namespace cv {
namespace aruco {

//! @addtogroup aruco
//! @{

/** @brief
 * rvec/tvec define the right handed coordinate system of the marker.
 * PatternPos defines center this system and axes direction.
 * Axis X (red color) - first coordinate, axis Y (green color) - second coordinate,
 * axis Z (blue color) - third coordinate.
 * @sa estimatePoseSingleMarkers(), @ref tutorial_aruco_detection
 */
enum PatternPos {
    /** @brief The marker coordinate system is centered on the middle of the marker.
        * The coordinates of the four corners (CCW order) of the marker in its own coordinate system are:
        * (-markerLength/2, markerLength/2, 0), (markerLength/2, markerLength/2, 0),
        * (markerLength/2, -markerLength/2, 0), (-markerLength/2, -markerLength/2, 0).
        *
        * These pattern points define this coordinate system:
        * ![Image with axes drawn](images/singlemarkersaxes.jpg)
        */
    CCW_center,
    /** @brief The marker coordinate system is centered on the top-left corner of the marker.
        * The coordinates of the four corners (CW order) of the marker in its own coordinate system are:
        * (0, 0, 0), (markerLength, 0, 0),
        * (markerLength, markerLength, 0), (0, markerLength, 0).
        *
        * These pattern points define this coordinate system:
        * ![Image with axes drawn](images/singlemarkersaxes2.jpg)
        *
        * These pattern dots are convenient to use with a chessboard/ChArUco board.
        */
    CW_top_left_corner
};

/** @brief
 * Pose estimation parameters
 * @param pattern Defines center this system and axes direction (default PatternPos::CCW_center).
 * @param useExtrinsicGuess Parameter used for SOLVEPNP_ITERATIVE. If true (1), the function uses the provided
 * rvec and tvec values as initial approximations of the rotation and translation vectors, respectively, and further
 * optimizes them (default false).
 * @param solvePnPMethod Method for solving a PnP problem: see @ref calib3d_solvePnP_flags (default SOLVEPNP_ITERATIVE).
 * @sa PatternPos, solvePnP(), @ref tutorial_aruco_detection
 */
struct CV_EXPORTS_W EstimateParameters {
    CV_PROP_RW PatternPos pattern;
    CV_PROP_RW bool useExtrinsicGuess;
    CV_PROP_RW SolvePnPMethod solvePnPMethod;

    EstimateParameters(): pattern(CCW_center), useExtrinsicGuess(false),
                          solvePnPMethod(SOLVEPNP_ITERATIVE) {}

    CV_WRAP static Ptr<EstimateParameters> create() {
        return makePtr<EstimateParameters>();
    }
};


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
 * @param rvecs array of output rotation vectors (@sa Rodrigues) (e.g. std::vector<cv::Vec3d>).
 * Each element in rvecs corresponds to the specific marker in imgPoints.
 * @param tvecs array of output translation vectors (e.g. std::vector<cv::Vec3d>).
 * Each element in tvecs corresponds to the specific marker in imgPoints.
 * @param _objPoints array of object points of all the marker corners
 * @param estimateParameters set the origin of coordinate system and the coordinates of the four corners of the marker
 * (default estimateParameters.pattern = PatternPos::CCW_center, estimateParameters.useExtrinsicGuess = false,
 * estimateParameters.solvePnPMethod = SOLVEPNP_ITERATIVE).
 *
 * This function receives the detected markers and returns their pose estimation respect to
 * the camera individually. So for each marker, one rotation and translation vector is returned.
 * The returned transformation is the one that transforms points from each marker coordinate system
 * to the camera coordinate system.
 * The marker coordinate system is centered on the middle (by default) or on the top-left corner of the marker,
 * with the Z axis perpendicular to the marker plane.
 * estimateParameters defines the coordinates of the four corners of the marker in its own coordinate system (by default) are:
 * (-markerLength/2, markerLength/2, 0), (markerLength/2, markerLength/2, 0),
 * (markerLength/2, -markerLength/2, 0), (-markerLength/2, -markerLength/2, 0)
 * @sa use cv::drawFrameAxes to get world coordinate system axis for object points
 * @sa @ref tutorial_aruco_detection
 * @sa EstimateParameters
 * @sa PatternPos
 */
CV_EXPORTS_W void estimatePoseSingleMarkers(InputArrayOfArrays corners, float markerLength,
                                            InputArray cameraMatrix, InputArray distCoeffs,
                                            OutputArray rvecs, OutputArray tvecs, OutputArray _objPoints = noArray(),
                                            Ptr<EstimateParameters> estimateParameters = EstimateParameters::create());

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
 * (see cv::Rodrigues). Used as initial guess if not empty.
 * @param tvec Output vector (e.g. cv::Mat) corresponding to the translation vector of the board.
 * @param useExtrinsicGuess defines whether initial guess for \b rvec and \b tvec will be used or not.
 * Used as initial guess if not empty.
 *
 * This function receives the detected markers and returns the pose of a marker board composed
 * by those markers.
 * A Board of marker has a single world coordinate system which is defined by the board layout.
 * The returned transformation is the one that transforms points from the board coordinate system
 * to the camera coordinate system.
 * Input markers that are not included in the board layout are ignored.
 * The function returns the number of markers from the input employed for the board pose estimation.
 * Note that returning a 0 means the pose has not been estimated.
 * @sa use cv::drawFrameAxes to get world coordinate system axis for object points
 */
CV_EXPORTS_W int estimatePoseBoard(InputArrayOfArrays corners, InputArray ids, const Ptr<Board> &board,
                                   InputArray cameraMatrix, InputArray distCoeffs, InputOutputArray rvec,
                                   InputOutputArray tvec, bool useExtrinsicGuess = false);

/**
 * @brief Given a board configuration and a set of detected markers, returns the corresponding
 * image points and object points to call solvePnP
 *
 * @param board Marker board layout.
 * @param detectedCorners List of detected marker corners of the board.
 * @param detectedIds List of identifiers for each marker.
 * @param objPoints Vector of vectors of board marker points in the board coordinate space.
 * @param imgPoints Vector of vectors of the projections of board marker corner points.
*/
CV_EXPORTS_W void getBoardObjectAndImagePoints(const Ptr<Board> &board, InputArrayOfArrays detectedCorners,
                                               InputArray detectedIds, OutputArray objPoints, OutputArray imgPoints);

/**
 * @brief Calibrate a camera using aruco markers
 *
 * @param corners vector of detected marker corners in all frames.
 * The corners should have the same format returned by detectMarkers (see #detectMarkers).
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
 * @param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic parameters.
 * Order of deviations values:
 * \f$(f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6 , s_1, s_2, s_3,
 * s_4, \tau_x, \tau_y)\f$ If one of parameters is not estimated, it's deviation is equals to zero.
 * @param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic parameters.
 * Order of deviations values: \f$(R_1, T_1, \dotsc , R_M, T_M)\f$ where M is number of pattern views,
 * \f$R_i, T_i\f$ are concatenated 1x3 vectors.
 * @param perViewErrors Output vector of average re-projection errors estimated for each pattern view.
 * @param flags flags Different flags  for the calibration process (see #calibrateCamera for details).
 * @param criteria Termination criteria for the iterative optimization algorithm.
 *
 * This function calibrates a camera using an Aruco Board. The function receives a list of
 * detected markers from several views of the Board. The process is similar to the chessboard
 * calibration in calibrateCamera(). The function returns the final re-projection error.
 */
CV_EXPORTS_AS(calibrateCameraArucoExtended)
double calibrateCameraAruco(InputArrayOfArrays corners, InputArray ids, InputArray counter, const Ptr<Board> &board,
                            Size imageSize, InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                            OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, OutputArray stdDeviationsIntrinsics,
                            OutputArray stdDeviationsExtrinsics, OutputArray perViewErrors, int flags = 0,
                            TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON));

/** @brief It's the same function as #calibrateCameraAruco but without calibration error estimation.
 */
CV_EXPORTS_W double calibrateCameraAruco(InputArrayOfArrays corners, InputArray ids, InputArray counter,
                                         const Ptr<Board> &board, Size imageSize, InputOutputArray cameraMatrix,
                                         InputOutputArray distCoeffs, OutputArrayOfArrays rvecs = noArray(),
                                         OutputArrayOfArrays tvecs = noArray(), int flags = 0,
                                         TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,
                                                                              30, DBL_EPSILON));

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
 * (see cv::Rodrigues).
 * @param tvec Output vector (e.g. cv::Mat) corresponding to the translation vector of the board.
 * @param useExtrinsicGuess defines whether initial guess for \b rvec and \b tvec will be used or not.
 *
 * This function estimates a Charuco board pose from some detected corners.
 * The function checks if the input corners are enough and valid to perform pose estimation.
 * If pose estimation is valid, returns true, else returns false.
 * @sa use cv::drawFrameAxes to get world coordinate system axis for object points
 */
CV_EXPORTS_W bool estimatePoseCharucoBoard(InputArray charucoCorners, InputArray charucoIds,
                                           const Ptr<CharucoBoard> &board, InputArray cameraMatrix,
                                           InputArray distCoeffs, InputOutputArray rvec,
                                           InputOutputArray tvec, bool useExtrinsicGuess = false);

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
 * @param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic parameters.
 * Order of deviations values:
 * \f$(f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6 , s_1, s_2, s_3,
 * s_4, \tau_x, \tau_y)\f$ If one of parameters is not estimated, it's deviation is equals to zero.
 * @param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic parameters.
 * Order of deviations values: \f$(R_1, T_1, \dotsc , R_M, T_M)\f$ where M is number of pattern views,
 * \f$R_i, T_i\f$ are concatenated 1x3 vectors.
 * @param perViewErrors Output vector of average re-projection errors estimated for each pattern view.
 * @param flags flags Different flags  for the calibration process (see #calibrateCamera for details).
 * @param criteria Termination criteria for the iterative optimization algorithm.
 *
 * This function calibrates a camera using a set of corners of a  Charuco Board. The function
 * receives a list of detected corners and its identifiers from several views of the Board.
 * The function returns the final re-projection error.
 */
CV_EXPORTS_AS(calibrateCameraCharucoExtended)
double calibrateCameraCharuco(InputArrayOfArrays charucoCorners, InputArrayOfArrays charucoIds,
                              const Ptr<CharucoBoard> &board, Size imageSize, InputOutputArray cameraMatrix,
                              InputOutputArray distCoeffs, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                              OutputArray stdDeviationsIntrinsics, OutputArray stdDeviationsExtrinsics,
                              OutputArray perViewErrors, int flags = 0, TermCriteria criteria = TermCriteria(
                              TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON));

/** @brief It's the same function as #calibrateCameraCharuco but without calibration error estimation.
*/
CV_EXPORTS_W double calibrateCameraCharuco(InputArrayOfArrays charucoCorners, InputArrayOfArrays charucoIds,
                                           const Ptr<CharucoBoard> &board, Size imageSize,
                                           InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                           OutputArrayOfArrays rvecs = noArray(),
                                           OutputArrayOfArrays tvecs = noArray(), int flags = 0,
                                           TermCriteria criteria=TermCriteria(TermCriteria::COUNT +
                                                                 TermCriteria::EPS, 30, DBL_EPSILON));
//! @}

}
}
#endif
