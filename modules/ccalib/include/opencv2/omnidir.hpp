/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2015, Baisheng Lai (laibaisheng@gmail.com), Zhejiang University,
// all rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#ifndef __OPENCV_OMNIDIR_HPP__
#define __OPENCV_OMNIDIR_HPP__
#ifdef __cplusplus

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
namespace cv
{

/* @defgroup calib3d_omnidir Omnidirectional camera model\

    Here is a brief description of implenmented omnidirectional camera model. This model can be
    used for both catadioptric and fisheye cameras. Especially, catadioptric cameras have very
    large field of view (FOV), i.e., a 360 degrees of horizontal FOV, means the scene around the
    camera can be all taken in a singel photo. Compared with perspective cameras, omnidirectional
    cameras get more information in a single shot and avoid things like image stitching.

    The large FOV of omnidirectional cameras also introduces large distortion, so that it is
    not vivid for human's eye. Rectification that removes distortion is also included in this module.

    For a 3D point Xw in world coordinate, it is first transformed to camera coordinate:

    \f[X_c = R X_w + T \f]

    where R and T are rotation and translation matrix. Then \f$ X_c \f$ is then projected to unit sphere:

    \f[ X_s = \frac{Xc}{||Xc||}  \f]

    Let \f$ X_s = (x, y, z) \f$, then \f$ X_s \f$ is projected to normalized plane:

    \f[ (x_u, y_u, 1) = (\frac{x}{z + \xi}, \frac{y}{z + \xi}, 1) \f]

    where \f$ \xi \f$ is a parameter of camera. So far the point contains no distortion, add distortion by

    \f[ x_d = (1 + k_1 r^2 + k_2 r^4 )*x_u + 2p_1 x_u y_u + p_2(r^2 + 2x_u^2 )  \\
        y_d = (1 + k_1 r^2 + k_2 r^4 )*y_u + p_1 (r^2 + 2y_u^2) + 2p_2 x_u y_u \f]

    where \f$ r^2 = x_u^2 + y_u^2\f$ and \f$(k_1, k_2, p_1, p_2)\f$ are distortion coefficients.

    At last, convert to pixel coordinates:

    \f[ u = f_x x_d + s y_d + c_x \\
        v = f_y y_d + c_y \f]

    where \f$ s\f$ is the skew coefficient and \f$ (cx, cy\f$ are image centers.
*/
/** @brief The methods in this namespace is to calibrate omnidirectional cameras.
    This module was accepted as a GSoC 2015 project for OpenCV, authored by
    Baisheng Lai, mentored by Bo Li.
  @ingroup calib3d_omnidir
*/
namespace omnidir
{
    //! @addtogroup calib3d_omnidir
    //! @{
    enum {
        CALIB_USE_GUESS             = 1,
        CALIB_FIX_SKEW              = 2,
        CALIB_FIX_K1                = 4,
        CALIB_FIX_K2                = 8,
        CALIB_FIX_P1                = 16,
        CALIB_FIX_P2                = 32,
        CALIB_FIX_XI                = 64,
        CALIB_FIX_GAMMA             = 128,
        CALIB_FIX_CENTER            = 256
    };

    /** @brief Projects points for omnidirectional camera using CMei's model

    @param objectPoints Object points in world coordiante, 1xN/Nx1 3-channel of type CV_64F and N
    is the number of points.
    @param imagePoints Output array of image points, 1xN/Nx1 2-channel of type CV_64F
    @param rvec vector of rotation between world coordinate and camera coordinate, i.e., om
    @param tvec vector of translation between pattern coordinate and camera coordinate
    @param K Camera matrix \f$K = \vecthreethree{f_x}{s}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, p_1, p_2)\f$.
    @param xi The parameter xi for CMei's model
    @param jacobian Optional output 2Nx16 of type CV_64F jacobian matrix, constains the derivatives of
    image pixel points wrt parametes including \f$om, T, f_x, f_y, s, c_x, c_y, xi, k_1, k_2, p_1, p_2\f$.
    This matrix will be used in calibration by optimization.

    The function projects object 3D points of world coordiante to image pixels, parametered by intrinsic
    and extrinsic parameters. Also, it optionaly compute a by-product: the jacobian matrix containing
    onstains the derivatives of image pixel points wrt intrinsic and extrinsic parametes.
     */
    CV_EXPORTS_W void projectPoints(InputArray objectPoints, OutputArray imagePoints, InputArray rvec, InputArray tvec,
                       InputArray K, double xi, InputArray D, OutputArray jacobian = noArray());

    /** @brief Undistort 2D image points for omnidirectional camera using CMei's model

    @param distorted Array of distorted image points, 1xN/Nx1 2-channel of tyep CV_64F
    @param K Camera matrix \f$K = \vecthreethree{f_x}{s}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Distortion coefficients \f$(k_1, k_2, p_1, p_2)\f$.
    @param xi The parameter xi for CMei's model
    @param R Rotation trainsform between the original and object space : 3x3 1-channel, or vector: 3x1/1x3
    1-channel or 1x1 3-channel
    @param undistorted array of normalized object points, 1xN/Nx1 2-channel of type CV_64F
     */

    CV_EXPORTS_W void undistortPoints(InputArray distorted, OutputArray undistorted, InputArray K, InputArray D, double xi, InputArray R);

    /** @brief Computes undistortion and rectification maps for omnidirectional camera image transform by cv::remap().
    If D is empty zero distortion is used, if R or P is empty identity matrixes are used.

    @param K Camera matrix \f$K = \vecthreethree{f_x}{s}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, p_1, p_2)\f$.
    @param xi The parameter xi for CMei's model
    @param R Rotation trainsform between the original and object space : 3x3 1-channel, or vector: 3x1/1x3
    @param P New camera matrix (3x3) or new projection matrix (3x4)
    @param size Undistorted image size.
    @param mltype Type of the first output map that can be CV_32FC1 or CV_16SC2 . See convertMaps()
    for details.
    @param map1 The first output map.
    @param map2 The second output map.
     */
    CV_EXPORTS_W void initUndistortRectifyMap(InputArray K, InputArray D, double xi, InputArray R, InputArray P, const cv::Size& size,
        int mltype, OutputArray map1, OutputArray map2);

    /** @brief Undistort omnidirectional images to perspective images

    @param distorted omnidirectional image with very large distortion
    @param undistorted The output undistorted image
    @param K Camera matrix \f$K = \vecthreethree{f_x}{s}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, p_1, p_2)\f$.
    @param xi The parameter xi for CMei's model
    @param Knew Camera matrix of the distorted image. By default, it is just K.
    @param new_size The new image size. By default, it is the size of distorted.
    */
    CV_EXPORTS_W void undistortImage(InputArray distorted, OutputArray undistorted, InputArray K, InputArray D, double xi,
        InputArray Knew = cv::noArray(), const Size& new_size = Size());

        /** @brief Perform omnidirectional camera calibration

    @param patternPoints Vector of vector of pattern points in world (pattern) coordiante, 1xN/Nx1 3-channel
    @param imagePoints Vector of vector of correspoinding image points of objectPoints
    @param size Image size of calibration images.
    @param K Output calibrated camera matrix. If you want to initialize K by yourself, input a non-empty K.
    @param xi Ouput parameter xi for CMei's model
    @param D Output distortion parameters \f$(k_1, k_2, p_1, p_2)\f$
    @param omAll Output rotations for each calibration images
    @param tAll Output translation for each calibration images
    @param flags The flags that control calibrate
    @param criteria Termination criteria for optimization
    */
    CV_EXPORTS_W double calibrate(InputOutputArrayOfArrays patternPoints, InputOutputArrayOfArrays imagePoints, Size size,
        InputOutputArray K, InputOutputArray xi, InputOutputArray D, OutputArrayOfArrays omAll, OutputArrayOfArrays tAll,
        int flags, TermCriteria criteria);


//! @} calib3d_omnidir
namespace internal
{
    void initializeCalibration(InputOutputArrayOfArrays objectPoints, InputOutputArrayOfArrays imagePoints, Size size, OutputArrayOfArrays omAll, OutputArrayOfArrays tAll, OutputArray K, double& xi);
    void computeJacobian(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, InputArray parameters, Mat& JTJ_inv, Mat& JTE, int flags);
    void encodeParameters(InputArray K, OutputArrayOfArrays omAll, OutputArrayOfArrays tAll, InputArray distoaration, double xi, int n, OutputArray parameters);
    void decodeParameters(InputArray paramsters, OutputArray K, OutputArrayOfArrays omAll, OutputArrayOfArrays tAll, OutputArray distoration, double& xi);
    void estimateUncertainties(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, InputArray parameters, Mat& errors, Vec2d& std_error, double& rms, int flags);
    double computeMeanReproerr(InputArrayOfArrays imagePoints, InputArrayOfArrays proImagePoints);
    void checkFixed(Mat &G, int flags, int n);
    void subMatrix(const Mat& src, Mat& dst, const std::vector<int>& cols, const std::vector<int>& rows);
    void flags2idx(int flags, std::vector<int>& idx, int n);
    void fillFixed(Mat&G, int flags, int n);
} // internal


} // omnidir

} //cv
#endif
#endif