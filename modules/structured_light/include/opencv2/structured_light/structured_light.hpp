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
 // Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 // Third party copyrights are property of their respective owners.
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

#ifndef __OPENCV_STRUCTURED_LIGHT_HPP__
#define __OPENCV_STRUCTURED_LIGHT_HPP__

#include "opencv2/core.hpp"

namespace cv {
namespace structured_light {
//! @addtogroup structured_light
//! @{

//! type of the decoding algorithm
enum
{
  DECODE_3D_UNDERWORLD = 0  //!< K. Herakleous, C. Poullis. “3DUNDERWORLD-SLS: An Open-Source Structured-Light Scanning System for Rapid Geometry Acquisition”, ICT-TR-2014-01
// other algorithms can be implemented
};

/** @brief Abstract base class for generating and decoding structured light pattern.
 */
class CV_EXPORTS_W StructuredLightPattern : public virtual Algorithm
{
 public:
  /** @brief Generates the structured light pattern.

   @param patternImages The generated pattern: a std::vector<cv::Mat>
   @param darkColor The dark color of the pattern; default is black.
   @param lightColor The light color of the pattern; default is white.
   */
  CV_WRAP
  virtual bool generate(OutputArrayOfArrays patternImages, const Scalar darkColor = Scalar(0, 0, 0),
                        const Scalar lightColor = Scalar(255, 255, 255));

  /** @brief Decodes the structured light pattern, generating a disparity map

   @param patternImages The pattern to decode.
   @param camerasMatrix The intrinsics of the cameras.
   @param camerasDistCoeffs The distortion coefficients of the cameras.
   @param camerasRotationMatrix The rotation matrix of the cameras.
   @param camerasTranslationVector The translation vector of the cameras.
   @param disparityMap The decoding result: a disparity map.
   @param darkImages The all-dark images needed for shadowMasks computation.
   @param lightImages The all-light images needed for shadowMasks computation.
   @param flags Flags setting decoding algorithms.
   */
  CV_WRAP
  virtual bool decode(InputArrayOfArrays patternImages, InputArrayOfArrays camerasMatrix,
                      InputArrayOfArrays camerasDistCoeffs, InputArrayOfArrays camerasRotationMatrix,
                      InputArrayOfArrays camerasTranslationVector, OutputArray disparityMap,
                      InputArrayOfArrays darkImages = noArray(), InputArrayOfArrays lightImages = noArray(), int flags =
                          DECODE_3D_UNDERWORLD) const;
};

/** @brief Class implementing the Gray Code Pattern
 */
class CV_EXPORTS_W GrayCodePattern : public StructuredLightPattern
{
 public:

  struct CV_EXPORTS_W_SIMPLE Params
  {
    CV_WRAP
    Params();CV_PROP_RW
    int width;CV_PROP_RW
    int height;
  };

  /** @brief Constructor
   @param parameters GrayCodePattern parameters GrayCodePattern::Params
   */
  CV_WRAP
  static Ptr<GrayCodePattern> create(const GrayCodePattern::Params &parameters = GrayCodePattern::Params());

  /** @brief Sets the value for set the value for light threshold, needed for decoding.

   @param value The desired light threshold value.
   */
  CV_WRAP
  virtual void setLightThreshold(int value) = 0;

  /** @brief Sets the value for dark threshold, needed for decoding.

   @param value The desired dark threshold value.
   */
  CV_WRAP
  virtual void setDarkThreshold(int value) = 0;

  /** @brief Generates The all-dark and all-light images needed for shadowMasks computation.
   *  @param darkImage The generated all-dark image.
   *  @param lightImage The generated all-light image.
   */
  CV_WRAP
  virtual void getImagesForShadowMasks(InputOutputArray darkImage, InputOutputArray lightImage) const = 0;
};

/* @brief Load intrinsics and extrinsics parameters.
 *  @param path Location where the parameters are stored.
 *  @param cameraMatrix1 The intrinsics of first camera.
 *  @param cameraMatrix2 The intrinsics of second camera.
 *  @param distCoeffs1 The distortion coefficients of first camera.
 *  @param distCoeffs1 The distortion coefficients of second camera.
 *  @param rotationMatrix1 The rotation matrix of the first camera.
 *  @param rotationMatrix2 The rotation matrix of the second camera.
 *  @param translationVector1 The translation vector of first camera.
 *  @param translationVector2 he translation vector of second camera.
 */
CV_EXPORTS_W bool loadCameraCalibrationParameters(const std::string& path, OutputArray cameraMatrix1,
                                                  OutputArray cameraMatrix2, OutputArray distCoeffs1,
                                                  OutputArray distCoeffs2, OutputArray rotationMatrix1,
                                                  OutputArray rotationMatrix2, OutputArray translationVector1,
                                                  OutputArray translationVector2);

/* @brief Saves cameras intrinsics and extrinsics parameters (using cv::FileStorage).
 *  @param path Location where to save the parameters.
 *  @param cameraMatrix1 The intrinsics of first camera.
 *  @param cameraMatrix2 The intrinsics of second camera.
 *  @param distCoeffs1 The distortion coefficients of first camera.
 *  @param distCoeffs1 The distortion coefficients of second camera.
 *  @param rotationMatrix1 The rotation matrix of the first camera.
 *  @param rotationMatrix2 The rotation matrix of the second camera.
 *  @param translationVector1 The translation vector of first camera.
 *  @param translationVector2 he translation vector of second camera.
 */
CV_EXPORTS_W bool saveCalibrationParameters(const std::string& path, InputArray cameraMatrix1, InputArray cameraMatrix2,
                                            InputArray distCoeffs1, InputArray distCoeffs2, InputArray rotationMatrix1,
                                            InputArray rotationMatrix2, InputArray translationVector1,
                                            InputArray translationVector2);

/* @brief Calibrates the cameras (intrinsics and extrinsics parameters) using the classical OpenCV calibration functions.
 *  @param gridImages Images of calibration grid acquired with both cameras.
 *  @param cameraMatrix1 The intrinsics of first camera.
 *  @param cameraMatrix2 The intrinsics of second camera.
 *  @param distCoeffs1 The distortion coefficients of first camera.
 *  @param distCoeffs1 The distortion coefficients of second camera.
 *  @param rotationMatrix1 The rotation matrix of the first camera.
 *  @param rotationMatrix2 The rotation matrix of the second camera.
 *  @param translationVector1 The translation vector of first camera.
 *  @param translationVector2 he translation vector of second camera.
 */
CV_EXPORTS_W bool camerasProjectorCalibrate(InputArrayOfArrays gridImages, OutputArray cameraMatrix1,
                                            OutputArray cameraMatrix2, OutputArray distCoeffs1, OutputArray distCoeffs2,
                                            OutputArray rotationMatrix1, OutputArray rotationMatrix2,
                                            OutputArray translationVector1, OutputArray translationVector2);

//! @}

}
}
#endif