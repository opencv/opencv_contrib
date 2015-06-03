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

namespace cv
{
namespace structured_light
{
/** @brief Abstract base class for generating and decoding structured light pattern.
*/
class CV_EXPORTS_W StructuredLightPattern : public virtual Algorithm
{
public:
	virtual ~StructuredLightPattern();
	
    /** @brief Generates the structured light pattern.

    @param patternImages The generated pattern.
    */                            
	CV_WRAP virtual bool generate( OutputArrayOfArrays patternImages );
	
	/** @brief Decodes the structured light pattern, generating a disparity map

    @param patternImages The pattern to decode.
    @param disparityMap The decoding result: a disparity map.
    */    
	CV_WRAP virtual bool decode( InputArrayOfArrays patternImages, 
	                             cv::OutputArray disparityMap);
};

/** @brief Class implementing the structured light pattern generetor and decoder described in @cite 3D underworld SLS.
 */
class CV_EXPORTS_W GrayCodePattern : public StructuredLightPattern
{
public:
	 /** @brief The GrayCodePattern constructor.

    @param proj_width The projector width.
    @param proj_heigth The projector heigth.
     */
    CV_WRAP static Ptr<GrayCodePattern> create( int proj_width, int proj_heigth);
    
    /** @brief Generates the structured light pattern.

    @param patternImages The generated pattern.
    */                            
	CV_WRAP virtual bool generate( OutputArrayOfArrays patternImages ) =0;
   
    /** @brief Sets the value for set the value for white threshold.

    @param value The desired white thershold value.
     */
    CV_WRAP virtual void setWhiteThreshold(int value)=0;
    
     /** @brief Sets the value for set the value for black threshold.

    @param value The desired black thershold value.
     */
    CV_WRAP virtual void setBlackThreshold(int value)=0;
    
    /** @brief Decodes the structured light pattern, generating a disparity map

    @param patternImages The pattern to decode.
    @param disparityMap The decoding result: a disparity map.
    */    
	CV_WRAP virtual bool decode( InputArrayOfArrays patternImages, 
	                             InputArray cameraMatrix1, 
	                             InputArray cameraMatrix2, 
	                             InputArray distCoeffs1, 
	                             InputArray distCoeffs2, 
	                             InputArray rotationMatrix1, 
	                             InputArray rotationMatrix2, 
	                             InputArray translationVector1, 
	                             InputArray translationVector2,
	                             cv::OutputArray disparityMap )=0;
};


//'load intrinsics and extrinsics parameters'
CV_EXPORTS_W bool loadCameraCalibrationParameters(std::string path, 
                                                  OutputArray cameraMatrix1, 
                                                  OutputArray cameraMatrix2, 
                                                  OutputArray distCoeffs1, 
                                                  OutputArray distCoeffs2, 
                                                  OutputArray rotationMatrix1,
                                                  OutputArray rotationMatrix2, 
                                                  OutputArray translationVector1, 
                                                  OutputArray translationVector2 );
        
//save intrinsics and extrinsics parameters using cv::FileStorage'
CV_EXPORTS_W bool saveCalibrationParameters( std::string path, 
                                             InputArray cameraMatrix1, 
                                             InputArray cameraMatrix2, 
                                             InputArray distCoeffs1, 
                                             InputArray distCoeffs2,
                                             InputArray rotationMatrix1, 
                                             InputArray rotationMatrix2, 
                                             InputArray translationVector1, 
                                             InputArray translationVector2 ); 
        
/* calibrate the cameras (intrinsics and extrinsics parameters) using the calssical openCV calibration functions'
   the input is a vector of images or it could also be a list of names'
   it fills camMatrix, distortion,rotationMatrix, translationVector*/
CV_EXPORTS_W bool camerasProjectorCalibrate( InputArrayOfArrays gridImages, 
                                             OutputArray cameraMatrix1, 
                                             OutputArray cameraMatrix2,  
                                             OutputArray distCoeffs1, 
                                             OutputArray distCoeffs2, 
                                             OutputArray rotationMatrix1, 
                                             OutputArray rotationMatrix2, 
                                             OutputArray translationVector1, 
                                             OutputArray translationVector2 );

}
}
#endif
