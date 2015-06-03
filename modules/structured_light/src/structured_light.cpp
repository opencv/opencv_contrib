/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
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

#include "precomp.hpp"

namespace cv
{
namespace structured_light
{
// Destructor
StructuredLightPattern::~StructuredLightPattern() {}
    
/* Generates the structured light pattern.
*  patternImages   The generated pattern.
*/   
bool StructuredLightPattern::generate( OutputArrayOfArrays patternImages )
{
    //valutare CV_Error(Error::StsNotImplemented, "");
    return false;
}
	
/* Decodes the structured light pattern, generating a disparity map
* patternImages The pattern to decode.
* disparityMap The decoding result: a disparity map.
*/    
bool StructuredLightPattern::decode( InputArrayOfArrays patternImages, 
	                                 OutputArray disparityMap)
{
	return false;
}

//'load intrinsics and extrinsics parameters'
bool loadCameraCalibrationParameters(std::string path, 
                                                  OutputArray cameraMatrix1, 
                                                  OutputArray cameraMatrix2, 
                                                  OutputArray distCoeffs1, 
                                                  OutputArray distCoeffs2, 
                                                  OutputArray rotationMatrix1,
                                                  OutputArray rotationMatrix2, 
                                                  OutputArray translationVector1, 
                                                  OutputArray translationVector2)
{
	return true;
}
        
//save intrinsics and extrinsics parameters using cv::FileStorage'
bool saveCalibrationParameters(std::string path, 
                                            InputArray cameraMatrix1, 
                                            InputArray cameraMatrix2, 
                                            InputArray distCoeffs1, 
                                            InputArray distCoeffs2,
                                            InputArray rotationMatrix1, 
                                            InputArray rotationMatrix2, 
                                            InputArray translationVector1, 
                                            InputArray translationVector2)
{
	return true;
}
        
/* calibrate the cameras (intrinsics and extrinsics parameters) using the calssical openCV calibration functions'
   the input is a vector of images or it could also be a list of names'
   it fills camMatrix, distortion,rotationMatrix, translationVector*/
bool camerasProjectorCalibrate( InputArrayOfArrays gridImages, 
                                OutputArray cameraMatrix1, 
                                OutputArray cameraMatrix2,  
                                OutputArray distCoeffs1, 
                                OutputArray distCoeffs2, 
                                OutputArray rotationMatrix1, 
                                OutputArray rotationMatrix2, 
                                OutputArray translationVector1, 
                                OutputArray translationVector2)
{
	return true;
}


}
}
