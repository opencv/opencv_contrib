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
#include <stdlib.h>
#include <iostream>
namespace cv
{
namespace structured_light
{
class GrayCodePattern_Impl : public GrayCodePattern
{
public:
    // Constructor
    explicit GrayCodePattern_Impl(int proj_width, int proj_heigth);

    
    // Destructor
    virtual ~GrayCodePattern_Impl(){};

    // Generates the gray code pattern as a std::vector<cv::Mat>
    bool generate( OutputArrayOfArrays pattern);
    
    // Decodes the gray code pattern 
    bool decode ( InputArrayOfArrays patternImages, 
	              InputArray cameraMatrix1, 
	              InputArray cameraMatrix2, 
	              InputArray distCoeffs1, 
	              InputArray distCoeffs2, 
	              InputArray rotationMatrix1, 
	              InputArray rotationMatrix2, 
	              InputArray translationVector1, 
	              InputArray translationVector2,
	              OutputArray disparityMap );
        
    // Sets the value for black threshold
    void setBlackThreshold(int val);
        
    // Sets the value for set the value for white threshold
    void setWhiteThreshold(int val);
  
protected:
    // Converts a gray code sequence to a decimal number
    int grayToDec(std::vector<bool> gray);
		
    // Computes the required number of pattern images, allocing the pattern vector
	void computeNumberOfPatternImages();

    // The number of images of the pattern
    int numOfImgs;
		
    // The number of row images of the pattern
	int numOfRowImgs;
		
	// The number of column images of the pattern
    int numOfColImgs;
		
	// The projector height
	int height;
	
	// The projector width    
	int width;
	
	// Number between 0-255 that represents the minimum brightness difference
    // between the fully illuminated (white) and the non - illuminated images (black)
	int blackThreshold;
		
	// Number between 0-255 that represents the minimum brightness difference
    // between the gray-code pattern and its inverse images
	int whiteThreshold;
		
    // Computes the shadows occlusion where we cannot reconstruct the model'
	void computeShadowsMask(InputArray black, InputArray white, OutputArray shadows);
			
	// For a (x,y) pixel of the camera returns the corresponding projector pixel'
	void getProjPixel(InputArrayOfArrays patternImages, int x, int y, cv::Point &p_out);
};

GrayCodePattern_Impl::GrayCodePattern_Impl(int proj_width, int proj_heigth)
{
	width = proj_width;
	height = proj_heigth;
}

/* Non funziona
GrayCodePattern_Impl::~GrayCodePattern_Impl
{
	
}*/

bool 
GrayCodePattern_Impl::generate( OutputArrayOfArrays pattern)
{
	std::cout << std::endl << "pattern generato" << std::endl;
	return true;
}


bool
GrayCodePattern_Impl::decode( InputArrayOfArrays patternImages, 
                              InputArray cameraMatrix1, 
	                          InputArray cameraMatrix2, 
	                          InputArray distCoeffs1, 
	                          InputArray distCoeffs2, 
	                          InputArray rotationMatrix1, 
	                          InputArray rotationMatrix2, 
	                          InputArray translationVector1, 
	                          InputArray translationVector2,
	                          OutputArray disparityMap )
{
	return true;
}
        
// Sets the value for black threshold
void 
GrayCodePattern_Impl::setBlackThreshold(int val)
{
	return ;
}
        
// Sets the value for set the value for white threshold
void GrayCodePattern_Impl::setWhiteThreshold(int val)
{
	return ;
}

// Converts a gray code sequence to a decimal number
int 
GrayCodePattern_Impl::grayToDec(std::vector<bool> gray)
{
	return 3;
}
		
// Computes the required number of pattern images, allocing the pattern vector
void 
GrayCodePattern_Impl::computeNumberOfPatternImages()
{
	return ;
}

// Computes the shadows occlusion where we cannot reconstruct the model'
void
GrayCodePattern_Impl::computeShadowsMask(InputArray black, InputArray white, OutputArray shadows)
{
	
	return ;
}
		
// For a (x,y) pixel of the camera returns the corresponding projector pixel'
void 
GrayCodePattern_Impl::getProjPixel(InputArrayOfArrays patternImages, int x, int y, cv::Point &p_out)
{
	return ;
}

Ptr<GrayCodePattern> GrayCodePattern::create(int proj_width, int proj_heigth)
{
	std::cout << "graycode  classcreated\n"<< std::endl;
    return makePtr<GrayCodePattern_Impl>(proj_width, proj_heigth);
}

}
}





