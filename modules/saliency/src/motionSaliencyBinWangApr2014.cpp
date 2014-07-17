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
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

cv::Ptr<Size> MotionSaliencyBinWangApr2014::getWsize()
{
  return imgSize;
}
void MotionSaliencyBinWangApr2014::setWsize( const cv::Ptr<Size>& newSize )
{
  imgSize = newSize;
}

MotionSaliencyBinWangApr2014::MotionSaliencyBinWangApr2014()
{

  epslonPixelsValue = Mat::zeros( imgSize->height, imgSize->width, CV_8U );
  potentialBackground = Mat::zeros( imgSize->height, imgSize->width, CV_32FC2 );
  backgroundModel=std::vector<Mat>( 4, Mat::zeros( imgSize->height, imgSize->width, CV_32FC2 ) );

  K = 3;  // Number of background model template
  alpha = 0.01;  // Learning rate
  L0 = 6000;  // Upper-bound values for C0 (efficacy of the first template (matrices) of backgroundModel
  L1 = 4000;  // Upper-bound values for C1 (efficacy of the second template (matrices) of backgroundModel
  thetaL = 2500;  // T0, T1 swap threshold
  thetaA = 200;
  gamma = 3;

  className = "BinWangApr2014";
}

MotionSaliencyBinWangApr2014::~MotionSaliencyBinWangApr2014()
{

}

// classification (and adaptation) functions
bool MotionSaliencyBinWangApr2014::fullResolutionDetection( Mat image, Mat highResBFMask )
{

  return true;
}
bool MotionSaliencyBinWangApr2014::lowResolutionDetection( Mat image, Mat lowResBFMask )
{

  return true;
}
bool MotionSaliencyBinWangApr2014::templateUpdate( Mat highResBFMask )
{

  return true;
}

// Background model maintenance functions
bool MotionSaliencyBinWangApr2014::templateOrdering()
{

  return true;
}
bool MotionSaliencyBinWangApr2014::templateReplacement( Mat finalBFMask )
{

  return true;
}

bool MotionSaliencyBinWangApr2014::computeSaliencyImpl( const InputArray image, OutputArray saliencyMap )
{

  return true;
}

}  // namespace cv
