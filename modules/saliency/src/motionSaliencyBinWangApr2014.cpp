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

  K = 3;  // Number of background model template
  alpha = 0.01;  // Learning rate
  L0 = 6000;  // Upper-bound values for C0 (efficacy of the first template (matrices) of backgroundModel
  L1 = 4000;  // Upper-bound values for C1 (efficacy of the second template (matrices) of backgroundModel
  thetaL = 2500;  // T0, T1 swap threshold
  thetaA = 200;
  gamma = 3;

  className = "BinWangApr2014";
}

bool MotionSaliencyBinWangApr2014::init()
{

  epslonPixelsValue = Mat( imgSize->height, imgSize->width, CV_32F );
  potentialBackground = Mat( imgSize->height, imgSize->width, CV_32FC2 );
  backgroundModel = std::vector<Mat>( K + 1, Mat::zeros( imgSize->height, imgSize->width, CV_32FC2 ) );

  potentialBackground.setTo( NAN );

  for ( size_t i = 0; i < backgroundModel.size(); i++ )
    backgroundModel[i].setTo( NAN );

  epslonPixelsValue.setTo( 48.5 );  // Median of range [18, 80] advised in reference paper.
                                    // Since data is even, the median is estimated using two values ​​that occupy
                                    // the position (n / 2) and ((n / 2) +1) (choose their arithmetic mean).

  /* epslonPixelsValue = Mat::zeros( imgSize->height, imgSize->width, CV_8U );
   potentialBackground = Mat::NAN( imgSize->height, imgSize->width, CV_32FC2 );
   backgroundModel = std::vector<Mat>( 4, Mat::zeros( imgSize->height, imgSize->width, CV_32FC2 ) );*/

  return true;

}

MotionSaliencyBinWangApr2014::~MotionSaliencyBinWangApr2014()
{

}

// classification (and adaptation) functions
bool MotionSaliencyBinWangApr2014::fullResolutionDetection( Mat image, Mat highResBFMask )
{
  Mat currentTemplateMat;
  float currentB;
  float currentC;
  Vec2f elem;
  float currentPixelValue;
  float currentEpslonValue;
  //bool backgFlag=false;

  // Initially, all pixels are considered as foreground and then we evaluate with the background model
  highResBFMask.setTo( 1 );

  // Scan all pixels of image
  for ( size_t i = 0; i < image.size().width; i++ )
  {
    for ( size_t j = 0; j < image.size().height; j++ )
    {
      // TODO replace "at" with more efficient matrix access
      currentPixelValue = image.at<float>( i, j );
      currentEpslonValue = epslonPixelsValue.at<float>( i, j );

      // scan background model vector
      for ( size_t z = 0; z < backgroundModel.size(); z++ )
      {
        currentTemplateMat = backgroundModel[z];  // Current Background Model matrix
        // TODO replace "at" with more efficient matrix access
        elem = currentTemplateMat.at<Vec2f>( i, j );
        currentB = elem[0];
        currentC = elem[1];

        if( currentC > 0 )  //The current template is active
        {
          // If there is a match with a current background template
          if( abs( currentPixelValue - currentB ) < currentEpslonValue )
          {
            // The correspondence pixel in the  BF mask is set as background ( 0 value)
            // TODO replace "at" with more efficient matrix access
            highResBFMask.at<int>( i, j ) = 0;
            // backgFlag=true;
            break;
          }
        }
      }  // end "for" cicle of template vector

      // if the pixel has not matched with none of the active background models, is set as the foreground
      // in the high resolution mask
      //if(!backgFlag)
      // highResBFMask.at<int>( i, j ) = 1;
    }
  } // end "for" cicle of all image's pixels

  //////// THERE WE CALL THE templateUpdate FUNCTION (to be implemented) ////////
  templateUpdate(highResBFMask);

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
