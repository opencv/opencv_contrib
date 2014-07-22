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
  N = 4;   // NxN is the size of the block for downsampling in the lowlowResolutionDetection
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
bool MotionSaliencyBinWangApr2014::fullResolutionDetection( const Mat& image, Mat& highResBFMask )
{
  float* currentB;
  float* currentC;
  float currentPixelValue;
  float currentEpslonValue;
  bool backgFlag = false;

  // Initially, all pixels are considered as foreground and then we evaluate with the background model
  highResBFMask.create( image.rows, image.cols, CV_8UC1 );
  highResBFMask.setTo( 1 );

  // Scan all pixels of image
  for ( int i = 0; i < image.rows; i++ )
  {
    for ( int j = 0; j < image.cols; j++ )
    {
      backgFlag = false;
      // TODO replace "at" with more efficient matrix access
      currentPixelValue = image.at<uchar>( i, j );
      currentEpslonValue = epslonPixelsValue.at<float>( i, j );

      // scan background model vector
      for ( size_t z = 0; z <backgroundModel.size(); z++ )
      {
        // TODO replace "at" with more efficient matrix access
        currentB = &backgroundModel[z].at<Vec2f>( i, j )[0];
        currentC = &backgroundModel[z].at<Vec2f>( i, j )[1];

        if( *currentC > 0 )  //The current template is active
        {
          // If there is a match with a current background template
          if( abs( currentPixelValue - * ( currentB ) ) < currentEpslonValue && !backgFlag )
          {
            // The correspondence pixel in the  BF mask is set as background ( 0 value)
            // TODO replace "at" with more efficient matrix access
            highResBFMask.at<uchar>( i, j ) = 0;

            if((*currentC< L0 && z==0) || (*currentC< L1 && z==1) || (z>1))
              *currentC += 1;  // increment the efficacy of this template

            *currentB = ( ( 1 - alpha ) * * ( currentB ) ) + ( alpha * currentPixelValue );  // Update the template value
            backgFlag = true;
            //break;
          }
          else
          {
            currentC -= 1;  // decrement the efficacy of this template
          }

        }
      }  // end "for" cicle of template vector

    }
  }  // end "for" cicle of all image's pixels

  return true;
}

//typedef Rect_<uint> Rect;
bool MotionSaliencyBinWangApr2014::lowResolutionDetection( const Mat& image, Mat& lowResBFMask )
{
  float currentPixelValue;
  float currentEpslonValue;

  //Size resizedDimension = image.size() / ( N * N );
  Mat resizedImageResults( image.size().height / ( N * N ), image.size().width / ( N * N ), CV_8UC1 );
  Mat resizedBackGroundModelResults( image.size().height / ( N * N ), image.size().width / ( N * N ), CV_32FC2 );

  // Create a mask to select ROI in the original Image and Backgound model and at the same time compute the mean
  Mat ROIMask( image.rows, image.cols, CV_8UC1 );
  ROIMask.setTo( 0 );

  Rect roi( Point( 0, 0 ), Size( N, N ) );
  Scalar imageROImean;
  Scalar backGModelROImean;
  Mat currentModel;

  // Initially, all pixels are considered as foreground and then we evaluate with the background model
  lowResBFMask.create( image.size().height / ( N * N ), image.size().width / ( N * N ), CV_8UC1 );
  lowResBFMask.setTo( 1 );

  // Scan all the ROI of original matrices that correspond to the pixels of new resized matrices
  for ( int i = 0; i < resizedImageResults.rows; i++ )
  {
    for ( int j = 0; j < resizedImageResults.cols; j++ )
    {
      // Reser ROI mask
      ROIMask.setTo( 0 );
      // Compute the mean of image's block based on ROI
      rectangle( ROIMask, roi, Scalar( 255 ), FILLED );
      imageROImean = mean( image, ROIMask );

      // Compute the mean of epslonMatrix's block based on ROI

      // Insert the just calculated mean value in the correct pixel of the resized image
      resizedImageResults.at<uchar>( i, j ) = imageROImean.val[0];

      // TODO replace "at" with more efficient matrix access
      currentPixelValue = imageROImean.val[0];
      currentEpslonValue = mean(epslonPixelsValue, ROIMask).val[0];

      // scan background model vector
      for ( size_t z = 0; z < 2 /* first two template*/; z++ )
      {
        // Select the current template 2 channel matrix, select ROI and compute the mean for each channel separately
        currentModel = backgroundModel[z];
        backGModelROImean = mean( currentModel, ROIMask );
        // Insert the just calculated mean values in the correct pixels of the resized background template
        resizedBackGroundModelResults.at<Vec2f>( i, j )[0] = backGModelROImean.val[0];
        resizedBackGroundModelResults.at<Vec2f>( i, j )[1] = backGModelROImean.val[1];
      }
      // Shift the ROI from left to right follow the block dimension
      roi = roi + Point( 0, N );
    }
    //Shift the ROI from up to down follow the block dimension, also bringing it back to beginning of row
    roi = roi + Point( N, - ( image.cols - N ) );
  }

//resize( image, resizedImage, resizedDimension, 0, 0, INTER_LINEAR );
//resize( image, resizedBackGroundModel, resizedDimension, 0, 0, INTER_LINEAR );

  return true;
}
/*bool MotionSaliencyBinWangApr2014::templateUpdate( Mat highResBFMask )
 {

 return true;
 }*/

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

  Mat Test( 36, 36, CV_32F );
  Mat Results;

  std::ofstream ofs;
  ofs.open( "TEST.txt", std::ofstream::out );

  for ( int i = 0; i < Test.size().height; i++ )
  {
    for ( int j = 0; j < Test.size().width; j++ )
    {
      Test.at<float>( i, j ) = i + j;
      stringstream str;
      str << i + j << " ";
      ofs << str.str();
    }
    stringstream str2;
    str2 << "\n";
    ofs << str2.str();
  }
  ofs.close();

//blur( Test, Results, Size( 4, 4 ) );
  medianBlur( Test, Results, 3 );
//pyrDown(Results,Results, Size(Test.size().height/9, Test.size().width/9));

  std::ofstream ofs2;
  ofs2.open( "RESULTS.txt", std::ofstream::out );

  for ( int i = 0; i < Results.size().height; i++ )
  {
    for ( int j = 0; j < Results.size().width; j++ )
    {
      stringstream str;
      str << Results.at<float>( i, j ) << " ";
      ofs2 << str.str();
    }
    stringstream str2;
    str2 << "\n";
    ofs2 << str2.str();
  }
  ofs2.close();

  std::cout << "TEST SIZE: " << Test.size().height << " " << Test.size().width << "    RESULTS SIZE: " << Results.size().height << " "
            << Results.size().width << std::endl;

  return true;
}

}  // namespace cv
