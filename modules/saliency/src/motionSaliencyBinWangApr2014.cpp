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
//TODO delete highgui include
#include <opencv2/highgui.hpp>

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
  N_DS = 2;  // Number of template to be downsampled and used in lowResolutionDetection function
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
  //TODO set to nan
  potentialBackground.setTo( 0 );

  //TODO set to nan
  for ( size_t i = 0; i < backgroundModel.size(); i++ )
  {
    backgroundModel[i].setTo( 0 );
  }

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
      for ( size_t z = 0; z < backgroundModel.size(); z++ )
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

            if( ( *currentC < L0 && z == 0 ) || ( *currentC < L1 && z == 1 ) || ( z > 1 ) )
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

bool MotionSaliencyBinWangApr2014::lowResolutionDetection( const Mat& image, Mat& lowResBFMask )
{
  float currentPixelValue;
  float currentEpslonValue;
  float currentB;
  float currentC;

  // Create a mask to select ROI in the original Image and Backgound model and at the same time compute the mean
  Mat ROIMask( image.rows, image.cols, CV_8UC1 );
  ROIMask.setTo( 0 );

  Rect roi( Point( 0, 0 ), Size( N, N ) );
  Scalar imageROImean;
  Scalar backGModelROImean;
  Mat currentModel;

  // Initially, all pixels are considered as foreground and then we evaluate with the background model
  //lowResBFMask.create( image.size().height / ( N * N ), image.size().width / ( N * N ), CV_8UC1 );
  //lowResBFMask.setTo( 1 );

  lowResBFMask.create( image.rows, image.cols, CV_8UC1 );
  lowResBFMask.setTo( 1 );

  // Scan all the ROI of original matrices that correspond to the pixels of new resized matrices
  for ( int i = 0; i < image.rows / N; i++ )
  {
    for ( int j = 0; j < image.cols / N; j++ )
    {
      // Reset and update ROI mask
      ROIMask.setTo( 0 );
      rectangle( ROIMask, roi, Scalar( 255 ), FILLED );

      // Compute the mean of image's block and epslonMatrix's block based on ROI
      // TODO replace "at" with more efficient matrix access
      currentPixelValue = mean( image, ROIMask ).val[0];
      currentEpslonValue = mean( epslonPixelsValue, ROIMask ).val[0];

      // scan background model vector
      for ( size_t z = 0; z < N_DS; z++ )
      {

        // Select the current template 2 channel matrix, select ROI and compute the mean for each channel separately
        currentB = mean( backgroundModel[z], ROIMask ).val[0];
        currentC = mean( backgroundModel[z], ROIMask ).val[1];

        if( currentC > 0 )  //The current template is active
        {
          // If there is a match with a current background template
          if( abs( currentPixelValue - ( currentB ) ) < currentEpslonValue )
          {
            // The correspondence pixel in the  BF mask is set as background ( 0 value)
            // TODO replace "at" with more efficient matrix access
            //lowResBFMask.at<uchar>( i, j ) = 0;
            lowResBFMask.setTo( 0, ROIMask );
            break;
          }
        }
      }
      // Shift the ROI from left to right follow the block dimension
      roi = roi + Point( N, 0 );
    }
    //Shift the ROI from up to down follow the block dimension, also bringing it back to beginning of row
    roi.x = 0;
    roi.y += N;
  }

  // UPSAMPLE the lowResBFMask to the original image dimension, so that it's then possible to compare the results
  // of lowlResolutionDetection with the fullResolutionDetection
  //resize( lowResBFMask, lowResBFMask, image.size(), 0, 0, INTER_LINEAR );

  return true;
}

/*bool MotionSaliencyBinWangApr2014::templateUpdate( Mat highResBFMask )
 {

 return true;
 }*/

bool inline pairCompare( pair<float, float> t, pair<float, float> t_plusOne )
{

  return ( t.second > t_plusOne.second );

}

// Background model maintenance functions
bool MotionSaliencyBinWangApr2014::templateOrdering()
{
  vector<pair<float, float> > pixelTemplates( backgroundModel.size() );
  float temp;

  // Scan all pixels of image
  for ( int i = 0; i < backgroundModel[0].rows; i++ )
  {
    for ( int j = 0; j < backgroundModel[0].cols; j++ )
    {
      // scan background model vector from T1 to Tk
      for ( size_t z = 1; z < backgroundModel.size(); z++ )
      {
        // Fill vector of pairs
        pixelTemplates[z - 1].first = backgroundModel[z].at<Vec2f>( i, j )[0];  // Current B (background value)
        pixelTemplates[z - 1].second = backgroundModel[z].at<Vec2f>( i, j )[1];  // Current C (efficacy value)
      }

      //SORT template from T1 to Tk
      std::sort( pixelTemplates.begin(), pixelTemplates.end(), pairCompare );

      //REFILL CURRENT MODEL ( T1...Tk)
      for ( size_t zz = 1; zz < backgroundModel.size(); zz++ )
      {
        backgroundModel[zz].at<Vec2f>( i, j )[0] = pixelTemplates[zz - 1].first;  // Replace previous B with new ordered B value
        backgroundModel[zz].at<Vec2f>( i, j )[1] = pixelTemplates[zz - 1].second;  // Replace previous C with new ordered C value
      }

      // SORT Template T0 and T1
      if( backgroundModel[1].at<Vec2f>( i, j )[1] > thetaL && backgroundModel[0].at<Vec2f>( i, j )[1] < thetaL )
      {

        // swap B value of T0 with B value of T1 (for current model)
        temp = backgroundModel[0].at<Vec2f>( i, j )[0];
        backgroundModel[0].at<Vec2f>( i, j )[0] = backgroundModel[1].at<Vec2f>( i, j )[0];
        backgroundModel[1].at<Vec2f>( i, j )[0] = temp;

        // set new C0 value for current model)
        temp = backgroundModel[0].at<Vec2f>( i, j )[1];
        backgroundModel[0].at<Vec2f>( i, j )[1] = gamma * thetaL;
        backgroundModel[1].at<Vec2f>( i, j )[1] = temp;

      }

    }
  }

  return true;
}
bool MotionSaliencyBinWangApr2014::templateReplacement( Mat finalBFMask, Mat image )
{
  int roiSize = 3;
  int countNonZeroElements = NAN;
  Mat replicateCurrentBAMat( roiSize, roiSize, CV_8U );
  Mat backgroundModelROI( roiSize, roiSize, CV_32F );
  Mat diffResult( roiSize, roiSize, CV_32F );

  // Scan all pixels of finalBFMask and all pixels of others models (the dimension are the same)
  for ( int i = 0; i < finalBFMask.rows; i++ )
  {
    for ( int j = 0; j < finalBFMask.cols; j++ )
    {
      /////////////////// MAINTENANCE of potentialBackground model ///////////////////
      if( finalBFMask.at<uchar>( i, j ) == 1 )  // i.e. the corresponding frame pixel has been market as foreground
      {
        /* For the pixels with CA= 0, if the current frame pixel has been classified as foreground, its value
         * will be loaded into BA and CA will be set to 1*/
        if( potentialBackground.at<Vec2f>( i, j )[1] == 0 )
        {
          potentialBackground.at<Vec2f>( i, j )[0] = image.at<uchar>( i, j );
          potentialBackground.at<Vec2f>( i, j )[1] = 1;
        }

        /*the distance between this pixel value and BA is calculated, and if this distance is smaller than
         the decision threshold epslon, then CA is increased by 1, otherwise is decreased by 1*/
        else if( abs( image.at<uchar>( i, j ) - potentialBackground.at<Vec2f>( i, j )[0] ) < epslonPixelsValue.at<float>( i, j ) )
        {
          potentialBackground.at<Vec2f>( i, j )[1] += 1;
        }
        else
        {
          potentialBackground.at<Vec2f>( i, j )[1] -= 1;
        }
      }  /////////////////// END of potentialBackground model MAINTENANCE///////////////////

      /////////////////// EVALUATION of potentialBackground values ///////////////////
      if( potentialBackground.at<Vec2f>( i, j )[1] > thetaA )
      {
        // replicate currentBA value
        replicateCurrentBAMat.setTo( potentialBackground.at<Vec2f>( i, j )[0] );
        for ( size_t z = 1; z < backgroundModel.size(); z++ )
        {
          // Neighborhood of current pixel in the current background model template.
          // The ROI is centered in the pixel coordinates
          // TODO border check
          backgroundModelROI = ( backgroundModel[z], Rect( i - floor( roiSize / 2 ), j - floor( roiSize / 2 ), roiSize, roiSize ) );

          /* Check if the value of current pixel BA in potentialBackground model is already contained in at least one of its neighbors'
           * background model
           */
          absdiff( replicateCurrentBAMat, backgroundModelROI, diffResult );
          threshold( diffResult, diffResult, epslonPixelsValue.at<float>( i, j ), 255, THRESH_BINARY_INV );
          countNonZeroElements = countNonZero( diffResult );

          if( countNonZeroElements > 0 )
          {
            /////////////////// REPLACEMENT of backgroundModel template ///////////////////
            //replace TA with current TK
            break;

          }
        }
      }

    }  // end of second for
  }  // end of first for

  return true;
}

bool MotionSaliencyBinWangApr2014::computeSaliencyImpl( const InputArray image, OutputArray saliencyMap )
{

  Mat highResBFMask;
  Mat lowResBFMask;
  Mat not_lowResBFMask;
  Mat finalBFMask;
  Mat noisePixelsMask;
  /*Mat t( image.getMat().rows, image.getMat().cols, CV_32FC2 );
   t.setTo( 50 );
   backgroundModel.at( 0 ) = t; */

  fullResolutionDetection( image.getMat(), highResBFMask );
  lowResolutionDetection( image.getMat(), lowResBFMask );

// Compute the final background-foreground mask. One pixel is marked as foreground if and only if it is
// foreground in both masks (full and low)
  bitwise_and( highResBFMask, lowResBFMask, finalBFMask );

// Detect the noise pixels (i.e. for a given pixel, fullRes(pixel) = foreground and lowRes(pixel)= background)
  bitwise_not( lowResBFMask, not_lowResBFMask );
  bitwise_and( highResBFMask, not_lowResBFMask, noisePixelsMask );

  templateOrdering();
  templateReplacement( finalBFMask, image.getMat() );
  templateOrdering();

  return true;
}

}  // namespace cv
