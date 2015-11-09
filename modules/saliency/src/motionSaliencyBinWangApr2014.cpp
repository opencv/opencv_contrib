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
 // Copyright (C) 2014, OpenCV Foundation, all rights reserved.
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

#include <limits>
#include "precomp.hpp"
//TODO delete highgui include
//#include <opencv2/highgui.hpp>

#define thetaA_VAL 200
#define thetaL_VAL 250

namespace cv
{
namespace saliency
{

void MotionSaliencyBinWangApr2014::setImagesize( int W, int H )
{
  imageWidth = W;
  imageHeight = H;
}

MotionSaliencyBinWangApr2014::MotionSaliencyBinWangApr2014()
{
  N_DS = 2;  // Number of template to be downsampled and used in lowResolutionDetection function
  K = 3;  // Number of background model template
  N = 4;   // NxN is the size of the block for downsampling in the lowlowResolutionDetection
  alpha = (float) 0.01;  // Learning rate
  L0 = 1000;  // Upper-bound values for C0 (efficacy of the first template (matrices) of backgroundModel
  L1 = 800;  // Upper-bound values for C1 (efficacy of the second template (matrices) of backgroundModel
  thetaL = thetaL_VAL;  // T0, T1 swap threshold
  thetaA = thetaA_VAL;
  gamma = 3;
  neighborhoodCheck = true;

  className = "BinWangApr2014";
}

bool MotionSaliencyBinWangApr2014::init()
{
  Size imgSize( imageWidth, imageHeight );
  epslonPixelsValue = Mat( imgSize.height, imgSize.width, CV_32F, Scalar( 20 ) );
  // Median of range [18, 80] advised in reference paper.
  // Since data is even, the median is estimated using two values ​​that occupy
  // the position (n / 2) and ((n / 2) +1) (choose their arithmetic mean).

  potentialBackground = Mat( imgSize.height, imgSize.width, CV_32FC2, Scalar( std::numeric_limits<float>::quiet_NaN(), 0 ) );

  backgroundModel.resize( K + 1 );

  for ( int i = 0; i < K + 1; i++ )
  {
    Mat* tmpm = new Mat;
    tmpm->create( imgSize.height, imgSize.width, CV_32FC2 );
    tmpm->setTo( Scalar( std::numeric_limits<float>::quiet_NaN(), 0 ) );
    Ptr<Mat> tmp = Ptr<Mat>( tmpm );
    backgroundModel[i] = tmp;
  }

  return true;

}

MotionSaliencyBinWangApr2014::~MotionSaliencyBinWangApr2014()
{

}

// classification (and adaptation) functions
bool MotionSaliencyBinWangApr2014::fullResolutionDetection( const Mat& image2, Mat& highResBFMask )
{
  Mat image = image2.clone();

  float currentPixelValue;
  float currentEpslonValue;
  bool backgFlag = false;

  // Initially, all pixels are considered as foreground and then we evaluate with the background model
  highResBFMask.create( image.rows, image.cols, CV_32F );
  highResBFMask.setTo( 1 );

  uchar* pImage;
  float* pEpslon;
  float* pMask;

  // Scan all pixels of image
  for ( int i = 0; i < image.rows; i++ )
  {

    pImage = image.ptr<uchar>( i );
    pEpslon = epslonPixelsValue.ptr<float>( i );
    pMask = highResBFMask.ptr<float>( i );
    for ( int j = 0; j < image.cols; j++ )
    {

      backgFlag = false;
      currentPixelValue = pImage[j];
      currentEpslonValue = pEpslon[j];

      int counter = 0;
      for ( size_t z = 0; z < backgroundModel.size(); z++ )
      {

        counter += (int) backgroundModel[z]->ptr<Vec2f>( i )[j][1];
      }

      if( counter != 0 )  //if at least the first template is activated / initialized
      {

        // scan background model vector
        for ( size_t z = 0; z < backgroundModel.size(); z++ )
        {
          float* currentB;
          float* currentC;
          currentB = & ( backgroundModel[z]->ptr<Vec2f>( i )[j][0] );
          currentC = & ( backgroundModel[z]->ptr<Vec2f>( i )[j][1] );

          //continue;
          if( ( *currentC ) > 0 )  //The current template is active
          {
            // If there is a match with a current background template
            if( abs( currentPixelValue - ( *currentB ) ) < currentEpslonValue && !backgFlag )
            {
              // The correspondence pixel in the  BF mask is set as background ( 0 value)
              pMask[j] = 0;
              if( ( *currentC < L0 && z == 0 ) || ( *currentC < L1 && z == 1 ) || ( z > 1 ) )
              {
                *currentC += 1;  // increment the efficacy of this template
              }

              *currentB = ( ( 1 - alpha ) * ( *currentB ) ) + ( alpha * currentPixelValue );  // Update the template value
              backgFlag = true;
            }
            else
            {
              *currentC -= 1;  // decrement the efficacy of this template
            }

          }

        }  // end "for" cicle of template vector

      }
      else
      {
        pMask[j] = 1;  //if the model of the current pixel is not yet initialized, we mark the pixels as foreground
      }

    }
  }  // end "for" cicle of all image's pixels

  return true;
}

bool MotionSaliencyBinWangApr2014::lowResolutionDetection( const Mat& image, Mat& lowResBFMask )
{
  std::vector<Mat> mv;
  split( *backgroundModel[0], mv );

  //if at least the first template is activated / initialized for all pixels
  if( countNonZero( mv[1] ) > ( mv[1].cols * mv[1].rows ) / 2 )
  {
    float currentPixelValue;
    float currentEpslonValue;
    float currentB;
    float currentC;

    // Create a mask to select ROI in the original Image and Backgound model and at the same time compute the mean

    Rect roi( Point( 0, 0 ), Size( N, N ) );
    Scalar imageROImean;
    Scalar backGModelROImean;
    Mat currentModel;

    // Initially, all pixels are considered as foreground and then we evaluate with the background model
    lowResBFMask.create( image.rows, image.cols, CV_32F );
    lowResBFMask.setTo( 1 );

    // Scan all the ROI of original matrices
    for ( int i = 0; i < ceil( (float) image.rows / N ); i++ )
    {
      if( ( roi.y + ( N - 1 ) ) <= ( image.rows - 1 ) )
      {
        // Reset original ROI dimension
        roi = Rect( Point( roi.x, roi.y ), Size( N, N ) );
      }

      for ( int j = 0; j < ceil( (float) image.cols / N ); j++ )
      {
        // Compute the mean of image's block and epslonMatrix's block based on ROI
        Mat roiImage = image( roi );
        Mat roiEpslon = epslonPixelsValue( roi );
        currentPixelValue = (float) mean( roiImage ).val[0];
        currentEpslonValue = (float) mean( roiEpslon ).val[0];

        // scan background model vector
        for ( int z = 0; z < N_DS; z++ )
        {
          // Select the current template 2 channel matrix, select ROI and compute the mean for each channel separately
          Mat roiTemplate = ( * ( backgroundModel[z] ) )( roi );
          Scalar templateMean = mean( roiTemplate );
          currentB = (float) templateMean[0];
          currentC = (float) templateMean[1];

          if( ( currentC ) > 0 )  //The current template is active
          {
            // If there is a match with a current background template
            if( abs( currentPixelValue - ( currentB ) ) < currentEpslonValue )
            {
              // The correspondence pixel in the  BF mask is set as background ( 0 value)
              rectangle( lowResBFMask, roi, Scalar( 0 ), FILLED );
              break;
            }
          }
        }
        // Shift the ROI from left to right follow the block dimension
        roi = roi + Point( N, 0 );
        if( ( roi.x + ( roi.width - 1 ) ) > ( image.cols - 1 ) && ( roi.y + ( N - 1 ) ) <= ( image.rows - 1 ) )
        {
          roi = Rect( Point( roi.x, roi.y ), Size( abs( ( image.cols - 1 ) - roi.x ) + 1, N ) );
        }
        else if( ( roi.x + ( roi.width - 1 ) ) > ( image.cols - 1 ) && ( roi.y + ( N - 1 ) ) > ( image.rows - 1 ) )
        {
          roi = Rect( Point( roi.x, roi.y ), Size( abs( ( image.cols - 1 ) - roi.x ) + 1, abs( ( image.rows - 1 ) - roi.y ) + 1 ) );
        }
      }
      //Shift the ROI from up to down follow the block dimension, also bringing it back to beginning of row
      roi.x = 0;
      roi.y += N;
      if( ( roi.y + ( roi.height - 1 ) ) > ( image.rows - 1 ) )
      {
        roi = Rect( Point( roi.x, roi.y ), Size( N, abs( ( image.rows - 1 ) - roi.y ) + 1 ) );
      }

    }
    return true;
  }
  else
  {
    lowResBFMask.create( image.rows, image.cols, CV_32F );
    lowResBFMask.setTo( 1 );
    return false;
  }

}

bool inline pairCompare( std::pair<float, float> t, std::pair<float, float> t_plusOne )
{

  return ( t.second > t_plusOne.second );

}

// Background model maintenance functions
bool MotionSaliencyBinWangApr2014::templateOrdering()
{
  std::vector<std::pair<float, float> > pixelTemplates( backgroundModel.size() );

  Vec2f* bgModel_0P;
  Vec2f* bgModel_1P;

// Scan all pixels of image
  for ( int i = 0; i < backgroundModel[0]->rows; i++ )
  {
    bgModel_0P = backgroundModel[0]->ptr<Vec2f>( i );
    bgModel_1P = backgroundModel[1]->ptr<Vec2f>( i );
    for ( int j = 0; j < backgroundModel[0]->cols; j++ )
    {
      // scan background model vector from T1 to Tk
      for ( size_t z = 1; z < backgroundModel.size(); z++ )
      {
        Vec2f bgModel_zP = backgroundModel[z]->ptr<Vec2f>( i )[j];
        // Fill vector of pairs
        pixelTemplates[z - 1].first = bgModel_zP[0];  // Current B (background value)
        pixelTemplates[z - 1].second = bgModel_zP[1];  // Current C (efficacy value)
      }

      //SORT template from T1 to Tk
      std::sort( pixelTemplates.begin(), pixelTemplates.end(), pairCompare );

      //REFILL CURRENT MODEL ( T1...Tk)
      for ( size_t zz = 1; zz < backgroundModel.size(); zz++ )
      {
        backgroundModel[zz]->ptr<Vec2f>( i )[j][0] = pixelTemplates[zz - 1].first;  // Replace previous B with new ordered B value
        backgroundModel[zz]->ptr<Vec2f>( i )[j][1] = pixelTemplates[zz - 1].second;  // Replace previous C with new ordered C value
      }

      // SORT Template T0 and T1
      if( bgModel_1P[j][1] > thetaL && bgModel_0P[j][1] < thetaL )
      {

        // swap B value of T0 with B value of T1 (for current model)
        swap( bgModel_0P[j][0], bgModel_1P[j][0] );

        // set new C0 value for current model)
        swap( bgModel_0P[j][1], bgModel_1P[j][1] );
        bgModel_0P[j][1] = (float) gamma * thetaL;

      }

    }
  }

  return true;
}
bool MotionSaliencyBinWangApr2014::templateReplacement( const Mat& finalBFMask, const Mat& image )
{
  std::vector<Mat> temp;
  split( *backgroundModel[0], temp );

//if at least the first template is activated / initialized for all pixels
  if( countNonZero( temp[1] ) <= ( temp[1].cols * temp[1].rows ) / 2 )
  {
    thetaA = 50;
    thetaL = 150;
    neighborhoodCheck = false;

  }
  else
  {
    thetaA = thetaA_VAL;
    thetaL = thetaL_VAL;
    neighborhoodCheck = true;
  }

  int roiSize = 3;  // FIXED ROI SIZE, not change until you first appropriately adjust the following controls in the EVALUATION section!
  int countNonZeroElements = 0;
  std::vector<Mat> mv;
  Mat replicateCurrentBAMat( roiSize, roiSize, CV_32F );
  Mat backgroundModelROI( roiSize, roiSize, CV_32F );
  Mat diffResult( roiSize, roiSize, CV_32F );

// Scan all pixels of finalBFMask and all pixels of others models (the dimension are the same)
  const float* finalBFMaskP;
  Vec2f* pbgP;
  const uchar* imageP;
  float* epslonP;
  for ( int i = 0; i < finalBFMask.rows; i++ )
  {
    finalBFMaskP = finalBFMask.ptr<float>( i );
    pbgP = potentialBackground.ptr<Vec2f>( i );
    imageP = image.ptr<uchar>( i );
    epslonP = epslonPixelsValue.ptr<float>( i );
    for ( int j = 0; j < finalBFMask.cols; j++ )
    {
      /////////////////// MAINTENANCE of potentialBackground model ///////////////////
      if( finalBFMaskP[j] == 1 )  // i.e. the corresponding frame pixel has been market as foreground
      {
        /* For the pixels with CA= 0, if the current frame pixel has been classified as foreground, its value
         * will be loaded into BA and CA will be set to 1*/
        if( pbgP[j][1] == 0 )
        {
          pbgP[j][0] = (float) imageP[j];
          pbgP[j][1] = 1;
        }

        /*the distance between this pixel value and BA is calculated, and if this distance is smaller than
         the decision threshold epslon, then CA is increased by 1, otherwise is decreased by 1*/
        else if( abs( (float) imageP[j] - pbgP[j][0] ) < epslonP[j] )
        {
          pbgP[j][1] += 1;
        }
        else
        {
          pbgP[j][1] -= 1;
        }
        /*}*/  /////////////////// END of potentialBackground model MAINTENANCE///////////////////
        /////////////////// EVALUATION of potentialBackground values ///////////////////
        if( pbgP[j][1] > thetaA )
        {
          if( neighborhoodCheck )
          {
            // replicate currentBA value
            replicateCurrentBAMat.setTo( pbgP[j][0] );

            for ( size_t z = 0; z < backgroundModel.size(); z++ )
            {
              // Neighborhood of current pixel in the current background model template.
              // The ROI is centered in the pixel coordinates

              if( i > 0 && j > 0 && i < ( backgroundModel[z]->rows - 1 ) && j < ( backgroundModel[z]->cols - 1 ) )
              {
                split( *backgroundModel[z], mv );
                backgroundModelROI = mv[0]( Rect( j - (int) floor((float) roiSize / 2 ), i - (int) floor((float) roiSize / 2 ), roiSize, roiSize ) );
              }
              else if( i == 0 && j == 0 )  // upper leftt
              {
                split( *backgroundModel[z], mv );
                backgroundModelROI = mv[0]( Rect( j, i, (int) ceil((float) roiSize / 2 ), (int) ceil((float) roiSize / 2 ) ) );
              }
              else if( j == 0 && i > 0 && i < ( backgroundModel[z]->rows - 1 ) )  // middle left
              {
                split( *backgroundModel[z], mv );
                backgroundModelROI = mv[0]( Rect( j, i - (int) floor((float) roiSize / 2 ), (int) ceil((float) roiSize / 2 ), roiSize ) );
              }
              else if( i == ( backgroundModel[z]->rows - 1 ) && j == 0 )  //down left
              {
                split( *backgroundModel[z], mv );
                backgroundModelROI = mv[0]( Rect( j, i - (int) floor((float) roiSize / 2 ), (int) ceil((float) roiSize / 2 ), (int) ceil((float) roiSize / 2 ) ) );
              }
              else if( i == 0 && j > 0 && j < ( backgroundModel[z]->cols - 1 ) )  // upper - middle
              {
                split( *backgroundModel[z], mv );
                backgroundModelROI = mv[0]( Rect( ( j - (int) floor((float) roiSize / 2 ) ), i, roiSize, (int) ceil((float) roiSize / 2 ) ) );
              }
              else if( i == ( backgroundModel[z]->rows - 1 ) && j > 0 && j < ( backgroundModel[z]->cols - 1 ) )  //down middle
              {
                split( *backgroundModel[z], mv );
                backgroundModelROI = mv[0](
                    Rect( j - (int) floor((float) roiSize / 2 ), i - (int) floor((float) roiSize / 2 ), roiSize, (int) ceil((float) roiSize / 2 ) ) );
              }
              else if( i == 0 && j == ( backgroundModel[z]->cols - 1 ) )  // upper right
              {
                split( *backgroundModel[z], mv );
                backgroundModelROI = mv[0]( Rect( j - (int) floor((float) roiSize / 2 ), i, (int) ceil((float) roiSize / 2 ), (int) ceil((float) roiSize / 2 ) ) );
              }
              else if( j == ( backgroundModel[z]->cols - 1 ) && i > 0 && i < ( backgroundModel[z]->rows - 1 ) )  // middle - right
              {
                split( *backgroundModel[z], mv );
                backgroundModelROI = mv[0](
                    Rect( j - (int) floor((float) roiSize / 2 ), i - (int) floor((float) roiSize / 2 ), (int) ceil((float) roiSize / 2 ), roiSize ) );
              }
              else if( i == ( backgroundModel[z]->rows - 1 ) && j == ( backgroundModel[z]->cols - 1 ) )  // down right
              {
                split( *backgroundModel[z], mv );
                backgroundModelROI = mv[0](
                    Rect( j - (int) floor((float) roiSize / 2 ), i - (int) floor((float) roiSize / 2 ), (int) ceil((float) roiSize / 2 ), (int) ceil((float) roiSize / 2 ) ) );
              }

              /* Check if the value of current pixel BA in potentialBackground model is already contained in at least one of its neighbors'
               * background model
               */
              resize( replicateCurrentBAMat, replicateCurrentBAMat, Size( backgroundModelROI.cols, backgroundModelROI.rows ), 0, 0, INTER_LINEAR );
              resize( diffResult, diffResult, Size( backgroundModelROI.cols, backgroundModelROI.rows ), 0, 0, INTER_LINEAR );

              absdiff( replicateCurrentBAMat, backgroundModelROI, diffResult );
              threshold( diffResult, diffResult, epslonP[j], 255, THRESH_BINARY_INV );
              countNonZeroElements = countNonZero( diffResult );

              if( countNonZeroElements > 0 )
              {
                /////////////////// REPLACEMENT of backgroundModel template ///////////////////
                //replace TA with current TK
                backgroundModel[backgroundModel.size() - 1]->at<Vec2f>( i, j ) = potentialBackground.at<Vec2f>( i, j );
                potentialBackground.at<Vec2f>( i, j )[0] = std::numeric_limits<float>::quiet_NaN();
                potentialBackground.at<Vec2f>( i, j )[1] = 0;

                break;
              }
            }  // end for backgroundModel size
          }
          else
          {
            backgroundModel[backgroundModel.size() - 1]->at<Vec2f>( i, j ) = potentialBackground.at<Vec2f>( i, j );
            potentialBackground.at<Vec2f>( i, j )[0] = std::numeric_limits<float>::quiet_NaN();
            potentialBackground.at<Vec2f>( i, j )[1] = 0;
          }
        }  // close if of EVALUATION
      }  // end of  if( finalBFMask.at<uchar>( i, j ) == 1 )  // i.e. the corresponding frame pixel has been market as foreground

    }  // end of second for
  }  // end of first for

  return true;
}

bool MotionSaliencyBinWangApr2014::computeSaliencyImpl( InputArray image, OutputArray saliencyMap )
{
  Mat highResBFMask;
  Mat lowResBFMask;
  Mat not_lowResBFMask;
  Mat noisePixelsMask;

  fullResolutionDetection( image.getMat(), highResBFMask );
  lowResolutionDetection( image.getMat(), lowResBFMask );


// Compute the final background-foreground mask. One pixel is marked as foreground if and only if it is
// foreground in both masks (full and low)
  bitwise_and( highResBFMask, lowResBFMask, saliencyMap );

  templateOrdering();
  templateReplacement( saliencyMap.getMat(), image.getMat() );
  templateOrdering();

  return true;
}

}  // namespace saliency
}  // namespace cv
