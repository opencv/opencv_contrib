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

#define thetaA_VAL 200
#define thetaL_VAL 250
#define epslonGeneric 20

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

  Ainc = 6;  // Activity Incrementation;
  Bmax = 80;  // Upper-bound value for pixel activity
  Bth = 20;  //70;  // Max activity threshold
  Binc = 15;  //50;
  Bdec = 5;  //20;  // Threshold for pixel-level decision threshold (epslon) adaptation
  deltaINC = 20;
  deltaDEC = 0.125;  // Increment-decrement value for epslon adaptation
  epslonMIN = 18;
  epslonMAX = 80;

  className = "BinWangApr2014";
}

bool MotionSaliencyBinWangApr2014::init()
{
  activityControlFlag = false;
  Size imgSize( imageWidth, imageHeight );
  epslonPixelsValue = Mat( imgSize.height, imgSize.width, CV_32F, Scalar( epslonGeneric ) );
  potentialBackground = Mat( imgSize.height, imgSize.width, CV_8UC2, Scalar( 0, 0 ) );
  backgroundModel.resize( K + 1 );

  for ( int i = 0; i < K + 1; i++ )
  {
    Mat* tmpm = new Mat;
    tmpm->create( imgSize.height, imgSize.width, CV_32FC2 );
    tmpm->setTo( Scalar( std::numeric_limits<float>::quiet_NaN(), 0 ) );
    Ptr<Mat> tmp = Ptr<Mat>( tmpm );
    backgroundModel[i] = tmp;
  }

  noisePixelMask.create( imgSize.height, imgSize.width, CV_8U );
  noisePixelMask.setTo( Scalar( 0 ) );
  activityPixelsValue.create( imgSize.height, imgSize.width, CV_8U );
  activityPixelsValue.setTo( Scalar( 0 ) );

  return true;

}

MotionSaliencyBinWangApr2014::~MotionSaliencyBinWangApr2014()
{

}

namespace
{
static int countActiveTemplates0(const Mat& bg0)
{
  int active = 0;
  for (int i = 0; i < bg0.rows; ++i)
  {
    const Vec2f* row = bg0.ptr<Vec2f>(i);
    for (int j = 0; j < bg0.cols; ++j)
      active += (row[j][1] != 0.0f) ? 1 : 0;
  }
  return active;
}
} // anonymous namespace

// classification (and adaptation) functions
bool MotionSaliencyBinWangApr2014::fullResolutionDetection( const Mat& image2, Mat& highResBFMask )
{
  uchar currentPixelValue;
  float currentEpslonValue;
  bool backgFlag = false;

  // Initially, all pixels are considered as foreground and then we evaluate with the background model
  highResBFMask.create( image2.rows, image2.cols, CV_8U );
  highResBFMask.setTo( 1 );

  const uchar* pImage;
  float* pEpslon;
  uchar* pMask;

  // Scan all pixels of image
  for ( int i = 0; i < image2.rows; i++ )
  {

    pImage = image2.ptr<uchar>( i );
    pEpslon = epslonPixelsValue.ptr<float>( i );
    pMask = highResBFMask.ptr<uchar>( i );
    for ( int j = 0; j < image2.cols; j++ )
    {
      /*    Pixels with activity greater than Bth are eliminated from the detection result. In this way,
       continuously blinking noise-pixels will be eliminated from the detection results,
       preventing the generation of false positives.*/
      if( activityPixelsValue.at<uchar>( i, j ) < Bth )
      {
        backgFlag = false;
        currentPixelValue = pImage[j];
        currentEpslonValue = pEpslon[j];

        int counter = 0;
        for ( size_t z = 0; z < backgroundModel.size(); z++ )
        {

          counter += (int) backgroundModel[z]->ptr<Vec2f>( i )[j][1];
          if( counter != 0 )
            break;
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
      else
      {
        pMask[j] = 0;
      }

    }
  }  // end "for" cicle of all image's pixels

  return true;
}

bool MotionSaliencyBinWangApr2014::lowResolutionDetection( const Mat& image, Mat& lowResBFMask )
{
  const Mat& bg0 = *backgroundModel[0];

  //if at least the first template is activated / initialized for all pixels
  if( countActiveTemplates0( bg0 ) > ( bg0.cols * bg0.rows ) / 2 )
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
    lowResBFMask.create( image.rows, image.cols, CV_8U );
    lowResBFMask.setTo( 1 );

    // Scan all the ROI of original matrices
    for ( int i = 0; i < (int)ceil( (float) image.rows / N ); i++ )
    {
      if( ( roi.y + ( N - 1 ) ) <= ( image.rows - 1 ) )
      {
        // Reset original ROI dimension
        roi = Rect( Point( roi.x, roi.y ), Size( N, N ) );
      }

      for ( int j = 0; j < (int)ceil( (float) image.cols / N ); j++ )
      {
        /* Pixels with activity greater than Bth are eliminated from the detection result. In this way,
         continuously blinking noise-pixels will be eliminated from the detection results,
         preventing the generation of false positives.*/
        if( activityPixelsValue.at<uchar>( i, j ) < Bth )
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
        else
        {
          // The correspondence pixel in the  BF mask is set as background ( 0 value)
          rectangle( lowResBFMask, roi, Scalar( 0 ), FILLED );
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
    lowResBFMask.create( image.rows, image.cols, CV_8U );
    lowResBFMask.setTo( 1 );
    return false;
  }

}

bool inline pairCompare( std::pair<float, float> t, std::pair<float, float> t_plusOne )
{

  return ( t.second > t_plusOne.second );

}

bool MotionSaliencyBinWangApr2014::templateOrdering()
{
  const int backGroundModelSize = (int)backgroundModel.size();
  if (backGroundModelSize <= 1)
    return true;

  const int rows = backgroundModel[0]->rows;
  const int cols = backgroundModel[0]->cols;
  const float thetaLf = static_cast<float>(thetaL);
  const float newC0 = static_cast<float>(gamma * thetaL);

  AutoBuffer<Vec2f*, 8> rowPtrsBuf(backGroundModelSize);
  Vec2f** rowPtrs = rowPtrsBuf.data();

  for (int r = 0; r < rows; ++r)
  {
    for (int z = 0; z < backGroundModelSize; ++z)
      rowPtrs[z] = backgroundModel[z]->ptr<Vec2f>(r);

    for (int c = 0; c < cols; ++c)
    {
      // Bubble sort: Template T1 .. TK
      for (int i = 1; i < backGroundModelSize - 1; ++i)
      {
        for (int j = i + 1; j < backGroundModelSize; ++j)
        {
          if (rowPtrs[j][c][1] > rowPtrs[i][c][1])
          {
            const Vec2f tmp = rowPtrs[i][c];
            rowPtrs[i][c] = rowPtrs[j][c];
            rowPtrs[j][c] = tmp;
          }
        }
      }

      // Sort Template T0 and T1
      if ((rowPtrs[1][c][1] > thetaLf) && (thetaLf > rowPtrs[0][c][1]))
      {
        const float tmpB = rowPtrs[0][c][0];
        rowPtrs[0][c][0] = rowPtrs[1][c][0];
        rowPtrs[1][c][0] = tmpB;
        rowPtrs[1][c][1] = rowPtrs[0][c][1];
        rowPtrs[0][c][1] = newC0;
      }
    }
  }

  return true;
}

bool MotionSaliencyBinWangApr2014::templateReplacement( const Mat& finalBFMask, const Mat& image )
{
  const Mat& bg0 = *backgroundModel[0];

  // if at least the first template is activated / initialized for half of pixels
  if( countActiveTemplates0( bg0 ) <= ( bg0.cols * bg0.rows ) / 2 )
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

  const int rows = finalBFMask.rows;
  const int cols = finalBFMask.cols;
  const size_t bgCount = backgroundModel.size();

  // Scan all pixels of finalBFMask and all pixels of other models (dimensions are identical)
  for ( int i = 0; i < rows; i++ )
  {
    const uchar* finalBFMaskP = finalBFMask.ptr<uchar>( i );
    Vec2b* pbgP = potentialBackground.ptr<Vec2b>( i );
    const uchar* imageP = image.ptr<uchar>( i );
    const float* epslonP = epslonPixelsValue.ptr<float>( i );

    for ( int j = 0; j < cols; j++ )
    {
      /////////////////// MAINTENANCE of potentialBackground model ///////////////////
      if( finalBFMaskP[j] == 1 )  // foreground pixel
      {
        if( pbgP[j][1] == 0 )
        {
          pbgP[j][0] = imageP[j];
          pbgP[j][1] = 1;
        }
        else if( std::abs( (float) imageP[j] - pbgP[j][0] ) < epslonP[j] )
        {
          pbgP[j][1] += 1;
        }
        else
        {
          pbgP[j][1] -= 1;
        }
        /////////////////// EVALUATION of potentialBackground values ///////////////////
        if( pbgP[j][1] > thetaA )
        {
          if( neighborhoodCheck )
          {
            const int curBA = static_cast<int>(pbgP[j][0]);
            const int epsFloor = cvFloor(epslonP[j]);
            const int r0 = std::max(0, i - 1);
            const int r1 = std::min(rows - 1, i + 1);
            const int c0 = std::max(0, j - 1);
            const int c1 = std::min(cols - 1, j + 1);

            for ( size_t z = 0; z < bgCount; z++ )
            {
              const Mat& bgZ = *backgroundModel[z];
              bool matchedNeighbor = false;
              for (int nr = r0; nr <= r1 && !matchedNeighbor; ++nr)
              {
                const Vec2f* bgRow = bgZ.ptr<Vec2f>(nr);
                for (int nc = c0; nc <= c1; ++nc)
                {
                  const int bgVal = static_cast<int>(saturate_cast<uchar>(bgRow[nc][0]));
                  if (std::abs(curBA - bgVal) <= epsFloor)
                  {
                    matchedNeighbor = true;
                    break;
                  }
                }
              }

              if( matchedNeighbor )
              {
                backgroundModel[bgCount - 1]->ptr<Vec2f>( i )[j] = Vec2f(pbgP[j][0], pbgP[j][1]);
                pbgP[j][0] = 0;
                pbgP[j][1] = 0;
                break;
              }
            }
          }
          else
          {
            backgroundModel[bgCount - 1]->ptr<Vec2f>( i )[j] = Vec2f(pbgP[j][0], pbgP[j][1]);
            pbgP[j][0] = 0;
            pbgP[j][1] = 0;
          }
        }
      }
    }
  }

  return true;
}

bool MotionSaliencyBinWangApr2014::activityControl( const Mat& current_noisePixelsMask )
{
  const int rows = activityPixelsValue.rows;
  const int cols = activityPixelsValue.cols;
  for (int i = 0; i < rows; ++i)
  {
    const uchar* curRow = current_noisePixelsMask.ptr<uchar>(i);
    uchar* prevNoiseRow = noisePixelMask.ptr<uchar>(i);
    uchar* actRow = activityPixelsValue.ptr<uchar>(i);
    for (int j = 0; j < cols; ++j)
    {
      const uchar curNoise = curRow[j];
      const bool discord = (prevNoiseRow[j] != 0) && (curNoise == 0);
      if (discord)
      {
        if (actRow[j] < Bmax)
          actRow[j] = static_cast<uchar>(actRow[j] + Ainc);
      }
      else
      {
        if (actRow[j] > 0)
          actRow[j] = static_cast<uchar>(actRow[j] - 1);
      }
      prevNoiseRow[j] = curNoise;
    }
  }

  return true;
}

bool MotionSaliencyBinWangApr2014::decisionThresholdAdaptation()
{
  for ( int i = 0; i < activityPixelsValue.rows; i++ )
  {
    const uchar* actRow = activityPixelsValue.ptr<uchar>( i );
    float* epsRow = epslonPixelsValue.ptr<float>( i );
    for ( int j = 0; j < activityPixelsValue.cols; j++ )
    {
      if( actRow[j] > Binc && ( epsRow[j] + deltaINC ) < epslonMAX )
      {
        epsRow[j] += deltaINC;
      }
      else if( actRow[j] < Bdec && ( epsRow[j] - deltaDEC ) > epslonMIN )
      {
        epsRow[j] -= deltaDEC;
      }
    }
  }

  return true;
}

bool MotionSaliencyBinWangApr2014::computeSaliencyImpl( InputArray image, OutputArray saliencyMap )
{
  CV_Assert(image.channels() == 1);

  Mat src = image.getMat();
  Mat highResBFMask;
  Mat lowResBFMask;

  fullResolutionDetection( src, highResBFMask );
  lowResolutionDetection( src, lowResBFMask );

  // Compute the final background-foreground mask. One pixel is marked as foreground if and only if it is
  // foreground in both masks (full and low)
  bitwise_and( highResBFMask, lowResBFMask, saliencyMap );

  if( activityControlFlag )
  {
    // Reuse lowResBFMask in-place to form current_noisePixelsMask = highResBFMask & (lowResBFMask == 0)
    threshold( lowResBFMask, lowResBFMask, 0.5, 1.0, THRESH_BINARY_INV );
    bitwise_and( highResBFMask, lowResBFMask, lowResBFMask );

    activityControl( lowResBFMask );
    decisionThresholdAdaptation();
  }

  highResBFMask.release();
  lowResBFMask.release();

  templateOrdering();
  templateReplacement( saliencyMap.getMat(), src );
  templateOrdering();

  activityControlFlag = true;
  return true;
}

}  // namespace saliency
}  // namespace cv
