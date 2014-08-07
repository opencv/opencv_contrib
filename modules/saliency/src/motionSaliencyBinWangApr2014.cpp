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
  thetaL = 50;  // T0, T1 swap threshold
  thetaA = 200;
  gamma = 3;
  neighborhoodCheck = true;

  className = "BinWangApr2014";
}

bool MotionSaliencyBinWangApr2014::init()
{

  epslonPixelsValue = Mat( imgSize->height, imgSize->width, CV_32F );
  potentialBackground = Mat( imgSize->height, imgSize->width, CV_32FC2 );
  //backgroundModel = std::vector<Mat>( K + 1, Mat::zeros( imgSize->height, imgSize->width, CV_32FC2 ) );
  //TODO set to nan
  potentialBackground.setTo( 0 );

  backgroundModel.resize(K+1);
  //TODO set to nan
  for ( int i = 0; i < K + 1; i++ )
  {
    Mat* tmpm = new Mat;
    tmpm->create(imgSize->height, imgSize->width, CV_32FC2);
    tmpm->setTo(0);
    Ptr<Mat> tmp = Ptr<Mat>( tmpm );
    backgroundModel[i] = tmp;
  }

  epslonPixelsValue.setTo( 70 );  // Median of range [18, 80] advised in reference paper.
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
bool MotionSaliencyBinWangApr2014::fullResolutionDetection( const Mat& image2, Mat& highResBFMask )
{
  Mat image = image2.clone();

  float currentPixelValue;
  float currentEpslonValue;
  bool backgFlag = false;

  Mat test( image.rows, image.cols, CV_8U );
  Mat test1( image.rows, image.cols, CV_8U );
  test.setTo( 255 );
  for ( int i = 0; i < test.rows; i++ )
  {
    for ( int j = 0; j < test.cols; j++ )
    {
      if( backgroundModel[0]->at<Vec2f>( i, j )[1] == 0 )
      {
        test.at<uchar>( i, j ) = 0;
      }
      test1.at<uchar>( i, j ) = (int) backgroundModel[0]->at<Vec2f>( i, j )[0];
    }
  }
  imshow( "test_T0c", test );
  imshow( "test_T0b", test1 );

  // Initially, all pixels are considered as foreground and then we evaluate with the background model
  highResBFMask.create( image.rows, image.cols, CV_32F );
  highResBFMask.setTo( 1 );

  uchar* pImage;
  float* pEpslon;
  float* pMask;

  int countDec = 0;

  // Scan all pixels of image
  for ( int i = 0; i < image.rows; i++ )
  {

    pImage = image.ptr<uchar>( i );
    pEpslon = epslonPixelsValue.ptr<float>( i );
    pMask = highResBFMask.ptr<float>( i );
    for ( int j = 0; j < image.cols; j++ )
    {

      backgFlag = false;
      // TODO replace "at" with more efficient matrix access
      //currentPixelValue = image.at<uchar>( i, j );
      //currentEpslonValue = epslonPixelsValue.at<float>( i, j );
      currentPixelValue = pImage[j];
      currentEpslonValue = pEpslon[j];

      if( i == 50 && j == 50 )
        cout << "currentPixelValue :" << currentPixelValue << endl << "currentEpslonValue :" << currentEpslonValue << endl;

      int counter = 0;
      for ( size_t z = 0; z < backgroundModel.size(); z++ )
      {
        counter += backgroundModel.at(z)->at<Vec2f>( i, j )[1];
      }

      if( counter != 0 )  //if at least the first template is activated / initialized
      {

        // scan background model vector
        for ( size_t z = 0; z < backgroundModel.size(); z++ )
        {
          float currentB;
          float currentC;
          // TODO replace "at" with more efficient matrix access
          currentB = (backgroundModel.at(z)->at<Vec2f>( i, j )[0]);
          currentC = (backgroundModel.at(z)->at<Vec2f>( i, j )[1]);

          if( i == 50 && j == 50 )
          {
            cout << "zeta:" << z << " currentB :" << currentB << endl << "currentC :" << currentC << endl;
          }
          //continue;

          if( currentC > 0 )  //The current template is active
          {
            //cout<< "DIFFERENCE: "<<abs( currentPixelValue -  ( *currentB ) )<<endl;
            // If there is a match with a current background template
            if( abs( currentPixelValue - ( currentB ) ) < currentEpslonValue && !backgFlag )
            {
              // The correspondence pixel in the  BF mask is set as background ( 0 value)
              //highResBFMask.at<uchar>( i, j ) = 0;
              pMask[j] = 0;
              //if( ( *currentC < L0 && z == 0 ) || ( *currentC < L1 && z == 1 ) || ( z > 1 ) )
              (backgroundModel.at(z)->at<Vec2f>( i, j )[1]) = (backgroundModel.at(z)->at<Vec2f>( i, j )[1]) + 1;  // increment the efficacy of this template

              (backgroundModel.at(z)->at<Vec2f>( i, j )[0]) = ( ( 1 - alpha ) * ( currentB ) ) + ( alpha * currentPixelValue );  // Update the template value
              backgFlag = true;
              //break;
            }
            else
            {
              if( z == 0 )
                countDec++;

              (backgroundModel.at(z)->at<Vec2f>( i, j )[1]) = (backgroundModel.at(z)->at<Vec2f>( i, j )[1]) - 1;  // decrement the efficacy of this template
            }

          }

          if( i == 50 && j == 50 )
          {
            cout << "DOPO IF: " << endl;
            cout << "zeta:" << z << " currentB_A :" << &currentB << endl << "currentC_A :" << &currentC<<endl;
            cout << "zeta:" << z << " currentB :" <<  (backgroundModel.at(z)->at<Vec2f>( i, j )[0]) << endl << "currentC :" <<  (backgroundModel.at(z)->at<Vec2f>( i, j )[1]) << endl<<endl;
          }
        }  // end "for" cicle of template vector

      }
      else
      {
        pMask[j] = 1;  //if the model of the current pixel is not yet initialized, we mark the pixels as foreground
      }

    }
  }  // end "for" cicle of all image's pixels

  //cout<<" STATISTICA :"<<countDec<<"/"<< image.rows*image.cols<< " = "<<(float)countDec/(float)(image.rows*image.cols)*100<<" %"<<endl;

  return true;
}

bool MotionSaliencyBinWangApr2014::lowResolutionDetection( const Mat& image, Mat& lowResBFMask )
{
  std::vector<Mat> mv;
  split( *backgroundModel[0], mv );

  //if at least the first template is activated / initialized for all pixels
  if( countNonZero( mv.at( 1 ) ) > ( mv.at( 1 ).cols * mv.at( 1 ).rows ) / 2 )
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
        currentPixelValue = mean( roiImage ).val[0];
        currentEpslonValue = mean( roiEpslon ).val[0];

        // scan background model vector
        for ( int z = 0; z < N_DS; z++ )
        {
          // Select the current template 2 channel matrix, select ROI and compute the mean for each channel separately
          Mat roiTemplate = (*(backgroundModel[z]))( roi );
          Scalar templateMean = mean( roiTemplate );
          currentB = templateMean[0];
          currentC = templateMean[1];

          if( currentC > 0 )  //The current template is active
          {
            // If there is a match with a current background template
            if( abs( currentPixelValue - ( currentB ) ) < currentEpslonValue )
            {
              // The correspondence pixel in the  BF mask is set as background ( 0 value)
              // TODO replace "at" with more efficient matrix access
              //lowResBFMask.at<uchar>( i, j ) = 0;
              //lowResBFMask.setTo( 0, ROIMask );
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
  for ( int i = 0; i < backgroundModel[0]->rows; i++ )
  {
    for ( int j = 0; j < backgroundModel[0]->cols; j++ )
    {
      // scan background model vector from T1 to Tk
      for ( size_t z = 1; z < backgroundModel.size(); z++ )
      {
        // Fill vector of pairs
        pixelTemplates[z - 1].first = backgroundModel[z]->at<Vec2f>( i, j )[0];  // Current B (background value)
        pixelTemplates[z - 1].second = backgroundModel[z]->at<Vec2f>( i, j )[1];  // Current C (efficacy value)
      }

      //SORT template from T1 to Tk
      std::sort( pixelTemplates.begin(), pixelTemplates.end(), pairCompare );

      //REFILL CURRENT MODEL ( T1...Tk)
      for ( size_t zz = 1; zz < backgroundModel.size(); zz++ )
      {
        backgroundModel[zz]->at<Vec2f>( i, j )[0] = pixelTemplates[zz - 1].first;  // Replace previous B with new ordered B value
        backgroundModel[zz]->at<Vec2f>( i, j )[1] = pixelTemplates[zz - 1].second;  // Replace previous C with new ordered C value
      }

      // SORT Template T0 and T1
      if( backgroundModel[1]->at<Vec2f>( i, j )[1] > thetaL && backgroundModel[0]->at<Vec2f>( i, j )[1] < thetaL )
      {

        // swap B value of T0 with B value of T1 (for current model)
        temp = backgroundModel[0]->at<Vec2f>( i, j )[0];
        backgroundModel[0]->at<Vec2f>( i, j )[0] = backgroundModel[1]->at<Vec2f>( i, j )[0];
        backgroundModel[1]->at<Vec2f>( i, j )[0] = temp;

        // set new C0 value for current model)
        temp = backgroundModel[0]->at<Vec2f>( i, j )[1];
        backgroundModel[0]->at<Vec2f>( i, j )[1] = gamma * thetaL;
        backgroundModel[1]->at<Vec2f>( i, j )[1] = temp;

      }

    }
  }

  return true;
}
bool MotionSaliencyBinWangApr2014::templateReplacement( const Mat& finalBFMask, const Mat& image )
{
  Mat test( image.rows, image.cols, CV_8U );
  for ( int i = 0; i < test.rows; i++ )
  {
    for ( int j = 0; j < test.cols; j++ )
    {

      test.at<uchar>( i, j ) = (int) potentialBackground.at<Vec2f>( i, j )[0];
    }
  }
  imshow( "test_BA", test );

  std::vector<Mat> temp;
  split( *backgroundModel[0], temp );

//if at least the first template is activated / initialized for all pixels
  if( countNonZero( temp.at( 1 ) ) <= ( temp.at( 1 ).cols * temp.at( 1 ).rows ) / 2 )
  {
    thetaA = 2;
    neighborhoodCheck = false;

  }
  else
  {
    thetaA = 200;
    neighborhoodCheck = false;
  }

  float roiSize = 3;  // FIXED ROI SIZE, not change until you first appropriately adjust the following controls in the EVALUATION section!
  int countNonZeroElements = 0;
  std::vector<Mat> mv;
  Mat replicateCurrentBAMat( roiSize, roiSize, CV_32FC1 );
  Mat backgroundModelROI( roiSize, roiSize, CV_32FC1 );
  Mat diffResult( roiSize, roiSize, CV_32FC1 );

// Scan all pixels of finalBFMask and all pixels of others models (the dimension are the same)
  for ( int i = 0; i < finalBFMask.rows; i++ )
  {
    for ( int j = 0; j < finalBFMask.cols; j++ )
    {
      /////////////////// MAINTENANCE of potentialBackground model ///////////////////
      if( finalBFMask.at<float>( i, j ) == 1 )  // i.e. the corresponding frame pixel has been market as foreground
      {
        /* For the pixels with CA= 0, if the current frame pixel has been classified as foreground, its value
         * will be loaded into BA and CA will be set to 1*/
        if( potentialBackground.at<Vec2f>( i, j )[1] == 0 )
        {
          potentialBackground.at<Vec2f>( i, j )[0] = (float) image.at<uchar>( i, j );
          potentialBackground.at<Vec2f>( i, j )[1] = 1;
        }

        /*the distance between this pixel value and BA is calculated, and if this distance is smaller than
         the decision threshold epslon, then CA is increased by 1, otherwise is decreased by 1*/
        else if( abs( (float) image.at<uchar>( i, j ) - potentialBackground.at<Vec2f>( i, j )[0] ) < epslonPixelsValue.at<float>( i, j ) )
        {
          potentialBackground.at<Vec2f>( i, j )[1] += 1;
        }
        else
        {
          potentialBackground.at<Vec2f>( i, j )[1] -= 1;
        }
        /*}*/  /////////////////// END of potentialBackground model MAINTENANCE///////////////////
        /////////////////// EVALUATION of potentialBackground values ///////////////////
        if( potentialBackground.at<Vec2f>( i, j )[1] > thetaA )
        {
          if( neighborhoodCheck )
          {
            // replicate currentBA value
            replicateCurrentBAMat.setTo( potentialBackground.at<Vec2f>( i, j )[0] );

            for ( size_t z = 0; z < backgroundModel.size(); z++ )
            {
              // Neighborhood of current pixel in the current background model template.
              // The ROI is centered in the pixel coordinates

              /*if( ( i - floor( roiSize / 2 ) >= 0 ) && ( j - floor( roiSize / 2 ) >= 0 )
               && ( i + floor( roiSize / 2 ) <= ( backgroundModel[z].rows - 1 ) )
               && ( j + floor( roiSize / 2 ) <= ( backgroundModel[z].cols - 1 ) ) ) */
              if( i > 0 && j > 0 && i < ( backgroundModel[z]->rows - 1 ) && j < ( backgroundModel[z]->cols - 1 ) )
              {
                split( *backgroundModel[z], mv );
                backgroundModelROI = mv.at( 0 )( Rect( j - floor( roiSize / 2 ), i - floor( roiSize / 2 ), roiSize, roiSize ) );
              }
              else if( i == 0 && j == 0 )  // upper left
              {
                split( *backgroundModel[z], mv );
                backgroundModelROI = mv.at( 0 )( Rect( j, i, ceil( roiSize / 2 ), ceil( roiSize / 2 ) ) );
              }
              else if( j == 0 && i > 0 && i < ( backgroundModel[z]->rows - 1 ) )  // middle left
              {
                split( *backgroundModel[z], mv );
                backgroundModelROI = mv.at( 0 )( Rect( j, i - floor( roiSize / 2 ), ceil( roiSize / 2 ), roiSize ) );
              }
              else if( i == ( backgroundModel[z]->rows - 1 ) && j == 0 )  //down left
              {
                split( *backgroundModel[z], mv );
                backgroundModelROI = mv.at( 0 )( Rect( j, i - floor( roiSize / 2 ), ceil( roiSize / 2 ), ceil( roiSize / 2 ) ) );
              }
              else if( i == 0 && j > 0 && j < ( backgroundModel[z]->cols - 1 ) )  // upper - middle
              {
                split( *backgroundModel[z], mv );
                backgroundModelROI = mv.at( 0 )( Rect( ( j - floor( roiSize / 2 ) ), i, roiSize, ceil( roiSize / 2 ) ) );
              }
              else if( i == ( backgroundModel[z]->rows - 1 ) && j > 0 && j < ( backgroundModel[z]->cols - 1 ) )  //down middle
              {
                split( *backgroundModel[z], mv );
                backgroundModelROI = mv.at( 0 )( Rect( j - floor( roiSize / 2 ), i - floor( roiSize / 2 ), roiSize, ceil( roiSize / 2 ) ) );
              }
              else if( i == 0 && j == ( backgroundModel[z]->cols - 1 ) )  // upper right
              {
                split( *backgroundModel[z], mv );
                backgroundModelROI = mv.at( 0 )( Rect( j - floor( roiSize / 2 ), i, ceil( roiSize / 2 ), ceil( roiSize / 2 ) ) );
              }
              else if( j == ( backgroundModel[z]->cols - 1 ) && i > 0 && i < ( backgroundModel[z]->rows - 1 ) )  // middle - right
              {
                split( *backgroundModel[z], mv );
                backgroundModelROI = mv.at( 0 )( Rect( j - floor( roiSize / 2 ), i - floor( roiSize / 2 ), ceil( roiSize / 2 ), roiSize ) );
              }
              else if( i == ( backgroundModel[z]->rows - 1 ) && j == ( backgroundModel[z]->cols - 1 ) )  // down right
              {
                split( *backgroundModel[z], mv );
                backgroundModelROI = mv.at( 0 )(
                    Rect( j - floor( roiSize / 2 ), i - floor( roiSize / 2 ), ceil( roiSize / 2 ), ceil( roiSize / 2 ) ) );
              }

              /* Check if the value of current pixel BA in potentialBackground model is already contained in at least one of its neighbors'
               * background model
               */
              resize( replicateCurrentBAMat, replicateCurrentBAMat, Size( backgroundModelROI.cols, backgroundModelROI.rows ), 0, 0, INTER_LINEAR );
              resize( diffResult, diffResult, Size( backgroundModelROI.cols, backgroundModelROI.rows ), 0, 0, INTER_LINEAR );

              absdiff( replicateCurrentBAMat, backgroundModelROI, diffResult );
              threshold( diffResult, diffResult, epslonPixelsValue.at<float>( i, j ), 255, THRESH_BINARY_INV );
              countNonZeroElements = countNonZero( diffResult );

              if( countNonZeroElements > 0 )
              {
                /////////////////// REPLACEMENT of backgroundModel template ///////////////////
                //replace TA with current TK

                //TODO CHECK BACKGROUND MODEL COUNTER ASSIGNEMENT
                //backgroundModel[backgroundModel.size()-1].at<Vec2f>( i, j )[0]=potentialBackground.at<Vec2f>( i, j )[0];
                //backgroundModel[backgroundModel.size()-1].at<Vec2f>( i, j )[1]= potentialBackground.at<Vec2f>( i, j )[1];
                backgroundModel[backgroundModel.size() - 1]->at<Vec2f>( i, j ) = potentialBackground.at<Vec2f>( i, j );
                potentialBackground.at<Vec2f>( i, j )[0] = 0;
                potentialBackground.at<Vec2f>( i, j )[1] = 0;

                break;
              }
            }  // end for backgroundModel size
          }
          else
          {
            ((backgroundModel.at(backgroundModel.size() - 1))->at<Vec2f>( i, j ))[0] = potentialBackground.at<Vec2f>( i, j )[0];
            ((backgroundModel.at(backgroundModel.size() - 1))->at<Vec2f>( i, j ))[1] = potentialBackground.at<Vec2f>( i, j )[1];
            potentialBackground.at<Vec2f>( i, j )[0] = 0;
            potentialBackground.at<Vec2f>( i, j )[1] = 0;
            //((backgroundModel.at(0))->at<Vec2f>( i, j ))[1] = 3;
          }
        }  // close if of EVALUATION
      }  // end of  if( finalBFMask.at<uchar>( i, j ) == 1 )  // i.e. the corresponding frame pixel has been market as foreground

    }  // end of second for
  }  // end of first for

  return true;
}

bool MotionSaliencyBinWangApr2014::computeSaliencyImpl( const InputArray image, OutputArray saliencyMap )
{

  Mat highResBFMask;
  Mat lowResBFMask;
  Mat not_lowResBFMask;
  Mat noisePixelsMask;

  /*Mat t( image.getMat().rows, image.getMat().cols, CV_32FC2 );
   t.setTo( 50 );
   backgroundModel.at( 0 ) = t; */

  std::ofstream ofs4;
  ofs4.open( "TEMPLATE_0_B.txt", std::ofstream::out );

  for ( int i = 0; i < backgroundModel[0]->rows; i++ )
  {
    for ( int j = 0; j < backgroundModel[0]->cols; j++ )
    {
      //highResBFMask.at<int>( i, j ) = i + j;
      stringstream str;
      str << backgroundModel[0]->at<Vec2f>( i, j )[0] << " ";
      ofs4 << str.str();
    }
    stringstream str2;
    str2 << "\n";
    ofs4 << str2.str();
  }
  ofs4.close();

  std::ofstream ofs5;
  ofs5.open( "TEMPLATE_0_C.txt", std::ofstream::out );

  for ( int i = 0; i < backgroundModel[0]->rows; i++ )
  {
    for ( int j = 0; j < backgroundModel[0]->cols; j++ )
    {
      //highResBFMask.at<int>( i, j ) = i + j;
      stringstream str;
      str << backgroundModel[0]->at<Vec2f>( i, j )[1] << " ";
      ofs5 << str.str();
    }
    stringstream str2;
    str2 << "\n";
    ofs5 << str2.str();
  }
  ofs5.close();

  fullResolutionDetection( image.getMat(), highResBFMask );
  lowResolutionDetection( image.getMat(), lowResBFMask );

  imshow( "highResBFMask", highResBFMask * 255 );
  imshow( "lowResBFMask", lowResBFMask * 255 );

// Compute the final background-foreground mask. One pixel is marked as foreground if and only if it is
// foreground in both masks (full and low)
  bitwise_and( highResBFMask, lowResBFMask, saliencyMap );

// Detect the noise pixels (i.e. for a given pixel, fullRes(pixel) = foreground and lowRes(pixel)= background)
  //bitwise_not( lowResBFMask, not_lowResBFMask );
  //bitwise_and( highResBFMask, not_lowResBFMask, noisePixelsMask );

  templateOrdering();
  templateReplacement( saliencyMap.getMat(), image.getMat() );
  //templateReplacement( highResBFMask, image.getMat() );
  templateOrdering();

  //highResBFMask.copyTo( saliencyMap );

  std::ofstream ofs;
  ofs.open( "highResBFMask.txt", std::ofstream::out );

  for ( int i = 0; i < highResBFMask.rows; i++ )
  {
    for ( int j = 0; j < highResBFMask.cols; j++ )
    {
      //highResBFMask.at<int>( i, j ) = i + j;
      stringstream str;
      str << highResBFMask.at<float>( i, j ) << " ";
      ofs << str.str();
    }
    stringstream str2;
    str2 << "\n";
    ofs << str2.str();
  }
  ofs.close();

  std::ofstream ofs2;
  ofs2.open( "lowResBFMask.txt", std::ofstream::out );

  for ( int i = 0; i < lowResBFMask.rows; i++ )
  {
    for ( int j = 0; j < lowResBFMask.cols; j++ )
    {
      stringstream str;
      str << lowResBFMask.at<float>( i, j ) << " ";
      ofs2 << str.str();
    }
    stringstream str2;
    str2 << "\n";
    ofs2 << str2.str();
  }
  ofs2.close();

  std::ofstream ofs3;
  ofs3.open( "SALMAP.txt", std::ofstream::out );

  for ( int i = 0; i < saliencyMap.getMat().rows; i++ )
  {
    for ( int j = 0; j < saliencyMap.getMat().cols; j++ )
    {
      stringstream str;
      str << saliencyMap.getMat().at<float>( i, j ) << " ";
      ofs3 << str.str();
    }
    stringstream str2;
    str2 << "\n";
    ofs3 << str2.str();
  }
  ofs3.close();

  return true;
}

}  // namespace cv
