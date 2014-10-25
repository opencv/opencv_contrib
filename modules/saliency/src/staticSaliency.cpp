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

#include "precomp.hpp"

namespace cv
{
namespace saliency
{

/**
 * StaticSaliency
 */

bool StaticSaliency::computeBinaryMap( const Mat& saliencyMap, Mat& BinaryMap )
{

  Mat labels = Mat::zeros( saliencyMap.rows * saliencyMap.cols, 1, 1 );
  Mat samples = Mat_<float>( saliencyMap.rows * saliencyMap.cols, 1 );
  Mat centers;
  TermCriteria terminationCriteria;
  terminationCriteria.epsilon = 0.2;
  terminationCriteria.maxCount = 1000;
  terminationCriteria.type = TermCriteria::COUNT + TermCriteria::EPS;

  int elemCounter = 0;
  for ( int i = 0; i < saliencyMap.rows; i++ )
  {
    for ( int j = 0; j < saliencyMap.cols; j++ )
    {
      samples.at<float>( elemCounter, 0 ) = saliencyMap.at<float>( i, j );
      elemCounter++;
    }
  }

  kmeans( samples, 5, labels, terminationCriteria, 5, KMEANS_RANDOM_CENTERS, centers );

  Mat outputMat = Mat_<float>( saliencyMap.size() );
  int intCounter = 0;
  for ( int x = 0; x < saliencyMap.rows; x++ )
  {
    for ( int y = 0; y < saliencyMap.cols; y++ )
    {
      outputMat.at<float>( x, y ) = centers.at<float>( labels.at<int>( intCounter, 0 ), 0 );
      intCounter++;
    }

  }

  //Convert
  outputMat = outputMat * 255;
  outputMat.convertTo( outputMat, CV_8U );

  // adaptative thresholding using Otsu's method, to make saliency map binary
  threshold( outputMat, BinaryMap, 0, 255, THRESH_BINARY | THRESH_OTSU );

  return true;

}

}/* namespace saliency */
}/* namespace cv */
