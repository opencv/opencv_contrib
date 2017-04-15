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

#include "../precomp.hpp"
#include "CmShow.hpp"
#include "opencv2/core.hpp"
// #include <opencv2/highgui.hpp>

typedef std::pair<int, int> CostiIdx;

namespace cv
{
namespace saliency
{

Mat CmShow::HistBins( CMat& color3f, CMat& val, bool descendShow, CMat &with )
{
  // Prepare data
  int H = 300, spaceH = 6, barH = 10, n = color3f.cols;
  CV_Assert( color3f.size() == val.size() && color3f.rows == 1 );
  Mat binVal1i, binColor3b, width1i;
  if( with.size() == val.size() )
    with.convertTo( width1i, CV_32S, 400 / sum( with ).val[0] );  // Default shown width
  else
    width1i = Mat( 1, n, CV_32S, Scalar( 10 ) );  // Default bin width = 10
  int W = cvRound( sum( width1i ).val[0] );
  color3f.convertTo( binColor3b, CV_8UC3, 255 );
  double maxVal, minVal;
  minMaxLoc( val, &minVal, &maxVal );
  printf( "%g\n", H / max( maxVal, -minVal ) );
  val.convertTo( binVal1i, CV_32S, 20000 );
  Size szShow( W, H + spaceH + barH );
  szShow.height += minVal < 0 && !descendShow ? H + spaceH : 0;
  Mat showImg3b( szShow, CV_8UC3, Scalar( 255, 255, 255 ) );
  int* binH = (int*) ( binVal1i.data );
  Vec3b* binColor = (Vec3b*) ( binColor3b.data );
  int* binW = (int*) ( width1i.data );
  std::vector<CostiIdx> costIdx( n );
  if( descendShow )
  {
    for ( int i = 0; i < n; i++ )
      costIdx[i] = std::make_pair( binH[i], i );
    sort( costIdx.begin(), costIdx.end(), std::greater<CostiIdx>() );
  }

  // Show image
  for ( int i = 0, x = 0; i < n; i++ )
  {
    int idx = descendShow ? costIdx[i].second : i;
    int h = descendShow ? abs( binH[idx] ) : binH[idx];
    Scalar color( binColor[idx] );
    Rect reg( x, H + spaceH, binW[idx], barH );
    showImg3b( reg ) = color;  // Draw bar
    rectangle( showImg3b, reg, Scalar( 0 ) );

    reg.height = abs( h );
    reg.y = h >= 0 ? H - h : H + 2 * spaceH + barH;
    showImg3b( reg ) = color;
    rectangle( showImg3b, reg, Scalar( 0 ) );

    x += binW[idx];
  }
  return showImg3b;
}

}  // namespace saliency
}  // namespace cv
