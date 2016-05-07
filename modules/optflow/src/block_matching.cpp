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

namespace cv
{
namespace optflow
{

class OpticalFlowBlockMatching : public DenseOpticalFlow
{
protected:
  int windowSize;
  int blockSize;

  inline float submatrixAbsDiff( int x0, int y0, const Mat &I0, int x1, int y1, const Mat &I1 ) const;

public:
  OpticalFlowBlockMatching() : windowSize( 3 ), blockSize( 8 ){};

  void calc( InputArray I0, InputArray I1, InputOutputArray flow );
  void collectGarbage();
};

inline float OpticalFlowBlockMatching::submatrixAbsDiff( int x0, int y0, const Mat &I0, int x1, int y1,
                                                         const Mat &I1 ) const
{
  float error = 0;
  const Size size = I0.size();
  for ( int i = -windowSize; i <= windowSize; ++i )
  {
    if ( i + y0 < 0 || i + y0 >= size.height || i + y1 < 0 || i + y1 >= size.height )
    {
      error += 1;
      continue;
    }
    const Vec3f *I0X = I0.ptr<Vec3f>( i + y0 );
    const Vec3f *I1X = I1.ptr<Vec3f>( i + y1 );
    for ( int j = -windowSize; j <= windowSize; ++j )
    {
      if ( j + x0 < 0 || j + x0 >= size.width || j + x1 < 0 || j + x1 >= size.width )
      {
        error += 1;
        continue;
      }
      const Vec3f diff = I0X[j + x0] - I1X[j + x1];
      error += abs( diff[0] );
      error += abs( diff[1] );
      error += abs( diff[2] );
    }
  }
  return error;
}

void OpticalFlowBlockMatching::calc( InputArray I0, InputArray I1, InputOutputArray flow_out )
{
  CV_Assert( I0.channels() == 3 );
  CV_Assert( I1.channels() == 3 );
  Size size = I0.size();
  CV_Assert( size == I1.size() );

  flow_out.create( size, CV_32FC2 );
  Mat flow = flow_out.getMat();
  Mat from = I0.getMat();
  Mat to = I1.getMat();

  from.convertTo( from, CV_32FC3, 1.0 / 255.0 );
  to.convertTo( to, CV_32FC3, 1.0 / 255.0 );

  const float distNormalize = blockSize * sqrt( 2 );

  for ( int y0 = 0; y0 < size.height; ++y0 )
  {
    Vec2f *flowX = flow.ptr<Vec2f>( y0 );
    const int yEnd = std::min( size.height - 1, y0 + blockSize );

    for ( int x0 = 0; x0 < size.width; ++x0 )
    {
      float minDiff = 1e10;
      Vec2f du( 0, 0 );
      const int xEnd = std::min( size.width - 1, x0 + blockSize );
      for ( int y1 = std::max( 0, y0 - blockSize ); y1 <= yEnd; ++y1 )
        for ( int x1 = std::max( 0, x0 - blockSize ); x1 <= xEnd; ++x1 )
        {
          const float distance = sqrt( ( x0 - x1 ) * ( x0 - x1 ) + ( y0 - y1 ) * ( y0 - y1 ) ) / distNormalize;
          const float kernel = 1.0 + 0.5 * distance;
          const float diff = kernel * submatrixAbsDiff( x0, y0, from, x1, y1, to );
          if ( diff < minDiff )
          {
            minDiff = diff;
            du = Vec2f( ( x1 - x0 ), ( y1 - y0 ) );
          }
        }
      flowX[x0] = du;
    }
  }
}

void OpticalFlowBlockMatching::collectGarbage() {}

Ptr<DenseOpticalFlow> createOptFlow_BlockMatching() { return makePtr<OpticalFlowBlockMatching>(); }
}
}
