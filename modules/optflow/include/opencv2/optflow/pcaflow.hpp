/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#ifndef __OPENCV_OPTFLOW_PCAFLOW_HPP__
#define __OPENCV_OPTFLOW_PCAFLOW_HPP__

#include "opencv2/core.hpp"
#include "opencv2/video.hpp"

namespace cv
{
namespace optflow
{
/*
class PCAFlowBasis
{
public:
  Size size;

  PCAFlowBasis( Size basisSize = Size( 0, 0 ) ) : size( basisSize ) {}

  virtual ~PCAFlowBasis(){};

  virtual int getNumberOfComponents() const = 0;

  virtual void getBasisAtPoint( const Point2f &p, const Size &maxSize, float *outX, float *outY ) const = 0;

  virtual Point2f reduceAtPoint( const Point2f &p, const Size &maxSize, const float *w1, const float *w2 ) const = 0;
};*/

/*
 * Orthogonal basis from Discrete Cosine Transform.
 * Can be used without any learning or assumptions about flow structure for general purpose.
 * Gives low quality estimation.
 */
/*class PCAFlowGeneralBasis : public PCAFlowBasis
{
public:
  PCAFlowGeneralBasis( Size basisSize = Size( 18, 14 ) ) : PCAFlowBasis( basisSize ) {}

  int getNumberOfComponents() const { return size.area(); }

  void getBasisAtPoint( const Point2f &p, const Size &maxSize, float *outX, float *outY ) const
  {
    for ( int n1 = 0; n1 < size.width; ++n1 )
      for ( int n2 = 0; n2 < size.height; ++n2 )
        outX[n1 * size.height + n2] =
          cosf( ( n1 * M_PI / maxSize.width ) * ( p.x + 0.5 ) ) * cosf( ( n2 * M_PI / maxSize.height ) * ( p.y + 0.5 )
);
    memcpy( outY, outX, getNumberOfComponents() * sizeof( *outY ) );
  }

  Point2f reduceAtPoint( const Point2f &p, const Size &maxSize, const float *w1, const float *w2 ) const
  {
    Point2f res( 0, 0 );
    for ( int n1 = 0; n1 < size.width; ++n1 )
      for ( int n2 = 0; n2 < size.height; ++n2 )
      {
        const float c =
          cosf( ( n1 * M_PI / maxSize.width ) * ( p.x + 0.5 ) ) * cosf( ( n2 * M_PI / maxSize.height ) * ( p.y + 0.5 )
);
        res.x += c * w1[n1 * size.height + n2];
        res.y += c * w2[n1 * size.height + n2];
      }
    return res;
  }
};*/
/*
class PCAFlowLearnedBasis : public PCAFlowBasis
{
private:
  float *basisData;
  unsigned numberOfComponents;

public:
  PCAFlowLearnedBasis( const char *filename )
  {
    basisData = 0;
    FILE *f = fopen( filename, "r" );
    CV_Assert( f );

    numberOfComponents = 0;
    CV_Assert( fread( &numberOfComponents, sizeof( numberOfComponents ), 1, f ) == 1 );
    CV_Assert( fread( &size.height, sizeof( size.height ), 1, f ) == 1 );
    CV_Assert( fread( &size.width, sizeof( size.width ), 1, f ) == 1 );
    CV_Assert( ( numberOfComponents > 0 ) && ( numberOfComponents % 2 == 0 ) );

    basisData = new float[size.width * size.height * numberOfComponents];
    CV_Assert( fread( basisData, size.width * size.height * sizeof( *basisData ), numberOfComponents, f ) ==
               numberOfComponents );
    fclose( f );

    numberOfComponents /= 2;
  }

  ~PCAFlowLearnedBasis()
  {
    if ( basisData )
      delete[] basisData;
  }

  int getNumberOfComponents() const { return numberOfComponents; }

  void getBasisAtPoint( const Point2f &p, const Size &maxSize, float *outX, float *outY ) const
  {
    const size_t chunk = size.width * size.height;
    size_t offset = size_t( p.y * float(size.height) / maxSize.height ) * size.width + size_t( p.x * float(size.width) /
maxSize.width );
    for ( unsigned i = 0; i < numberOfComponents; ++i )
      outX[i] = basisData[i * chunk + offset];
    offset += numberOfComponents * chunk;
    for ( unsigned i = 0; i < numberOfComponents; ++i )
      outY[i] = basisData[i * chunk + offset];
  }

  Point2f reduceAtPoint( const Point2f &p, const Size &maxSize, const float *w1, const float *w2 ) const
  {
    Point2f res( 0, 0 );
    const size_t chunk = size.width * size.height;
    const size_t offset = size_t( p.y * float(size.height) / maxSize.height ) * size.width + size_t( p.x *
float(size.width) / maxSize.width );
    for ( unsigned i = 0; i < numberOfComponents; ++i )
    {
      const float c = basisData[i * chunk + offset];
      res.x += c * w1[i];
      res.y += c * w2[i];
    }
    return res;
  }
};*/

class OpticalFlowPCAFlow : public DenseOpticalFlow
{
protected:
  const Size basisSize;
  const float sparseRate;              // (0 .. 0.1)
  const float retainedCornersFraction; // [0 .. 1]
  const float occlusionsThreshold;
  const float dampingFactor;

public:
  OpticalFlowPCAFlow( const Size _basisSize = Size( 18, 14 ), float _sparseRate = 0.02,
                      float _retainedCornersFraction = 0.7, float _occlusionsThreshold = 0.0003,
                      float _dampingFactor = 0.00002 );

  void calc( InputArray I0, InputArray I1, InputOutputArray flow );
  void collectGarbage();

private:
  void findSparseFeatures( Mat &from, Mat &to, std::vector<Point2f> &features,
                           std::vector<Point2f> &predictedFeatures ) const;

  void removeOcclusions( Mat &from, Mat &to, std::vector<Point2f> &features,
                         std::vector<Point2f> &predictedFeatures ) const;

  void getSystem( OutputArray AOut, OutputArray b1Out, OutputArray b2Out, const std::vector<Point2f> &features,
                  const std::vector<Point2f> &predictedFeatures, const Size size );
};

CV_EXPORTS_W Ptr<DenseOpticalFlow> createOptFlow_PCAFlow();
}
}

#endif
