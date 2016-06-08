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
#include "opencv2/ximgproc/edge_filter.hpp"

namespace cv
{
namespace optflow
{

OpticalFlowPCAFlow::OpticalFlowPCAFlow( const Size _basisSize, float _sparseRate, float _retainedCornersFraction,
                                        float _occlusionsThreshold, float _dampingFactor )
    : basisSize( _basisSize ), sparseRate( _sparseRate ), retainedCornersFraction( _retainedCornersFraction ),
      occlusionsThreshold( _occlusionsThreshold ), dampingFactor( _dampingFactor )
{
  CV_Assert( sparseRate > 0 && sparseRate <= 0.1 );
  CV_Assert( retainedCornersFraction >= 0 && retainedCornersFraction <= 1.0 );
  CV_Assert( occlusionsThreshold > 0 );
}

inline float eDistSq( const Point2f &p1, const Point2f &p2 )
{
  const float dx = p1.x - p2.x;
  const float dy = p1.y - p2.y;
  return dx * dx + dy * dy;
}

inline float eNormSq( const Point2f &v ) { return v.x * v.x + v.y * v.y; }

template <typename T> static inline int mathSign( T val ) { return ( T( 0 ) < val ) - ( val < T( 0 ) ); }

static inline void symOrtho( double a, double b, double &c, double &s, double &r )
{
  if ( b == 0 )
  {
    c = mathSign( a );
    s = 0;
    r = std::abs( a );
  }
  else if ( a == 0 )
  {
    c = 0;
    s = mathSign( b );
    r = std::abs( b );
  }
  else if ( std::abs( b ) > std::abs( a ) )
  {
    const double tau = a / b;
    s = mathSign( b ) / sqrt( 1 + tau * tau );
    c = s * tau;
    r = b / s;
  }
  else
  {
    const double tau = b / a;
    c = mathSign( a ) / sqrt( 1 + tau * tau );
    s = c * tau;
    r = a / c;
  }
}

static void solveLSQR( const Mat &A, const Mat &b, OutputArray xOut, const double damp = 0.0,
                       const unsigned iter_lim = 10 )
{
  int m = A.size().height;
  int n = A.size().width;
  CV_Assert( m == b.size().height );
  CV_Assert( A.type() == CV_32F );
  CV_Assert( b.type() == CV_32F );
  xOut.create( n, 1, CV_32F );

  Mat v( n, 1, CV_32F, 0.0f );
  Mat u = b;
  Mat x = xOut.getMat();
  x = Mat::zeros( x.size(), x.type() );
  double alfa = 0;
  double beta = cv::norm( u, NORM_L2 );
  Mat w( n, 1, CV_32F, 0.0f );
  const Mat AT = A.t();

  if ( beta > 0 )
  {
    u *= 1 / beta;
    v = AT * u;
    alfa = cv::norm( v, NORM_L2 );
  }

  if ( alfa > 0 )
  {
    v *= 1 / alfa;
    w = v.clone();
  }

  double rhobar = alfa;
  double phibar = beta;
  if ( alfa * beta == 0 )
    return;

  for ( unsigned itn = 0; itn < iter_lim; ++itn )
  {
    u = A * v - alfa * u;
    beta = cv::norm( u, NORM_L2 );

    if ( beta > 0 )
    {
      u *= 1 / beta;
      v = AT * u - beta * v;
      alfa = cv::norm( v, NORM_L2 );
      if ( alfa > 0 )
        v = ( 1 / alfa ) * v;
    }

    double rhobar1 = sqrt( rhobar * rhobar + damp * damp );
    double cs1 = rhobar / rhobar1;
    phibar = cs1 * phibar;

    double cs, sn, rho;
    symOrtho( rhobar1, beta, cs, sn, rho );

    double theta = sn * alfa;
    rhobar = -cs * alfa;
    double phi = cs * phibar;
    phibar = sn * phibar;

    double t1 = phi / rho;
    double t2 = -theta / rho;

    x += t1 * w;
    w *= t2;
    w += v;
  }
}

void OpticalFlowPCAFlow::findSparseFeatures( Mat &from, Mat &to, std::vector<Point2f> &features,
                                             std::vector<Point2f> &predictedFeatures ) const
{
  Size size = from.size();
  const unsigned maxFeatures = size.area() * sparseRate;
  goodFeaturesToTrack( from, features, maxFeatures * retainedCornersFraction, 0.005, 3 );

  // Add points along the grid if not enough features
  if ( maxFeatures > features.size() )
  {
    const unsigned missingPoints = maxFeatures - features.size();
    const unsigned blockSize = sqrt( (float)size.area() / missingPoints );
    for ( int x = blockSize / 2; x < size.width; x += blockSize )
      for ( int y = blockSize / 2; y < size.height; y += blockSize )
        features.push_back( Point2f( x, y ) );
  }
  std::vector<uchar> predictedStatus;
  std::vector<float> predictedError;
  calcOpticalFlowPyrLK( from, to, features, predictedFeatures, predictedStatus, predictedError );

  size_t j = 0;
  for ( size_t i = 0; i < features.size(); ++i )
  {
    if ( predictedStatus[i] )
    {
      features[j] = features[i];
      predictedFeatures[j] = predictedFeatures[i];
      ++j;
    }
  }
  features.resize( j );
  predictedFeatures.resize( j );
}

void OpticalFlowPCAFlow::removeOcclusions( Mat &from, Mat &to, std::vector<Point2f> &features,
                                           std::vector<Point2f> &predictedFeatures ) const
{
  std::vector<uchar> predictedStatus;
  std::vector<float> predictedError;
  std::vector<Point2f> backwardFeatures;
  calcOpticalFlowPyrLK( to, from, predictedFeatures, backwardFeatures, predictedStatus, predictedError );

  size_t j = 0;
  const float threshold = occlusionsThreshold * sqrt( from.size().area() );
  for ( size_t i = 0; i < predictedFeatures.size(); ++i )
  {
    if ( predictedStatus[i] )
    {
      Point2f flowDiff = features[i] - backwardFeatures[i];
      if ( eNormSq( flowDiff ) <= threshold )
      {
        features[j] = features[i];
        predictedFeatures[j] = predictedFeatures[i];
        ++j;
      }
    }
  }
  features.resize( j );
  predictedFeatures.resize( j );
}

void OpticalFlowPCAFlow::getSystem( OutputArray AOut, OutputArray b1Out, OutputArray b2Out,
                                    const std::vector<Point2f> &features, const std::vector<Point2f> &predictedFeatures,
                                    const Size size )
{
  AOut.create( features.size(), basisSize.area(), CV_32F );
  b1Out.create( features.size(), 1, CV_32F );
  b2Out.create( features.size(), 1, CV_32F );
  Mat A = AOut.getMat();
  Mat b1 = b1Out.getMat();
  Mat b2 = b2Out.getMat();
  for ( size_t i = 0; i < features.size(); ++i )
  {
    const Point2f &p = features[i];
    float *row = A.ptr<float>( i );
    for ( int n1 = 0; n1 < basisSize.width; ++n1 )
      for ( int n2 = 0; n2 < basisSize.height; ++n2 )
        row[n1 * basisSize.height + n2] =
          cosf( ( n1 * M_PI / size.width ) * ( p.x + 0.5 ) ) * cosf( ( n2 * M_PI / size.height ) * ( p.y + 0.5 ) );
    const Point2f flow = predictedFeatures[i] - features[i];
    b1.at<float>( i ) = flow.x;
    b2.at<float>( i ) = flow.y;
  }
}

static void applyCLAHE( Mat &img )
{
  Ptr<CLAHE> clahe = createCLAHE();
  clahe->setClipLimit( 14 );
  clahe->apply( img, img );
}

static void reduceToFlow( const Mat &w1, const Mat &w2, Mat &flow, const Size &basisSize )
{
  const Size size = flow.size();
  Mat flowX( size, CV_32F, 0.0f );
  Mat flowY( size, CV_32F, 0.0f );

  const float mult = sqrt( size.area() ) * 0.5;

  for ( int i = 0; i < basisSize.width; ++i )
    for ( int j = 0; j < basisSize.height; ++j )
    {
      flowX.at<float>( j, i ) = w1.at<float>( i * basisSize.height + j ) * mult;
      flowY.at<float>( j, i ) = w2.at<float>( i * basisSize.height + j ) * mult;
    }
  for ( int i = 0; i < basisSize.height; ++i )
  {
    flowX.at<float>( i, 0 ) *= M_SQRT2;
    flowY.at<float>( i, 0 ) *= M_SQRT2;
  }
  for ( int i = 0; i < basisSize.width; ++i )
  {
    flowX.at<float>( 0, i ) *= M_SQRT2;
    flowY.at<float>( 0, i ) *= M_SQRT2;
  }

  dct( flowX, flowX, DCT_INVERSE );
  dct( flowY, flowY, DCT_INVERSE );
  for ( int i = 0; i < size.height; ++i )
    for ( int j = 0; j < size.width; ++j )
      flow.at<Point2f>( i, j ) = Point2f( flowX.at<float>( i, j ), flowY.at<float>( i, j ) );
}

void OpticalFlowPCAFlow::calc( InputArray I0, InputArray I1, InputOutputArray flowOut )
{
  const Size size = I0.size();
  CV_Assert( size == I1.size() );

  Mat from, to;
  if ( I0.channels() == 3 )
  {
    cvtColor( I0, from, COLOR_BGR2GRAY );
    from.convertTo( from, CV_8U );
  }
  else
  {
    I0.getMat().convertTo( from, CV_8U );
  }
  if ( I1.channels() == 3 )
  {
    cvtColor( I1, to, COLOR_BGR2GRAY );
    to.convertTo( to, CV_8U );
  }
  else
  {
    I1.getMat().convertTo( to, CV_8U );
  }

  CV_Assert( from.channels() == 1 );
  CV_Assert( to.channels() == 1 );

  const Mat fromOrig = from.clone();

  applyCLAHE( from );
  applyCLAHE( to );

  std::vector<Point2f> features, predictedFeatures;
  findSparseFeatures( from, to, features, predictedFeatures );
  removeOcclusions( from, to, features, predictedFeatures );

  flowOut.create( size, CV_32FC2 );
  Mat flow = flowOut.getMat();

  Mat A, b1, b2, w1, w2;
  getSystem( A, b1, b2, features, predictedFeatures, size );
  // solve( A1, b1, w1, DECOMP_CHOLESKY | DECOMP_NORMAL );
  // solve( A2, b2, w2, DECOMP_CHOLESKY | DECOMP_NORMAL );
  solveLSQR( A, b1, w1, dampingFactor * size.area() );
  solveLSQR( A, b2, w2, dampingFactor * size.area() );
  Mat flowSmall( (size / 8) * 2, CV_32FC2 );
  reduceToFlow( w1, w2, flowSmall, basisSize );
  resize( flowSmall, flow, size, 0, 0, INTER_LINEAR );
  ximgproc::fastGlobalSmootherFilter(fromOrig, flow, flow, 500, 2);
}

void OpticalFlowPCAFlow::collectGarbage() {}

Ptr<DenseOpticalFlow> createOptFlow_PCAFlow() { return makePtr<OpticalFlowPCAFlow>(); }
}
}
