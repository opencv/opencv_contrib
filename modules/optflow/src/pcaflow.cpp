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

#include "opencv2/ximgproc/edge_filter.hpp"
#include "precomp.hpp"

/* Disable "from double to float" and "from size_t to int" warnings.
 * Fixing these would make the code look ugly by introducing explicit cast all around.
 * Here these warning are pointless anyway.
 */
#ifdef _MSC_VER
#pragma warning( disable : 4305 4244 4267 4838 )
#endif
#ifdef __clang__
#pragma clang diagnostic ignored "-Wshorten-64-to-32"
#endif

namespace cv
{
namespace optflow
{
namespace
{

#ifndef M_SQRT2
const float M_SQRT2 = 1.41421356237309504880;
#endif

template <typename T> inline int mathSign( T val ) { return ( T( 0 ) < val ) - ( val < T( 0 ) ); }

/* Stable symmetric Householder reflection that gives c and s such that
 *   [ c  s ][a] = [d],
 *   [ s -c ][b]   [0]
 *
 * Output:
 *   c -- cosine(theta), where theta is the implicit angle of rotation
 *        (counter-clockwise) in a plane-rotation
 *   s -- sine(theta)
 *   r -- two-norm of [a; b]
 */
inline void symOrtho( double a, double b, double &c, double &s, double &r )
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
    s = mathSign( b ) / std::sqrt( 1 + tau * tau );
    c = s * tau;
    r = b / s;
  }
  else
  {
    const double tau = b / a;
    c = mathSign( a ) / std::sqrt( 1 + tau * tau );
    s = c * tau;
    r = a / c;
  }
}

/* Iterative LSQR algorithm for solving least squares problems.
 *
 * [1] Paige, C. C. and M. A. Saunders,
 * LSQR: An Algorithm for Sparse Linear Equations And Sparse Least Squares
 * ACM Trans. Math. Soft., Vol.8, 1982, pp. 43-71.
 *
 * Solves the following problem:
 *   argmin_x ||Ax - b|| + damp||x||
 *
 * Output:
 *   x -- approximate solution
 */
void solveLSQR( const Mat &A, const Mat &b, OutputArray xOut, const double damp = 0.0, const unsigned iter_lim = 10 )
{
  const int n = A.size().width;
  CV_Assert( A.size().height == b.size().height );
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
    u *= -alfa;
    u += A * v;
    beta = cv::norm( u, NORM_L2 );

    if ( beta > 0 )
    {
      u *= 1 / beta;
      v *= -beta;
      v += AT * u;
      alfa = cv::norm( v, NORM_L2 );
      if ( alfa > 0 )
        v *= 1 / alfa;
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

inline void _cpu_fillDCTSampledPoints( float *row, const Point2f &p, const Size &basisSize, const Size &size )
{
  for ( int n1 = 0; n1 < basisSize.width; ++n1 )
    for ( int n2 = 0; n2 < basisSize.height; ++n2 )
      row[n1 * basisSize.height + n2] =
        cosf( ( n1 * CV_PI / size.width ) * ( p.x + 0.5 ) ) * cosf( ( n2 * CV_PI / size.height ) * ( p.y + 0.5 ) );
}

ocl::ProgramSource _ocl_fillDCTSampledPointsSource(
  "__kernel void fillDCTSampledPoints(__global const uchar* features, int fstep, int foff, __global "
  "uchar* A, int Astep, int Aoff, int fs, int bsw, int bsh, int sw, int sh) {"
  "const int i = get_global_id(0);"
  "const int n1 = get_global_id(1);"
  "const int n2 = get_global_id(2);"
  "if (i >= fs || n1 >= bsw || n2 >= bsh) return;"
  "__global const float2* f = (__global const float2*)(features + (fstep * i + foff));"
  "__global float* a = (__global float*)(A + (Astep * i + Aoff + (n1 * bsh + n2) * sizeof(float)));"
  "const float2 p = f[0];"
  "const float pi = 3.14159265358979323846;"
  "a[0] = cos((n1 * pi / sw) * (p.x + 0.5)) * cos((n2 * pi / sh) * (p.y + 0.5));"
  "}" );

void applyCLAHE( UMat &img, float claheClip )
{
  Ptr<CLAHE> clahe = createCLAHE();
  clahe->setClipLimit( claheClip );
  clahe->apply( img, img );
}

void reduceToFlow( const Mat &w1, const Mat &w2, Mat &flow, const Size &basisSize )
{
  const Size size = flow.size();
  Mat flowX( size, CV_32F, 0.0f );
  Mat flowY( size, CV_32F, 0.0f );

  const float mult = sqrt( static_cast<float>(size.area()) ) * 0.5;

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
}

void OpticalFlowPCAFlow::findSparseFeatures( UMat &from, UMat &to, std::vector<Point2f> &features,
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

void OpticalFlowPCAFlow::removeOcclusions( UMat &from, UMat &to, std::vector<Point2f> &features,
                                           std::vector<Point2f> &predictedFeatures ) const
{
  std::vector<uchar> predictedStatus;
  std::vector<float> predictedError;
  std::vector<Point2f> backwardFeatures;
  calcOpticalFlowPyrLK( to, from, predictedFeatures, backwardFeatures, predictedStatus, predictedError );

  size_t j = 0;
  const float threshold = occlusionsThreshold * sqrt( static_cast<float>(from.size().area()) );
  for ( size_t i = 0; i < predictedFeatures.size(); ++i )
  {
    if ( predictedStatus[i] )
    {
      Point2f flowDiff = features[i] - backwardFeatures[i];
      if ( flowDiff.dot( flowDiff ) <= threshold )
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
  if ( useOpenCL )
  {
    UMat A = AOut.getUMat();
    Mat b1 = b1Out.getMat();
    Mat b2 = b2Out.getMat();

    ocl::Kernel kernel( "fillDCTSampledPoints", _ocl_fillDCTSampledPointsSource );
    size_t globSize[] = {features.size(), basisSize.width, basisSize.height};
    kernel
      .args( cv::ocl::KernelArg::ReadOnlyNoSize( Mat( features ).getUMat( ACCESS_READ ) ),
             cv::ocl::KernelArg::WriteOnlyNoSize( A ), (int)features.size(), (int)basisSize.width,
             (int)basisSize.height, (int)size.width, (int)size.height )
      .run( 3, globSize, 0, true );

    for ( size_t i = 0; i < features.size(); ++i )
    {
      const Point2f flow = predictedFeatures[i] - features[i];
      b1.at<float>( i ) = flow.x;
      b2.at<float>( i ) = flow.y;
    }
  }
  else
  {
    Mat A = AOut.getMat();
    Mat b1 = b1Out.getMat();
    Mat b2 = b2Out.getMat();

    for ( size_t i = 0; i < features.size(); ++i )
    {
      _cpu_fillDCTSampledPoints( A.ptr<float>( i ), features[i], basisSize, size );
      const Point2f flow = predictedFeatures[i] - features[i];
      b1.at<float>( i ) = flow.x;
      b2.at<float>( i ) = flow.y;
    }
  }
}

void OpticalFlowPCAFlow::getSystem( OutputArray A1Out, OutputArray A2Out, OutputArray b1Out, OutputArray b2Out,
                                    const std::vector<Point2f> &features, const std::vector<Point2f> &predictedFeatures,
                                    const Size size )
{
  CV_Assert( prior->getBasisSize() == basisSize.area() );

  A1Out.create( features.size() + prior->getPadding(), basisSize.area(), CV_32F );
  A2Out.create( features.size() + prior->getPadding(), basisSize.area(), CV_32F );
  b1Out.create( features.size() + prior->getPadding(), 1, CV_32F );
  b2Out.create( features.size() + prior->getPadding(), 1, CV_32F );

  if ( useOpenCL )
  {
    UMat A = A1Out.getUMat();
    Mat b1 = b1Out.getMat();
    Mat b2 = b2Out.getMat();

    ocl::Kernel kernel( "fillDCTSampledPoints", _ocl_fillDCTSampledPointsSource );
    size_t globSize[] = {features.size(), basisSize.width, basisSize.height};
    kernel
      .args( cv::ocl::KernelArg::ReadOnlyNoSize( Mat( features ).getUMat( ACCESS_READ ) ),
             cv::ocl::KernelArg::WriteOnlyNoSize( A ), (int)features.size(), (int)basisSize.width,
             (int)basisSize.height, (int)size.width, (int)size.height )
      .run( 3, globSize, 0, true );

    for ( size_t i = 0; i < features.size(); ++i )
    {
      const Point2f flow = predictedFeatures[i] - features[i];
      b1.at<float>( i ) = flow.x;
      b2.at<float>( i ) = flow.y;
    }
  }
  else
  {
    Mat A1 = A1Out.getMat();
    Mat b1 = b1Out.getMat();
    Mat b2 = b2Out.getMat();

    for ( size_t i = 0; i < features.size(); ++i )
    {
      _cpu_fillDCTSampledPoints( A1.ptr<float>( i ), features[i], basisSize, size );
      const Point2f flow = predictedFeatures[i] - features[i];
      b1.at<float>( i ) = flow.x;
      b2.at<float>( i ) = flow.y;
    }
  }

  Mat A1 = A1Out.getMat();
  Mat A2 = A2Out.getMat();
  Mat b1 = b1Out.getMat();
  Mat b2 = b2Out.getMat();

  memcpy( A2.ptr<float>(), A1.ptr<float>(), features.size() * basisSize.area() * sizeof( float ) );
  prior->fillConstraints( A1.ptr<float>( features.size(), 0 ), A2.ptr<float>( features.size(), 0 ),
                          b1.ptr<float>( features.size(), 0 ), b2.ptr<float>( features.size(), 0 ) );
}

void OpticalFlowPCAFlow::calc( InputArray I0, InputArray I1, InputOutputArray flowOut )
{
  const Size size = I0.size();
  CV_Assert( size == I1.size() );

  UMat from, to;
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

  const Mat fromOrig = from.getMat( ACCESS_READ ).clone();
  useOpenCL = flowOut.isUMat() && ocl::useOpenCL();

  applyCLAHE( from, claheClip );
  applyCLAHE( to, claheClip );

  std::vector<Point2f> features, predictedFeatures;
  findSparseFeatures( from, to, features, predictedFeatures );
  removeOcclusions( from, to, features, predictedFeatures );

  flowOut.create( size, CV_32FC2 );
  Mat flow = flowOut.getMat();

  Mat w1, w2;
  if ( prior.get() )
  {
    Mat A1, A2, b1, b2;
    getSystem( A1, A2, b1, b2, features, predictedFeatures, size );
    solveLSQR( A1, b1, w1, dampingFactor * size.area() );
    solveLSQR( A2, b2, w2, dampingFactor * size.area() );
  }
  else
  {
    Mat A, b1, b2;
    getSystem( A, b1, b2, features, predictedFeatures, size );
    solveLSQR( A, b1, w1, dampingFactor * size.area() );
    solveLSQR( A, b2, w2, dampingFactor * size.area() );
  }
  Mat flowSmall( ( size / 8 ) * 2, CV_32FC2 );
  reduceToFlow( w1, w2, flowSmall, basisSize );
  resize( flowSmall, flow, size, 0, 0, INTER_LINEAR );
  ximgproc::fastGlobalSmootherFilter( fromOrig, flow, flow, 500, 2 );
}

OpticalFlowPCAFlow::OpticalFlowPCAFlow( Ptr<const PCAPrior> _prior, const Size _basisSize, float _sparseRate,
                                        float _retainedCornersFraction, float _occlusionsThreshold,
                                        float _dampingFactor, float _claheClip )
    : prior( _prior ), basisSize( _basisSize ), sparseRate( _sparseRate ),
      retainedCornersFraction( _retainedCornersFraction ), occlusionsThreshold( _occlusionsThreshold ),
      dampingFactor( _dampingFactor ), claheClip( _claheClip ), useOpenCL( false )
{
  CV_Assert( sparseRate > 0 && sparseRate <= 0.1 );
  CV_Assert( retainedCornersFraction >= 0 && retainedCornersFraction <= 1.0 );
  CV_Assert( occlusionsThreshold > 0 );
}

void OpticalFlowPCAFlow::collectGarbage() {}

Ptr<DenseOpticalFlow> createOptFlow_PCAFlow() { return makePtr<OpticalFlowPCAFlow>(); }

PCAPrior::PCAPrior( const char *pathToPrior )
{
  FILE *f = fopen( pathToPrior, "rb" );
  CV_Assert( f );

  unsigned n = 0, m = 0;
  CV_Assert( fread( &n, sizeof( n ), 1, f ) == 1 );
  CV_Assert( fread( &m, sizeof( m ), 1, f ) == 1 );

  L1.create( n, m, CV_32F );
  L2.create( n, m, CV_32F );
  c1.create( n, 1, CV_32F );
  c2.create( n, 1, CV_32F );

  CV_Assert( fread( L1.ptr<float>(), n * m * sizeof( float ), 1, f ) == 1 );
  CV_Assert( fread( L2.ptr<float>(), n * m * sizeof( float ), 1, f ) == 1 );
  CV_Assert( fread( c1.ptr<float>(), n * sizeof( float ), 1, f ) == 1 );
  CV_Assert( fread( c2.ptr<float>(), n * sizeof( float ), 1, f ) == 1 );

  fclose( f );
}

void PCAPrior::fillConstraints( float *A1, float *A2, float *b1, float *b2 ) const
{
  memcpy( A1, L1.ptr<float>(), L1.size().area() * sizeof( float ) );
  memcpy( A2, L2.ptr<float>(), L2.size().area() * sizeof( float ) );
  memcpy( b1, c1.ptr<float>(), c1.size().area() * sizeof( float ) );
  memcpy( b2, c2.ptr<float>(), c2.size().area() * sizeof( float ) );
}
}
}
