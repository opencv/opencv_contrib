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

#include "opencv2/core/core_c.h"
#include "opencv2/core/private.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/highgui.hpp"
#include "precomp.hpp"
#include "opencl_kernels_optflow.hpp"

/* Disable "from double to float" and "from size_t to int" warnings.
 * Fixing these would make the code look ugly by introducing explicit cast all around.
 * Here these warning are pointless anyway.
 */
#ifdef _MSC_VER
#pragma warning( disable : 4244 4267 4838 )
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

#define PATCH_RADIUS 10
#define PATCH_RADIUS_DOUBLED 20
#define SQRT2_INV 0.7071067811865475

const int patchRadius = PATCH_RADIUS;
const int globalIters = 3;
const int localIters = 500;
const double thresholdOutliers = 0.98;
const double thresholdMagnitudeFrac = 0.8;
const double epsTolerance = 1e-12;
const unsigned scoreGainPos = 5;
const unsigned scoreGainNeg = 1;
const unsigned negSearchKNN = 5;
const double simulatedAnnealingTemperatureCoef = 200.0;
const double sigmaGrowthRate = 0.2;

RNG rng;

struct Magnitude
{
  float val;
  int i;
  int j;

  Magnitude( float _val, int _i, int _j ) : val( _val ), i( _i ), j( _j ) {}
  Magnitude() {}

  bool operator<( const Magnitude &m ) const { return val > m.val; }
};

struct PartitionPredicate1
{
  Vec< double, GPCPatchDescriptor::nFeatures > coef;
  double rhs;

  PartitionPredicate1( const Vec< double, GPCPatchDescriptor::nFeatures > &_coef, double _rhs ) : coef( _coef ), rhs( _rhs ) {}

  bool operator()( const GPCPatchSample &sample ) const
  {
    bool refdir, posdir, negdir;
    sample.getDirections( refdir, posdir, negdir, coef, rhs );
    return refdir == false && ( posdir == false || negdir == true );
  }
};

struct PartitionPredicate2
{
  Vec< double, GPCPatchDescriptor::nFeatures > coef;
  double rhs;

  PartitionPredicate2( const Vec< double, GPCPatchDescriptor::nFeatures > &_coef, double _rhs ) : coef( _coef ), rhs( _rhs ) {}

  bool operator()( const GPCPatchSample &sample ) const
  {
    bool refdir, posdir, negdir;
    sample.getDirections( refdir, posdir, negdir, coef, rhs );
    return refdir != posdir && refdir == negdir;
  }
};

struct CompareWithTolerance
{
  double val;

  CompareWithTolerance( double _val ) : val( _val ) {};

  bool operator()( const double &elem ) const
  {
    const double diff = ( val + elem == 0 ) ? std::abs( val - elem ) : std::abs( ( val - elem ) / ( val + elem ) );
    return diff <= epsTolerance;
  }
};

float normL2Sqr( const Vec2f &v ) { return v[0] * v[0] + v[1] * v[1]; }

int normL2Sqr( const Point2i &v ) { return v.x * v.x + v.y * v.y; }

bool checkBounds( int i, int j, Size sz )
{
  return i >= patchRadius && j >= patchRadius && i + patchRadius < sz.height && j + patchRadius < sz.width;
}

void getDCTPatchDescriptor( GPCPatchDescriptor &patchDescr, const Mat *imgCh, int i, int j )
{
  Rect roi( j - patchRadius, i - patchRadius, 2 * patchRadius, 2 * patchRadius );
  Mat freqDomain;
  dct( imgCh[0]( roi ), freqDomain );

  double *feature = patchDescr.feature.val;
  feature[0] = freqDomain.at< float >( 0, 0 );
  feature[1] = freqDomain.at< float >( 0, 1 );
  feature[2] = freqDomain.at< float >( 0, 2 );
  feature[3] = freqDomain.at< float >( 0, 3 );

  feature[4] = freqDomain.at< float >( 1, 0 );
  feature[5] = freqDomain.at< float >( 1, 1 );
  feature[6] = freqDomain.at< float >( 1, 2 );
  feature[7] = freqDomain.at< float >( 1, 3 );

  feature[8] = freqDomain.at< float >( 2, 0 );
  feature[9] = freqDomain.at< float >( 2, 1 );
  feature[10] = freqDomain.at< float >( 2, 2 );
  feature[11] = freqDomain.at< float >( 2, 3 );

  feature[12] = freqDomain.at< float >( 3, 0 );
  feature[13] = freqDomain.at< float >( 3, 1 );
  feature[14] = freqDomain.at< float >( 3, 2 );
  feature[15] = freqDomain.at< float >( 3, 3 );

  feature[16] = cv::sum( imgCh[1]( roi ) )[0] / ( 2 * patchRadius );
  feature[17] = cv::sum( imgCh[2]( roi ) )[0] / ( 2 * patchRadius );
}

double sumInt( const Mat &integ, int i, int j, int h, int w )
{
  return integ.at< double >( i + h, j + w ) - integ.at< double >( i + h, j ) - integ.at< double >( i, j + w ) + integ.at< double >( i, j );
}

void getWHTPatchDescriptor( GPCPatchDescriptor &patchDescr, const Mat *imgCh, int i, int j )
{
  i -= patchRadius;
  j -= patchRadius;
  const int k = 2 * patchRadius;
  const double s = sumInt( imgCh[0], i, j, k, k );
  double *feature = patchDescr.feature.val;

  feature[0] = s;
  feature[1] = s - 2 * sumInt( imgCh[0], i, j + k / 2, k, k / 2 );
  feature[2] = s - 2 * sumInt( imgCh[0], i, j + k / 4, k, k / 2 );
  feature[3] = s - 2 * sumInt( imgCh[0], i, j + k / 4, k, k / 4 ) - 2 * sumInt( imgCh[0], i, j + 3 * k / 4, k, k / 4 );

  feature[4] = s - 2 * sumInt( imgCh[0], i + k / 2, j, k / 2, k );
  feature[5] = s - 2 * sumInt( imgCh[0], i, j + k / 2, k / 2, k / 2 ) - 2 * sumInt( imgCh[0], i + k / 2, j, k / 2, k / 2 );
  feature[6] = s - 2 * sumInt( imgCh[0], i, j + k / 4, k / 2, k / 2 ) - 2 * sumInt( imgCh[0], i + k / 2, j, k / 2, k / 4 ) -
               2 * sumInt( imgCh[0], i + k / 2, j + 3 * k / 4, k / 2, k / 4 );
  feature[7] = s - 2 * sumInt( imgCh[0], i, j + k / 4, k / 2, k / 4 ) - 2 * sumInt( imgCh[0], i, j + 3 * k / 4, k / 2, k / 4 ) -
               2 * sumInt( imgCh[0], i + k / 2, j, k / 2, k / 4 ) - 2 * sumInt( imgCh[0], i + k / 2, j + k / 2, k / 2, k / 4 );

  feature[8] = s - 2 * sumInt( imgCh[0], i + k / 4, j, k / 2, k );
  feature[9] = s - 2 * sumInt( imgCh[0], i + k / 4, j, k / 2, k / 2 ) - 2 * sumInt( imgCh[0], i, j + k / 2, k / 4, k / 2 ) -
               2 * sumInt( imgCh[0], i + 3 * k / 4, j + k / 2, k / 4, k / 2 );
  feature[10] = s - 2 * sumInt( imgCh[0], i + k / 4, j, k / 2, k / 4 ) - 2 * sumInt( imgCh[0], i + k / 4, j + 3 * k / 4, k / 2, k / 4 ) -
                2 * sumInt( imgCh[0], i, j + k / 4, k / 4, k / 2 ) - 2 * sumInt( imgCh[0], i + 3 * k / 4, j + k / 4, k / 4, k / 2 );
  feature[11] = s - 2 * sumInt( imgCh[0], i, j + k / 4, k / 4, k / 4 ) - 2 * sumInt( imgCh[0], i, j + 3 * k / 4, k / 4, k / 4 ) -
                2 * sumInt( imgCh[0], i + k / 4, j, k / 2, k / 4 ) - 2 * sumInt( imgCh[0], i + k / 4, j + k / 2, k / 2, k / 4 ) -
                2 * sumInt( imgCh[0], i + 3 * k / 4, j + k / 4, k / 4, k / 4 ) -
                2 * sumInt( imgCh[0], i + 3 * k / 4, j + 3 * k / 4, k / 4, k / 4 );

  feature[12] = s - 2 * sumInt( imgCh[0], i + k / 4, j, k / 4, k ) - 2 * sumInt( imgCh[0], i + 3 * k / 4, j, k / 4, k );
  feature[13] = s - 2 * sumInt( imgCh[0], i + k / 4, j, k / 4, k / 2 ) - 2 * sumInt( imgCh[0], i + 3 * k / 4, j, k / 4, k / 2 ) -
                2 * sumInt( imgCh[0], i, j + k / 2, k / 4, k / 2 ) - 2 * sumInt( imgCh[0], i + k / 2, j + k / 2, k / 4, k / 2 );
  feature[14] = s - 2 * sumInt( imgCh[0], i + k / 4, j, k / 4, k / 4 ) - 2 * sumInt( imgCh[0], i + 3 * k / 4, j, k / 4, k / 4 ) -
                2 * sumInt( imgCh[0], i, j + k / 4, k / 4, k / 2 ) - 2 * sumInt( imgCh[0], i + k / 2, j + k / 4, k / 4, k / 2 ) -
                2 * sumInt( imgCh[0], i + k / 4, j + 3 * k / 4, k / 4, k / 4 ) -
                2 * sumInt( imgCh[0], i + 3 * k / 4, j + 3 * k / 4, k / 4, k / 4 );
  feature[15] = s - 2 * sumInt( imgCh[0], i, j + k / 4, k / 4, k / 4 ) - 2 * sumInt( imgCh[0], i, j + 3 * k / 4, k / 4, k / 4 ) -
                2 * sumInt( imgCh[0], i + k / 4, j, k / 4, k / 4 ) - 2 * sumInt( imgCh[0], i + k / 4, j + k / 2, k / 4, k / 4 ) -
                2 * sumInt( imgCh[0], i + k / 2, j + k / 4, k / 4, k / 4 ) -
                2 * sumInt( imgCh[0], i + k / 2, j + 3 * k / 4, k / 4, k / 4 ) - 2 * sumInt( imgCh[0], i + 3 * k / 4, j, k / 4, k / 4 ) -
                2 * sumInt( imgCh[0], i + 3 * k / 4, j + k / 2, k / 4, k / 4 );

  feature[16] = sumInt( imgCh[1], i, j, k, k );
  feature[17] = sumInt( imgCh[2], i, j, k, k );

  patchDescr.feature /= patchRadius;
}

class ParallelDCTFiller : public ParallelLoopBody
{
private:
  const Size sz;
  const Mat *imgCh;
  std::vector< GPCPatchDescriptor > *descr;

  ParallelDCTFiller &operator=( const ParallelDCTFiller & );

public:
  ParallelDCTFiller( const Size &_sz, const Mat *_imgCh, std::vector< GPCPatchDescriptor > *_descr )
      : sz( _sz ), imgCh( _imgCh ), descr( _descr ){};

  void operator()( const Range &range ) const
  {
    for ( int i = range.start; i < range.end; ++i )
    {
      int x, y;
      GPCDetails::getCoordinatesFromIndex( i, sz, x, y );
      getDCTPatchDescriptor( descr->at( i ), imgCh, y, x );
    }
  }
};

#ifdef HAVE_OPENCL

bool ocl_getAllDCTDescriptorsForImage( const Mat *imgCh, std::vector< GPCPatchDescriptor > &descr )
{
  const Size sz = imgCh[0].size();
  ocl::Kernel kernel( "getPatchDescriptor", ocl::optflow::sparse_matching_gpc_oclsrc,
                      format( "-DPATCH_RADIUS_DOUBLED=%d -DCV_PI=%f -DSQRT2_INV=%f", PATCH_RADIUS_DOUBLED, CV_PI, SQRT2_INV ) );
  size_t globSize[] = {sz.height - 2 * patchRadius, sz.width - 2 * patchRadius};
  UMat out( globSize[0] * globSize[1], GPCPatchDescriptor::nFeatures, CV_64F );
  if (
    kernel
    .args( cv::ocl::KernelArg::ReadOnlyNoSize( imgCh[0].getUMat( ACCESS_READ ) ),
           cv::ocl::KernelArg::ReadOnlyNoSize( imgCh[1].getUMat( ACCESS_READ ) ),
           cv::ocl::KernelArg::ReadOnlyNoSize( imgCh[2].getUMat( ACCESS_READ ) ),
           cv::ocl::KernelArg::WriteOnlyNoSize( out ),
           (int)globSize[0], (int)globSize[1], (int)patchRadius )
    .run( 2, globSize, 0, true ) == false )
    return false;
  Mat cpuOut = out.getMat( 0 );
  for ( int i = 0; i + 2 * patchRadius < sz.height; ++i )
    for ( int j = 0; j + 2 * patchRadius < sz.width; ++j )
      descr.push_back( *cpuOut.ptr< GPCPatchDescriptor >( i * globSize[1] + j ) );
  return true;
}

#endif

void getAllDCTDescriptorsForImage( const Mat *imgCh, std::vector< GPCPatchDescriptor > &descr, const GPCMatchingParams &mp )
{
  const Size sz = imgCh[0].size();
  descr.reserve( ( sz.height - 2 * patchRadius ) * ( sz.width - 2 * patchRadius ) );

  (void)mp; // Fix unused parameter warning in case OpenCL is not available
  CV_OCL_RUN( mp.useOpenCL, ocl_getAllDCTDescriptorsForImage( imgCh, descr ) )

  descr.resize( ( sz.height - 2 * patchRadius ) * ( sz.width - 2 * patchRadius ) );
  parallel_for_( Range( 0, descr.size() ), ParallelDCTFiller( sz, imgCh, &descr ) );
}

class ParallelWHTFiller : public ParallelLoopBody
{
private:
  const Size sz;
  const Mat *imgChInt;
  std::vector< GPCPatchDescriptor > *descr;

  ParallelWHTFiller &operator=( const ParallelWHTFiller & );

public:
  ParallelWHTFiller( const Size &_sz, const Mat *_imgChInt, std::vector< GPCPatchDescriptor > *_descr )
      : sz( _sz ), imgChInt( _imgChInt ), descr( _descr ){};

  void operator()( const Range &range ) const
  {
    for ( int i = range.start; i < range.end; ++i )
    {
      int x, y;
      GPCDetails::getCoordinatesFromIndex( i, sz, x, y );
      getWHTPatchDescriptor( descr->at( i ), imgChInt, y, x );
    }
  }
};

void getAllWHTDescriptorsForImage( const Mat *imgCh, std::vector< GPCPatchDescriptor > &descr, const GPCMatchingParams & )
{
  const Size sz = imgCh[0].size();
  descr.resize( ( sz.height - 2 * patchRadius ) * ( sz.width - 2 * patchRadius ) );

  Mat imgChInt[3];
  integral( imgCh[0], imgChInt[0], CV_64F );
  integral( imgCh[1], imgChInt[1], CV_64F );
  integral( imgCh[2], imgChInt[2], CV_64F );

  parallel_for_( Range( 0, descr.size() ), ParallelWHTFiller( sz, imgChInt, &descr ) );
}

void buildIndex( OutputArray featuresOut, flann::Index &index, const Mat *imgCh,
                 void ( *getAllDescrFn )( const Mat *, std::vector< GPCPatchDescriptor > &, const GPCMatchingParams & ) )
{
  std::vector< GPCPatchDescriptor > descriptors;
  getAllDescrFn( imgCh, descriptors, GPCMatchingParams() );

  featuresOut.create( descriptors.size(), GPCPatchDescriptor::nFeatures, CV_32F );
  Mat features = featuresOut.getMat();

  for ( size_t i = 0; i < descriptors.size(); ++i )
    *features.ptr< Vec< float, GPCPatchDescriptor::nFeatures > >( i ) = descriptors[i].feature;

  cv::flann::KDTreeIndexParams indexParams;
  index.build( features, indexParams, cvflann::FLANN_DIST_L2 );
}

void getTriplet( const Magnitude &mag, const Mat &gt, const Mat *fromCh, const Mat *toCh, GPCSamplesVector &samples, flann::Index &index,
                 void ( *getDescFn )( GPCPatchDescriptor &, const Mat *, int, int ) )
{
  const Size sz = gt.size();
  const int i0 = mag.i;
  const int j0 = mag.j;
  const int i1 = i0 + cvRound( gt.at< Vec2f >( i0, j0 )[1] );
  const int j1 = j0 + cvRound( gt.at< Vec2f >( i0, j0 )[0] );
  if ( checkBounds( i1, j1, sz ) )
  {
    GPCPatchSample ps;
    getDescFn( ps.ref, fromCh, i0, j0 );
    getDescFn( ps.pos, toCh, i1, j1 );
    ps.neg.markAsSeparated();

    Matx< float, 1, GPCPatchDescriptor::nFeatures > ref32;
    Matx< int, 1, negSearchKNN > indices;
    int maxDist = 0;

    for ( unsigned i = 0; i < GPCPatchDescriptor::nFeatures; ++i )
      ref32( 0, i ) = ps.ref.feature[i];

    index.knnSearch( ref32, indices, noArray(), negSearchKNN );

    for ( unsigned i = 0; i < negSearchKNN; ++i )
    {
      int i2, j2;
      GPCDetails::getCoordinatesFromIndex( indices( 0, i ), sz, j2, i2 );
      const int dist = ( i2 - i1 ) * ( i2 - i1 ) + ( j2 - j1 ) * ( j2 - j1 );
      if ( maxDist < dist )
      {
        maxDist = dist;
        getDescFn( ps.neg, toCh, i2, j2 );
      }
    }

    samples.push_back( ps );
  }
}

void getTrainingSamples( const Mat &from, const Mat &to, const Mat &gt, GPCSamplesVector &samples, const int type )
{
  const Size sz = gt.size();
  std::vector< Magnitude > mag;

  for ( int i = patchRadius; i + patchRadius < sz.height; ++i )
    for ( int j = patchRadius; j + patchRadius < sz.width; ++j )
      mag.push_back( Magnitude( normL2Sqr( gt.at< Vec2f >( i, j ) ), i, j ) );

  size_t n = size_t( mag.size() * thresholdMagnitudeFrac ); // As suggested in the paper, we discard part of the training samples
                                                            // with a small displacement and train to better distinguish hard pairs.
  std::nth_element( mag.begin(), mag.begin() + n, mag.end() );
  mag.resize( n );
  std::random_shuffle( mag.begin(), mag.end() );
  n /= patchRadius;
  mag.resize( n );

  if ( type == GPC_DESCRIPTOR_DCT )
  {
    Mat fromCh[3], toCh[3];
    split( from, fromCh );
    split( to, toCh );

    Mat allDescriptors;
    flann::Index index;
    buildIndex( allDescriptors, index, toCh, getAllDCTDescriptorsForImage );

    for ( size_t k = 0; k < n; ++k )
      getTriplet( mag[k], gt, fromCh, toCh, samples, index, getDCTPatchDescriptor );
  }
  else if ( type == GPC_DESCRIPTOR_WHT )
  {
    Mat fromCh[3], toCh[3], fromChInt[3], toChInt[3];
    split( from, fromCh );
    split( to, toCh );
    integral( fromCh[0], fromChInt[0], CV_64F );
    integral( fromCh[1], fromChInt[1], CV_64F );
    integral( fromCh[2], fromChInt[2], CV_64F );
    integral( toCh[0], toChInt[0], CV_64F );
    integral( toCh[1], toChInt[1], CV_64F );
    integral( toCh[2], toChInt[2], CV_64F );

    Mat allDescriptors;
    flann::Index index;
    buildIndex( allDescriptors, index, toCh, getAllWHTDescriptorsForImage );

    for ( size_t k = 0; k < n; ++k )
      getTriplet( mag[k], gt, fromChInt, toChInt, samples, index, getWHTPatchDescriptor );
  }
  else
    CV_Error( CV_StsBadArg, "Unknown descriptor type" );
}

/* Sample random number from Cauchy distribution. */
double getRandomCauchyScalar()
{
  return tan( rng.uniform( -1.54, 1.54 ) ); // I intentionally used the value slightly less than PI/2 to enforce strictly
                                            // zero probability for large numbers. Resulting PDF for Cauchy has
                                            // truncated "tails".
}

/* Sample random vector from Cauchy distribution (pointwise, i.e. vector whose components are independent random
 * variables from Cauchy distribution) */
void getRandomCauchyVector( Vec< double, GPCPatchDescriptor::nFeatures > &v )
{
  for ( unsigned i = 0; i < GPCPatchDescriptor::nFeatures; ++i )
    v[i] = getRandomCauchyScalar();
}

double getRobustMedian( double m ) { return m < 0 ? m * ( 1.0 + epsTolerance ) : m * ( 1.0 - epsTolerance ); }
}

double GPCPatchDescriptor::dot( const Vec< double, nFeatures > &coef ) const
{
#if CV_SIMD128_64F
  v_float64x2 sum = v_setzero_f64();
  for ( unsigned i = 0; i < nFeatures; i += 2 )
  {
    v_float64x2 x = v_load( &feature.val[i] );
    v_float64x2 y = v_load( &coef.val[i] );
    sum = v_muladd( x, y, sum );
  }
#if CV_SSE2
  __m128d sumrev = _mm_shuffle_pd( sum.val, sum.val, _MM_SHUFFLE2( 0, 1 ) );
  return _mm_cvtsd_f64( _mm_add_pd( sum.val, sumrev ) );
#else
  double CV_DECL_ALIGNED( 16 ) buf[2];
  v_store_aligned( buf, sum );
  return OPENCV_HAL_ADD( buf[0], buf[1] );
#endif

#else
  return feature.dot( coef );
#endif
}

void GPCPatchSample::getDirections( bool &refdir, bool &posdir, bool &negdir, const Vec< double, GPCPatchDescriptor::nFeatures > &coef, double rhs ) const
{
  refdir = ( ref.dot( coef ) < rhs );
  posdir = pos.isSeparated() ? ( !refdir ) : ( pos.dot( coef ) < rhs );
  negdir = neg.isSeparated() ? ( !refdir ) : ( neg.dot( coef ) < rhs );
}

void GPCDetails::getAllDescriptorsForImage( const Mat *imgCh, std::vector< GPCPatchDescriptor > &descr, const GPCMatchingParams &mp,
                                            int type )
{
  if ( type == GPC_DESCRIPTOR_DCT )
    getAllDCTDescriptorsForImage( imgCh, descr, mp );
  else if ( type == GPC_DESCRIPTOR_WHT )
    getAllWHTDescriptorsForImage( imgCh, descr, mp );
  else
    CV_Error( CV_StsBadArg, "Unknown descriptor type" );
}

void GPCDetails::getCoordinatesFromIndex( size_t index, Size sz, int &x, int &y )
{
  const size_t stride = sz.width - patchRadius * 2;
  y = int( index / stride );
  x = int( index - y * stride + patchRadius );
  y += patchRadius;
}

bool GPCTree::trainNode( size_t nodeId, SIter begin, SIter end, unsigned depth )
{
  const int nSamples = (int)std::distance( begin, end );

  if ( nSamples < params.minNumberOfSamples || depth >= params.maxTreeDepth )
    return false;

  if ( nodeId >= nodes.size() )
    nodes.resize( nodeId + 1 );

  Node &node = nodes[nodeId];

  // Select the best hyperplane
  unsigned globalBestScore = 0;
  std::vector< double > values;
  values.reserve( nSamples * 2 );

  for ( int j = 0; j < globalIters; ++j )
  { // Global search step
    Vec< double, GPCPatchDescriptor::nFeatures > coef;
    unsigned localBestScore = 0;
    getRandomCauchyVector( coef );

    for ( int i = 0; i < localIters; ++i )
    { // Local search step
      double randomModification = getRandomCauchyScalar() * ( 1.0 + sigmaGrowthRate * int( i / GPCPatchDescriptor::nFeatures ) );
      const int pos = i % GPCPatchDescriptor::nFeatures;
      std::swap( coef[pos], randomModification );
      values.clear();

      for ( SIter iter = begin; iter != end; ++iter )
        values.push_back( iter->ref.dot( coef ) );

      std::nth_element( values.begin(), values.begin() + nSamples / 2, values.end() );
      double median = values[nSamples / 2];

      // Skip obviously malformed division. This may happen in case there are a large number of equal samples.
      // Most likely this won't happen with samples collected from a good dataset.
      // Happens in case dataset contains plain (or close to plain) images.
      if ( std::count_if( values.begin(), values.end(), CompareWithTolerance( median ) ) > std::max( 1, nSamples / 4 ) )
        continue;

      median = getRobustMedian( median );

      unsigned score = 0;
      for ( SIter iter = begin; iter != end; ++iter )
      {
        bool refdir, posdir, negdir;
        iter->getDirections( refdir, posdir, negdir, coef, median );
        if ( refdir == posdir )
          score += scoreGainPos;
        if ( refdir != negdir )
          score += scoreGainNeg;
      }

      if ( score > localBestScore )
        localBestScore = score;
      else
      {
        const double beta = simulatedAnnealingTemperatureCoef * std::sqrt( i ) / ( nSamples * ( scoreGainPos + scoreGainNeg ) );
        if ( rng.uniform( 0.0, 1.0 ) > std::exp( -beta * ( localBestScore - score) ) )
          coef[pos] = randomModification;
      }

      if ( score > globalBestScore )
      {
        globalBestScore = score;
        node.coef = coef;
        node.rhs = median;
      }
    }
  }

  if ( globalBestScore == 0 )
    return false;

  if ( params.printProgress )
  {
    const int maxScore = nSamples * ( scoreGainPos + scoreGainNeg );
    const double correctRatio = double( globalBestScore ) / maxScore;
    printf( "[%u] Correct %.2f (%u/%d)\nWeights:", depth, correctRatio, globalBestScore, maxScore );
    for ( unsigned k = 0; k < GPCPatchDescriptor::nFeatures; ++k )
      printf( " %.3f", node.coef[k] );
    printf( "\n" );
  }

  for ( SIter iter = begin; iter != end; ++iter )
  {
    bool refdir, posdir, negdir;
    iter->getDirections( refdir, posdir, negdir, node.coef, node.rhs );
    // We shouldn't account for positive sample in the scoring in case it was separated before. So mark it as separated.
    // After all, we can't bring back samples which were separated from reference on early levels.
    if ( refdir != posdir )
      iter->pos.markAsSeparated();
    // The same for negative sample.
    if ( refdir != negdir )
      iter->neg.markAsSeparated();
    // If both positive and negative were separated before then such triplet doesn't make sense on deeper levels. We discard it.
  }

  // Partition vector with samples according to the hyperplane in QuickSort-like manner.
  // Unlike QuickSort, we need to partition it into 3 parts (left subtree samples; undefined samples; right subtree
  // samples), so we call it two times.
  SIter leftEnd = std::partition( begin, end, PartitionPredicate1( node.coef, node.rhs ) ); // Separate left subtree samples from others.
  SIter rightBegin =
    std::partition( leftEnd, end, PartitionPredicate2( node.coef, node.rhs ) ); // Separate undefined samples from right subtree samples.

  node.left = ( trainNode( nodeId * 2 + 1, begin, leftEnd, depth + 1 ) ) ? unsigned( nodeId * 2 + 1 ) : 0;
  node.right = ( trainNode( nodeId * 2 + 2, rightBegin, end, depth + 1 ) ) ? unsigned( nodeId * 2 + 2 ) : 0;

  return true;
}

void GPCTree::train( GPCTrainingSamples &samples, const GPCTrainingParams _params )
{
  if ( _params.descriptorType != samples.type() )
    CV_Error( CV_StsBadArg, "Descriptor type mismatch! Check that samples are collected with the same descriptor type." );
  nodes.clear();
  nodes.reserve( samples.size() * 2 - 1 ); // set upper bound for the possible number of nodes so all subsequent resize() will be no-op
  params = _params;
  GPCSamplesVector &sv = samples;
  trainNode( 0, sv.begin(), sv.end(), 0 );
}

void GPCTree::write( FileStorage &fs ) const
{
  if ( nodes.empty() )
    CV_Error( CV_StsBadArg, "Tree have not been trained" );
  fs << "nodes" << nodes;
  fs << "dtype" << (int)params.descriptorType;
}

void GPCTree::read( const FileNode &fn )
{
  fn["nodes"] >> nodes;
  fn["dtype"] >> (int &)params.descriptorType;
}

unsigned GPCTree::findLeafForPatch( const GPCPatchDescriptor &descr ) const
{
  unsigned id = 0, prevId;
  do
  {
    prevId = id;
    if ( descr.dot( nodes[id].coef ) < nodes[id].rhs )
      id = nodes[id].right;
    else
      id = nodes[id].left;
  } while ( id );
  return prevId;
}

Ptr< GPCTrainingSamples > GPCTrainingSamples::create( const std::vector< String > &imagesFrom, const std::vector< String > &imagesTo,
                                                      const std::vector< String > &gt, int _descriptorType )
{
  CV_Assert( imagesFrom.size() == imagesTo.size() );
  CV_Assert( imagesFrom.size() == gt.size() );

  Ptr< GPCTrainingSamples > ts = makePtr< GPCTrainingSamples >();

  ts->descriptorType = _descriptorType;

  for ( size_t i = 0; i < imagesFrom.size(); ++i )
  {
    Mat from = imread( imagesFrom[i] );
    Mat to = imread( imagesTo[i] );
    Mat gtFlow = readOpticalFlow( gt[i] );

    CV_Assert( from.size == to.size );
    CV_Assert( from.size == gtFlow.size );
    CV_Assert( from.channels() == 3 );
    CV_Assert( to.channels() == 3 );

    from.convertTo( from, CV_32FC3 );
    to.convertTo( to, CV_32FC3 );
    cvtColor( from, from, COLOR_BGR2YCrCb );
    cvtColor( to, to, COLOR_BGR2YCrCb );

    getTrainingSamples( from, to, gtFlow, ts->samples, ts->descriptorType );
  }

  return ts;
}

Ptr< GPCTrainingSamples > GPCTrainingSamples::create( InputArrayOfArrays imagesFrom, InputArrayOfArrays imagesTo,
                                                      InputArrayOfArrays gt, int _descriptorType )
{
  CV_Assert( imagesFrom.total() == imagesTo.total() );
  CV_Assert( imagesFrom.total() == gt.total() );

  Ptr< GPCTrainingSamples > ts = makePtr< GPCTrainingSamples >();

  ts->descriptorType = _descriptorType;

  for ( size_t i = 0; i < imagesFrom.total(); ++i )
  {
    Mat from = imagesFrom.getMat( static_cast<int>( i ) );
    Mat to = imagesTo.getMat( static_cast<int>( i ) );
    Mat gtFlow = gt.getMat( static_cast<int>( i ) );

    CV_Assert( from.size == to.size );
    CV_Assert( from.size == gtFlow.size );
    CV_Assert( from.channels() == 3 );
    CV_Assert( to.channels() == 3 );

    from.convertTo( from, CV_32FC3 );
    to.convertTo( to, CV_32FC3 );
    cvtColor( from, from, COLOR_BGR2YCrCb );
    cvtColor( to, to, COLOR_BGR2YCrCb );

    getTrainingSamples( from, to, gtFlow, ts->samples, ts->descriptorType );
  }

  return ts;
}

void GPCDetails::dropOutliers( std::vector< std::pair< Point2i, Point2i > > &corr )
{
  std::vector< float > mag( corr.size() );

  for ( size_t i = 0; i < corr.size(); ++i )
    mag[i] = normL2Sqr( corr[i].first - corr[i].second );

  const size_t threshold = size_t( mag.size() * thresholdOutliers );
  std::nth_element( mag.begin(), mag.begin() + threshold, mag.end() );
  const float percentile = mag[threshold];
  size_t i = 0, j = 0;

  while ( i < corr.size() )
  {
    if ( normL2Sqr( corr[i].first - corr[i].second ) <= percentile )
    {
      corr[j] = corr[i];
      ++j;
    }
    ++i;
  }

  corr.resize( j );
}

} // namespace optflow

void write( FileStorage &fs, const String &name, const optflow::GPCTree::Node &node )
{
  cv::internal::WriteStructContext ws( fs, name, CV_NODE_SEQ + CV_NODE_FLOW );
  for ( unsigned i = 0; i < optflow::GPCPatchDescriptor::nFeatures; ++i )
    write( fs, node.coef[i] );
  write( fs, node.rhs );
  write( fs, (int)node.left );
  write( fs, (int)node.right );
}

void read( const FileNode &fn, optflow::GPCTree::Node &node, optflow::GPCTree::Node )
{
  FileNodeIterator it = fn.begin();
  for ( unsigned i = 0; i < optflow::GPCPatchDescriptor::nFeatures; ++i )
    it >> node.coef[i];
  it >> node.rhs >> (int &)node.left >> (int &)node.right;
}

} // namespace cv
