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
#include "opencv2/highgui.hpp"
#include "precomp.hpp"

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
    const bool direction1 = ( sample.first.dot( coef ) < rhs );
    const bool direction2 = ( sample.second.dot( coef ) < rhs );
    return direction1 == false && direction2 == false;
  }
};

struct PartitionPredicate2
{
  Vec< double, GPCPatchDescriptor::nFeatures > coef;
  double rhs;

  PartitionPredicate2( const Vec< double, GPCPatchDescriptor::nFeatures > &_coef, double _rhs ) : coef( _coef ), rhs( _rhs ) {}

  bool operator()( const GPCPatchSample &sample ) const
  {
    const bool direction1 = ( sample.first.dot( coef ) < rhs );
    const bool direction2 = ( sample.second.dot( coef ) < rhs );
    return direction1 != direction2;
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

#define STR_EXPAND_( arg ) #arg
#define STR_EXPAND( arg ) STR_EXPAND_( arg )

ocl::ProgramSource _ocl_getDCTPatchDescriptorSource(
  "__kernel void getPatchDescriptor("
  "__global const uchar* imgCh0, int ic0step, int ic0off,"
  "__global const uchar* imgCh1, int ic1step, int ic1off,"
  "__global const uchar* imgCh2, int ic2step, int ic2off,"
  "__global uchar* out, int outstep, int outoff,"
  "const int gh, const int gw, const int PR"
  ") {"
  "const int i = get_global_id(0);"
  "const int j = get_global_id(1);"
  "if (i >= gh || j >= gw) return;"
  "__global double* desc = (__global double*)(out + (outstep * (i * gw + j) + outoff));"
  "const int patchRadius = PR * 2;"
  "float patch[" STR_EXPAND(PATCH_RADIUS_DOUBLED) "][" STR_EXPAND(PATCH_RADIUS_DOUBLED) "];"
  "for (int i0 = 0; i0 < patchRadius; ++i0) {"
  "  __global const float* ch0Row = (__global const float*)(imgCh0 + (ic0step * (i + i0) + ic0off + j * sizeof(float)));"
  "  for (int j0 = 0; j0 < patchRadius; ++j0)"
  "    patch[i0][j0] = ch0Row[j0];"
  "}"
  "const double pi = " STR_EXPAND(CV_PI) ";\n"
  "#pragma unroll\n"
  "for (int n0 = 0; n0 < 4; ++n0) {\n"
  "#pragma unroll\n"
  "  for (int n1 = 0; n1 < 4; ++n1) {"
  "    double sum = 0;"
  "    for (int i0 = 0; i0 < patchRadius; ++i0)"
  "      for (int j0 = 0; j0 < patchRadius; ++j0)"
  "        sum += patch[i0][j0] * cos(pi * (i0 + 0.5) * n0 / patchRadius) * cos(pi * (j0 + 0.5) * n1 / patchRadius);"
  "    desc[n0 * 4 + n1] = sum / PR;"
  "  }"
  "}"
  "for (int k = 0; k < 4; ++k) {"
  "  desc[k] *= " STR_EXPAND(SQRT2_INV) ";"
  "  desc[k * 4] *= " STR_EXPAND(SQRT2_INV) ";"
  "}"
  "double sum = 0;"
  "for (int i0 = 0; i0 < patchRadius; ++i0) {"
  "  __global const float* ch1Row = (__global const float*)(imgCh1 + (ic1step * (i + i0) + ic1off + j * sizeof(float)));"
  "  for (int j0 = 0; j0 < patchRadius; ++j0)"
  "    sum += ch1Row[j0];"
  "}"
  "desc[16] = sum / patchRadius;"
  "sum = 0;"
  "for (int i0 = 0; i0 < patchRadius; ++i0) {"
  "  __global const float* ch2Row = (__global const float*)(imgCh2 + (ic2step * (i + i0) + ic2off + j * sizeof(float)));"
  "  for (int j0 = 0; j0 < patchRadius; ++j0)"
  "    sum += ch2Row[j0];"
  "}"
  "desc[17] = sum / patchRadius;"
  "}" );

#undef STR_EXPAND_
#undef STR_EXPAND

void getAllDCTDescriptorsForImage( const Mat *imgCh, std::vector< GPCPatchDescriptor > &descr, const GPCMatchingParams &mp )
{
  const Size sz = imgCh[0].size();
  descr.reserve( ( sz.height - 2 * patchRadius ) * ( sz.width - 2 * patchRadius ) );

  if ( mp.useOpenCL && ocl::useOpenCL() )
  {
    ocl::Kernel kernel( "getPatchDescriptor", _ocl_getDCTPatchDescriptorSource );
    size_t globSize[] = {sz.height - 2 * patchRadius, sz.width - 2 * patchRadius};
    UMat out( globSize[0] * globSize[1], GPCPatchDescriptor::nFeatures, CV_64F );
    kernel
      .args( cv::ocl::KernelArg::ReadOnlyNoSize( imgCh[0].getUMat( ACCESS_READ ) ),
             cv::ocl::KernelArg::ReadOnlyNoSize( imgCh[1].getUMat( ACCESS_READ ) ),
             cv::ocl::KernelArg::ReadOnlyNoSize( imgCh[2].getUMat( ACCESS_READ ) ), cv::ocl::KernelArg::WriteOnlyNoSize( out ),
             (int)globSize[0], (int)globSize[1], (int)patchRadius )
      .run( 2, globSize, 0, true );
    Mat cpuOut = out.getMat( 0 );
    for ( int i = 0; i + 2 * patchRadius < sz.height; ++i )
      for ( int j = 0; j + 2 * patchRadius < sz.width; ++j )
        descr.push_back( *cpuOut.ptr< GPCPatchDescriptor >( i * globSize[1] + j ) );
    return;
  }

  for ( int i = patchRadius; i + patchRadius < sz.height; ++i )
    for ( int j = patchRadius; j + patchRadius < sz.width; ++j )
    {
      GPCPatchDescriptor pd;
      getDCTPatchDescriptor( pd, imgCh, i, j );
      descr.push_back( pd );
    }
}

void getAllWHTDescriptorsForImage( const Mat *imgCh, std::vector< GPCPatchDescriptor > &descr, const GPCMatchingParams & )
{
  const Size sz = imgCh[0].size();
  descr.reserve( ( sz.height - 2 * patchRadius ) * ( sz.width - 2 * patchRadius ) );

  Mat imgChInt[3];
  integral( imgCh[0], imgChInt[0], CV_64F );
  integral( imgCh[1], imgChInt[1], CV_64F );
  integral( imgCh[2], imgChInt[2], CV_64F );

  for ( int i = patchRadius; i + patchRadius < sz.height; ++i )
    for ( int j = patchRadius; j + patchRadius < sz.width; ++j )
    {
      GPCPatchDescriptor pd;
      getWHTPatchDescriptor( pd, imgChInt, i, j );
      descr.push_back( pd );
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

    for ( size_t k = 0; k < n; ++k )
    {
      int i0 = mag[k].i;
      int j0 = mag[k].j;
      int i1 = i0 + cvRound( gt.at< Vec2f >( i0, j0 )[1] );
      int j1 = j0 + cvRound( gt.at< Vec2f >( i0, j0 )[0] );
      if ( checkBounds( i1, j1, sz ) )
      {
        GPCPatchSample ps;
        getDCTPatchDescriptor( ps.first, fromCh, i0, j0 );
        getDCTPatchDescriptor( ps.second, toCh, i1, j1 );
        samples.push_back( ps );
      }
    }
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

    for ( size_t k = 0; k < n; ++k )
    {
      int i0 = mag[k].i;
      int j0 = mag[k].j;
      int i1 = i0 + cvRound( gt.at< Vec2f >( i0, j0 )[1] );
      int j1 = j0 + cvRound( gt.at< Vec2f >( i0, j0 )[0] );
      if ( checkBounds( i1, j1, sz ) )
      {
        GPCPatchSample ps;
        getWHTPatchDescriptor( ps.first, fromChInt, i0, j0 );
        getWHTPatchDescriptor( ps.second, toChInt, i1, j1 );
        samples.push_back( ps );
      }
    }
  }
  else
    CV_Error( CV_StsBadArg, "Unknown descriptor type" );
}

/* Sample random number from Cauchy distribution. */
double getRandomCauchyScalar()
{
  static RNG rng;
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

double getRobustMedian( double m )
{
  return m < 0 ? m * ( 1.0 + epsTolerance ) : m * ( 1.0 - epsTolerance );
}
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
      double randomModification = getRandomCauchyScalar();
      const int pos = i % GPCPatchDescriptor::nFeatures;
      std::swap( coef[pos], randomModification );
      values.clear();

      for ( SIter iter = begin; iter != end; ++iter )
      {
        values.push_back( iter->first.dot( coef ) );
        values.push_back( iter->second.dot( coef ) );
      }

      std::nth_element( values.begin(), values.begin() + ( nSamples + ( nSamples & 1 ) ), values.end() );
      const double median = getRobustMedian( values[nSamples + ( nSamples & 1 )] );
      unsigned correct = 0;

      // Skip obviously malformed division. This may happen in case there are a large number of equal samples.
      // Most likely this won't happen with samples collected from a good dataset.
      // Happens in case dataset contains plain (or close to plain) images.
      if ( std::count( values.begin(), values.end(), median ) > std::max( 1, nSamples / 8 ) )
        continue;

      for ( SIter iter = begin; iter != end; ++iter )
      {
        const bool direction = ( iter->first.dot( coef ) < median );
        if ( direction == ( iter->second.dot( coef ) < median ) )
          ++correct;
      }

      if ( correct > localBestScore )
        localBestScore = correct;
      else
        coef[pos] = randomModification;

      if ( correct > globalBestScore )
      {
        globalBestScore = correct;
        node.coef = coef;
        node.rhs = median;
      }
    }
  }

  if ( globalBestScore == 0 )
    return false;

  if ( params.printProgress )
  {
    printf( "[%u] Correct %.2f (%u/%d)\nWeights:", depth, double( globalBestScore ) / nSamples, globalBestScore, nSamples );
    for ( unsigned k = 0; k < GPCPatchDescriptor::nFeatures; ++k )
      printf( " %.3f", node.coef[k] );
    printf( "\n" );
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
