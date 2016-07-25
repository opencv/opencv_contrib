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

namespace cv
{
namespace optflow
{
namespace
{

const int patchRadius = 10;
const double thresholdMagnitudeFrac = 0.6666666666;
const int globalIters = 3;
const int localIters = 500;
const int minNumberOfSamples = 2;
//const bool debugOutput = true;

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
    const bool direction1 = ( coef.dot( sample.first.feature ) < rhs );
    const bool direction2 = ( coef.dot( sample.second.feature ) < rhs );
    return direction1 == false && direction1 == direction2;
  }
};

struct PartitionPredicate2
{
  Vec< double, GPCPatchDescriptor::nFeatures > coef;
  double rhs;

  PartitionPredicate2( const Vec< double, GPCPatchDescriptor::nFeatures > &_coef, double _rhs ) : coef( _coef ), rhs( _rhs ) {}

  bool operator()( const GPCPatchSample &sample ) const
  {
    const bool direction1 = ( coef.dot( sample.first.feature ) < rhs );
    const bool direction2 = ( coef.dot( sample.second.feature ) < rhs );
    return direction1 != direction2;
  }
};

float normL2Sqr( const Vec2f &v ) { return v[0] * v[0] + v[1] * v[1]; }

bool checkBounds( int i, int j, Size sz )
{
  return i >= patchRadius && j >= patchRadius && i + patchRadius < sz.height && j + patchRadius < sz.width;
}

void getTrainingSamples( const Mat &from, const Mat &to, const Mat &gt, GPCSamplesVector &samples )
{
  const Size sz = gt.size();
  std::vector< Magnitude > mag;

  for ( int i = patchRadius; i + patchRadius < sz.height; ++i )
    for ( int j = patchRadius; j + patchRadius < sz.width; ++j )
      mag.push_back( Magnitude( normL2Sqr( gt.at< Vec2f >( i, j ) ), i, j ) );

  size_t n = size_t(mag.size() * thresholdMagnitudeFrac); // As suggested in the paper, we discard part of the training samples
                                                          // with a small displacement and train to better distinguish hard pairs.
  std::nth_element( mag.begin(), mag.begin() + n, mag.end() );
  mag.resize( n );
  std::random_shuffle( mag.begin(), mag.end() );
  n /= patchRadius;
  mag.resize( n );

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
      samples.push_back( std::make_pair( GPCPatchDescriptor( fromCh, i0, j0 ), GPCPatchDescriptor( toCh, i1, j1 ) ) );
  }
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
}

GPCPatchDescriptor::GPCPatchDescriptor( const Mat *imgCh, int i, int j )
{
  Rect roi( j - patchRadius, i - patchRadius, 2 * patchRadius, 2 * patchRadius );
  Mat freqDomain;
  dct( imgCh[0]( roi ), freqDomain );

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

bool GPCTree::trainNode( size_t nodeId, SIter begin, SIter end, unsigned depth )
{
  if ( std::distance( begin, end ) < minNumberOfSamples )
    return false;

  if ( nodeId >= nodes.size() )
    nodes.resize( nodeId + 1 );

  Node &node = nodes[nodeId];

  // Select the best hyperplane
  unsigned globalBestScore = 0;
  std::vector< double > values;

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
        values.push_back( coef.dot( iter->first.feature ) );
        values.push_back( coef.dot( iter->second.feature ) );
      }

      std::nth_element( values.begin(), values.begin() + values.size() / 2, values.end() );
      const double median = values[values.size() / 2];
      unsigned correct = 0;

      for ( SIter iter = begin; iter != end; ++iter )
      {
        const bool direction = ( coef.dot( iter->first.feature ) < median );
        if ( direction == ( coef.dot( iter->second.feature ) < median ) )
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

        /*if ( debugOutput )
        {
          printf( "[%u] Updating weights: correct %.2f (%u/%ld)\n", depth, double( correct ) / std::distance( begin, end ), correct,
                  std::distance( begin, end ) );
          for ( unsigned k = 0; k < GPCPatchDescriptor::nFeatures; ++k )
            printf( "%.3f ", coef[k] );
          printf( "\n" );
        }*/
      }
    }
  }
  // Partition vector with samples according to the hyperplane in QuickSort-like manner.
  // Unlike QuickSort, we need to partition it into 3 parts (left subtree samples; undefined samples; right subtree
  // samples), so we call it two times.
  SIter leftEnd = std::partition( begin, end, PartitionPredicate1( node.coef, node.rhs ) ); // Separate left subtree samples from others.
  SIter rightBegin =
    std::partition( leftEnd, end, PartitionPredicate2( node.coef, node.rhs ) ); // Separate undefined samples from right subtree samples.

  node.left = ( trainNode( nodeId * 2 + 1, begin, leftEnd, depth + 1 ) ) ? unsigned(nodeId * 2 + 1) : 0;
  node.right = ( trainNode( nodeId * 2 + 2, rightBegin, end, depth + 1 ) ) ? unsigned(nodeId * 2 + 2) : 0;

  return true;
}

void GPCTree::train( GPCSamplesVector &samples )
{
  nodes.reserve( samples.size() * 2 - 1 ); // set upper bound for the possible number of nodes so all subsequent resize() will be no-op
  trainNode( 0, samples.begin(), samples.end(), 0 );
}

void GPCTree::write( FileStorage &fs ) const
{
  if ( nodes.empty() )
    CV_Error( CV_StsBadArg, "Tree have not been trained" );
  fs << "nodes" << nodes;
}

void GPCTree::read( const FileNode &fn ) { fn["nodes"] >> nodes; }

Ptr< GPCTrainingSamples > GPCTrainingSamples::create( const std::vector< String > &imagesFrom, const std::vector< String > &imagesTo,
                                                      const std::vector< String > &gt )
{
  CV_Assert( imagesFrom.size() == imagesTo.size() );
  CV_Assert( imagesFrom.size() == gt.size() );

  Ptr< GPCTrainingSamples > ts = makePtr< GPCTrainingSamples >();
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

    getTrainingSamples( from, to, gtFlow, ts->samples );
  }

  return ts;
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
