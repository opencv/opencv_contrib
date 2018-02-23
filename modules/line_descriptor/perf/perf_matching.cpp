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
 // Copyright (C) 2014, Biagio Montesano, all rights reserved.
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

#include "perf_precomp.hpp"

namespace opencv_test { namespace {

#define QUERY_DES_COUNT  300
#define DIM  32
#define COUNT_FACTOR  4
#define RADIUS 3

void generateData( Mat& query, Mat& train );
uchar invertSingleBits( uchar dividend_char, int numBits );

/* invert numBits bits in input char */
uchar invertSingleBits( uchar dividend_char, int numBits )
{
  std::vector<int> bin_vector;
  long dividend;
  long bin_num;

  /* convert input char to a long */
  dividend = (long) dividend_char;

  /*if a 0 has been obtained, just generate a 8-bit long vector of zeros */
  if( dividend == 0 )
    bin_vector = std::vector<int>( 8, 0 );

  /* else, apply classic decimal to binary conversion */
  else
  {
    while ( dividend >= 1 )
    {
      bin_num = dividend % 2;
      dividend /= 2;
      bin_vector.push_back( bin_num );
    }
  }

  /* ensure that binary vector always has length 8 */
  if( bin_vector.size() < 8 )
  {
    std::vector<int> zeros( 8 - bin_vector.size(), 0 );
    bin_vector.insert( bin_vector.end(), zeros.begin(), zeros.end() );
  }

  /* invert numBits bits */
  for ( int index = 0; index < numBits; index++ )
  {
    if( bin_vector[index] == 0 )
      bin_vector[index] = 1;

    else
      bin_vector[index] = 0;
  }

  /* reconvert to decimal */
  uchar result = 0;
  for ( int i = (int) bin_vector.size() - 1; i >= 0; i-- )
    result += (uchar) ( bin_vector[i] * ( 1 << i ) );

  return result;
}

void generateData( Mat& query, Mat& train )
{
  RNG& rng = theRNG();

  Mat buf( QUERY_DES_COUNT, DIM, CV_8UC1 );
  rng.fill( buf, RNG::UNIFORM, Scalar( 0 ), Scalar( 255 ) );
  buf.convertTo( query, CV_8UC1 );

  for ( int i = 0; i < query.rows; i++ )
  {
    for ( int j = 0; j < COUNT_FACTOR; j++ )
    {
      train.push_back( query.row( i ) );
      int randCol = rand() % 32;
      uchar u = query.at<uchar>( i, randCol );
      uchar modified_u = invertSingleBits( u, j + 1 );
      train.at<uchar>( i * COUNT_FACTOR + j, randCol ) = modified_u;
    }
  }
}

PERF_TEST(matching, single_match)
{
  Mat query, train;
  std::vector<DMatch> dm;
  Ptr<BinaryDescriptorMatcher> bd = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

  generateData( query, train );

  TEST_CYCLE()
    bd->match( query, train, dm );

  SANITY_CHECK_NOTHING();

}

PERF_TEST(knn_matching, knn_match_distances_test)
{
  Mat query, train, distances;
  std::vector<std::vector<DMatch> > dm;
  Ptr<BinaryDescriptorMatcher> bd = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

  generateData( query, train );

  TEST_CYCLE()
  {
    bd->knnMatch( query, train, dm, QUERY_DES_COUNT );
    for ( int i = 0; i < (int) dm.size(); i++ )
    {
      for ( int j = 0; j < (int) dm[i].size(); j++ )
        distances.push_back( dm[i][j].distance );
    }
  }

  SANITY_CHECK_NOTHING();
}

PERF_TEST(radius_match, radius_match_distances_test)
{
  Mat query, train, distances;
  std::vector<std::vector<DMatch> > dm;
  Ptr<BinaryDescriptorMatcher> bd = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

  generateData( query, train );

  TEST_CYCLE()
  {
    bd->radiusMatch( query, train, dm, RADIUS );
    for ( int i = 0; i < (int) dm.size(); i++ )
    {
      for ( int j = 0; j < (int) dm[i].size(); j++ )
        distances.push_back( dm[i][j].distance );
    }
  }

  SANITY_CHECK_NOTHING();

}


}} // namespace
