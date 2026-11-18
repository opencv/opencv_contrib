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

#include "test_precomp.hpp"

namespace opencv_test { namespace {

class CV_BinaryDescriptorMatcherTest : public cvtest::BaseTest
{
 public:
  CV_BinaryDescriptorMatcherTest( float _badPart ) :
      badPart( _badPart )
  {
    dmatcher = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
  }

 protected:
  static const int dim = 32;
  static const int queryDescCount = 300;  // must be even number because we split train data in some cases in two
  static const int countFactor = 4;  // do not change it
  const float badPart;

  virtual void run( int );
  void generateData( Mat& query, Mat& train );
  uchar invertSingleBits( uchar dividend_char, int numBits );
  void emptyDataTest();
  void matchTest( const Mat& query, const Mat& train );
  void knnMatchTest( const Mat& query, const Mat& train );
  void radiusMatchTest( const Mat& query, const Mat& train );

  std::string name;
  Ptr<BinaryDescriptorMatcher> dmatcher;

 private:
  CV_BinaryDescriptorMatcherTest& operator=( const CV_BinaryDescriptorMatcherTest& )
  {
    return *this;
  }

};

/* invert numBits bits in input char */
uchar CV_BinaryDescriptorMatcherTest::invertSingleBits( uchar dividend_char, int numBits )
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

void CV_BinaryDescriptorMatcherTest::emptyDataTest()
{
  Mat queryDescriptors, trainDescriptors, mask;
  std::vector<Mat> trainDescriptorCollection, masks;
  std::vector<DMatch> matches;
  std::vector<std::vector<DMatch> > vmatches;

  try
  {
    dmatcher->match( queryDescriptors, trainDescriptors, matches, mask );
  }

  catch ( ... )
  {
    ts->printf( cvtest::TS::LOG, "match() on empty descriptors must not generate exception (1).\n" );
    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
  }

  try
  {
    dmatcher->knnMatch( queryDescriptors, trainDescriptors, vmatches, 2, mask );
  }

  catch ( ... )
  {
    ts->printf( cvtest::TS::LOG, "knnMatch() on empty descriptors must not generate exception (1).\n" );
    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
  }

  try
  {
    dmatcher->radiusMatch( queryDescriptors, trainDescriptors, vmatches, 10.f, mask );
  }

  catch ( ... )
  {
    ts->printf( cvtest::TS::LOG, "radiusMatch() on empty descriptors must not generate exception (1).\n" );
    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
  }

  try
  {
    dmatcher->add( trainDescriptorCollection );
  }

  catch ( ... )
  {
    ts->printf( cvtest::TS::LOG, "add() on empty descriptors must not generate exception.\n" );
    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
  }

  try
  {
    dmatcher->match( queryDescriptors, matches, masks );
  }

  catch ( ... )
  {
    ts->printf( cvtest::TS::LOG, "match() on empty descriptors must not generate exception (2).\n" );
    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
  }

  try
  {
    dmatcher->knnMatch( queryDescriptors, vmatches, 2, masks );
  }

  catch ( ... )
  {
    ts->printf( cvtest::TS::LOG, "knnMatch() on empty descriptors must not generate exception (2).\n" );
    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
  }

  try
  {
    dmatcher->radiusMatch( queryDescriptors, vmatches, 10.f, masks );
  }

  catch ( ... )
  {
    ts->printf( cvtest::TS::LOG, "radiusMatch() on empty descriptors must not generate exception (2).\n" );
    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
  }

}

void CV_BinaryDescriptorMatcherTest::generateData( Mat& query, Mat& train )
{
  RNG& rng = theRNG();

  /* Generate query descriptors randomly.
   Descriptor vector elements are binary values. */
  Mat buf( queryDescCount, dim, CV_8UC1 );
  rng.fill( buf, RNG::UNIFORM, Scalar( 0 ), Scalar( 255 ) );
  buf.convertTo( query, CV_8UC1 );

  for ( int i = 0; i < query.rows; i++ )
  {
    for ( int j = 0; j < countFactor; j++ )
    {
      train.push_back( query.row( i ) );
      int randCol = rand() % 32;
      uchar u = query.at<uchar>( i, randCol );
      uchar modified_u = invertSingleBits( u, j + 1 );
      train.at<uchar>( i * countFactor + j, randCol ) = modified_u;
    }
  }
}

void CV_BinaryDescriptorMatcherTest::matchTest( const Mat& query, const Mat& train )
{
  dmatcher->clear();

  // test const version of match()
  {
    std::vector<DMatch> matches;
    dmatcher->match( query, train, matches );

    if( (int) matches.size() != queryDescCount )
    {
      ts->printf( cvtest::TS::LOG, "Incorrect matches count while test match() function (1).\n" );
      ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

    else
    {
      int badCount = 0;
      for ( size_t i = 0; i < matches.size(); i++ )
      {
        DMatch& match = matches[i];
        if( ( match.queryIdx != (int) i ) || ( match.trainIdx != (int) i * countFactor ) || ( match.imgIdx != 0 ) )
          badCount++;
      }
      if( (float) badCount > (float) queryDescCount * badPart )
      {
        ts->printf( cvtest::TS::LOG, "%f - too large bad matches part while test match() function (1).\n",
                    (float) badCount / (float) queryDescCount );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
      }
    }
  }

  // test const version of match() for the same query and test descriptors
  {
    std::vector<DMatch> matches;
    dmatcher->match( query, query, matches );

    if( (int) matches.size() != query.rows )
    {
      ts->printf( cvtest::TS::LOG, "Incorrect matches count while test match() function for the same query and test descriptors (1).\n" );
      ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }
    else
    {
      for ( size_t i = 0; i < matches.size(); i++ )
      {
        DMatch& match = matches[i];
        if( match.queryIdx != (int) i || match.trainIdx != (int) i || std::abs( match.distance ) > FLT_EPSILON )
        {
          ts->printf(
              cvtest::TS::LOG,
              "Bad match (i=%d, queryIdx=%d, trainIdx=%d, distance=%f) while test match() function for the same query and test descriptors (1).\n", i,
              match.queryIdx, match.trainIdx, match.distance );
          ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        }
      }
    }
  }

  // test version of match() with add()
  {
    dmatcher->clear();
    std::vector<DMatch> matches;

    // make add() twice to test such case
    dmatcher->add( std::vector<Mat>( 1, train.rowRange( 0, train.rows / 2 ) ) );
    dmatcher->add( std::vector<Mat>( 1, train.rowRange( train.rows / 2, train.rows ) ) );

    // prepare masks (make first nearest match illegal)
    std::vector<Mat> masks( 2 );
    for ( int mi = 0; mi < 2; mi++ )
      masks[mi] = Mat::ones( query.rows, 1/*train.rows / 2*/, CV_8UC1 );

    dmatcher->match( query, matches, masks );
    if( (int) matches.size() != queryDescCount )
    {
      ts->printf( cvtest::TS::LOG, "Incorrect matches count while test match() function (2).\n" );
      ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

    else
    {
      int badCount = 0;
      for ( size_t i = 0; i < matches.size(); i++ )
      {
        DMatch& match = matches[i];

        if( ( match.queryIdx != (int) i ) || ( match.trainIdx != (int) i * countFactor /*+ shift*/) || ( match.imgIdx > 1 ) )
          badCount++;
      }

      if( (float) badCount > (float) queryDescCount * badPart )
      {
        ts->printf( cvtest::TS::LOG, "%f - too large bad matches part while test match() function (2).\n",
                    (float) badCount / (float) queryDescCount );
        ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
      }
    }
  }
}

void CV_BinaryDescriptorMatcherTest::knnMatchTest( const Mat& query, const Mat& train )
{
  dmatcher->clear();

  // test const version of knnMatch()
  {
    const int knn = 3;

    std::vector<std::vector<DMatch> > matches;
    dmatcher->knnMatch( query, train, matches, knn );

    if( (int) matches.size() != queryDescCount )
    {
      ts->printf( cvtest::TS::LOG, "Incorrect matches count while test knnMatch() function (1).\n" );
      ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

    else
    {
      int badCount = 0;
      for ( size_t i = 0; i < matches.size(); i++ )
      {
        if( (int) matches[i].size() != knn )
          badCount++;

        else
        {
          int localBadCount = 0;
          for ( int k = 0; k < knn; k++ )
          {
            DMatch& match = matches[i][k];
            if( ( match.queryIdx != (int) i ) || ( match.trainIdx != (int) i * countFactor + k ) || ( match.imgIdx != 0 ) )
              localBadCount++;
          }
          badCount += localBadCount > 0 ? 1 : 0;
        }

      }

      if( (float) badCount > (float) queryDescCount * badPart )
      {
        ts->printf( cvtest::TS::LOG, "%f - too large bad matches part while test knnMatch() function (1).\n",
                    (float) badCount / (float) queryDescCount );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
      }
    }
  }

//  // test version of knnMatch() with add()
  {
    const int knn = 2;
    std::vector<std::vector<DMatch> > matches;

    // make add() twice to test such case
    dmatcher->add( std::vector<Mat>( 1, train.rowRange( 0, train.rows / 2 ) ) );
    dmatcher->add( std::vector<Mat>( 1, train.rowRange( train.rows / 2, train.rows ) ) );

    // prepare masks (make first nearest match illegal)
    std::vector<Mat> masks( 2 );
    for ( int mi = 0; mi < 2; mi++ )
    {
      masks[mi] = Mat::ones( query.rows, 1, CV_8UC1 );
    }

    dmatcher->knnMatch( query, matches, knn, masks );

    if( (int) matches.size() != queryDescCount )
    {
      ts->printf( cvtest::TS::LOG, "Incorrect matches count while test knnMatch() function (2).\n" );
      ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

    else
    {
      int badCount = 0;
      for ( size_t i = 0; i < matches.size(); i++ )
      {
        if( (int) matches[i].size() != knn )
          badCount++;

        else
        {
          int localBadCount = 0;
          for ( int k = 0; k < knn; k++ )
          {
            DMatch& match = matches[i][k];
            {
              if( i < queryDescCount / 2 )
              {
                if( ( match.queryIdx != (int) i ) || ( match.trainIdx != (int) i * countFactor + k ) || ( match.imgIdx != 0 ) )
                  localBadCount++;
              }

              else
              {
                if( ( match.queryIdx != (int) i ) || ( match.trainIdx != (int) i * countFactor + k ) || ( match.imgIdx != 1 ) )
                  localBadCount++;
              }
            }
          }

          badCount += localBadCount > 0 ? 1 : 0;
        }
      }

      if( (float) badCount > (float) queryDescCount * badPart )
      {
        ts->printf( cvtest::TS::LOG, "%f - too large bad matches part while test knnMatch() function (2).\n",
                    (float) badCount / (float) queryDescCount );
        ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
      }
    }
  }
}

void CV_BinaryDescriptorMatcherTest::radiusMatchTest( const Mat& query, const Mat& train )
{
  dmatcher->clear();
  // test const version of match()
  {
    const float radius = 1;
    std::vector<std::vector<DMatch> > matches;
    dmatcher->radiusMatch( query, train, matches, radius );

    if( (int) matches.size() != queryDescCount )
    {
      ts->printf( cvtest::TS::LOG, "Incorrect matches count while test radiusMatch() function (1).\n" );
      ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }
    else
    {
      int badCount = 0;
      for ( size_t i = 0; i < matches.size(); i++ )
      {

        if( (int) matches[i].size() != 1 )
        {
          badCount++;
        }

        else
        {
          DMatch& match = matches[i][0];
          if( ( match.queryIdx != (int) i ) || ( match.trainIdx != (int) i * countFactor ) || ( match.imgIdx != 0 ) )
            badCount++;
        }
      }

      if( (float) badCount > (float) queryDescCount * badPart )
      {
        ts->printf( cvtest::TS::LOG, "%f - too large bad matches part while test radiusMatch() function (1).\n",
                    (float) badCount / (float) queryDescCount );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
      }
    }
  }

  {
    const float radius = 3;
    std::vector<std::vector<DMatch> > matches;
    // make add() twice to test such case
    dmatcher->add( std::vector<Mat>( 1, train.rowRange( 0, train.rows / 2 ) ) );
    dmatcher->add( std::vector<Mat>( 1, train.rowRange( train.rows / 2, train.rows ) ) );

    // prepare masks
    std::vector<Mat> masks( 2 );
    for ( int mi = 0; mi < 2; mi++ )
      masks[mi] = Mat::ones( query.rows, 1, CV_8UC1 );

    dmatcher->radiusMatch( query, matches, radius, masks );

    //int curRes = cvtest::TS::OK;
    if( (int) matches.size() != queryDescCount )
    {
      ts->printf( cvtest::TS::LOG, "Incorrect matches count while test radiusMatch() function (1).\n" );
      ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    }

    int badCount = 0;
    for ( size_t i = 0; i < matches.size(); i++ )
    {
      if( (int) matches[i].size() != radius )
        badCount++;

      else
      {
        int localBadCount = 0;
        for ( int k = 0; k < radius; k++ )
        {
          DMatch& match = matches[i][k];
          {
            if( i < queryDescCount / 2 )
            {
              if( ( match.queryIdx != (int) i ) || ( match.trainIdx != (int) i * countFactor + k ) || ( match.imgIdx != 0 ) )
                localBadCount++;
            }

            else
            {
              if( ( match.queryIdx != (int) i ) || ( match.trainIdx != (int) i * countFactor + k ) || ( match.imgIdx != 1 ) )
                localBadCount++;
            }
          }
        }

        badCount += localBadCount > 0 ? 1 : 0;
      }
    }

    if( (float) badCount > (float) queryDescCount * badPart )
    {
      //curRes = cvtest::TS::FAIL_INVALID_OUTPUT;
      ts->printf( cvtest::TS::LOG, "%f - too large bad matches part while test radiusMatch() function (2).\n",
                  (float) badCount / (float) queryDescCount );
      ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
    }
  }
}

void CV_BinaryDescriptorMatcherTest::run( int )
{
  Mat query, train;
  emptyDataTest();
  generateData( query, train );
  matchTest( query, train );
  knnMatchTest( query, train );
  radiusMatchTest( query, train );
}

/****************************************************************************************\
*                                Tests registrations                                     *
 \****************************************************************************************/

TEST( BinaryDescriptor_Matcher, regression)
{
  CV_BinaryDescriptorMatcherTest test( 0.01f );
  test.safe_run();
}

static void sortMatchesByTrainIdx( std::vector<std::vector<DMatch> >& m )
{
  for ( size_t i = 0; i < m.size(); i++ )
  {
    std::sort( m[i].begin(), m[i].end(), []( const DMatch& a, const DMatch& b ) {
      if( a.imgIdx != b.imgIdx )
        return a.imgIdx < b.imgIdx;
      return a.trainIdx < b.trainIdx;
    } );
  }
}

TEST( BinaryDescriptor_Matcher, radiusMatch_dataset_after_train )
{
  Mat query( 40, 32, CV_8UC1 ), train( 35, 32, CV_8UC1 );
  theRNG().fill( query, RNG::UNIFORM, 0, 256 );
  theRNG().fill( train, RNG::UNIFORM, 0, 256 );

  Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
  std::vector<std::vector<DMatch> > expected, first, second;
  bdm->radiusMatch( query, train, expected, 120.f );

  // radiusMatch() calls train() again; nothing new was added, so the dataset must stay usable
  bdm->add( std::vector<Mat>( 1, train ) );
  bdm->train();
  bdm->radiusMatch( query, first, 120.f );
  bdm->radiusMatch( query, second, 120.f );

  sortMatchesByTrainIdx( expected );
  sortMatchesByTrainIdx( first );
  sortMatchesByTrainIdx( second );
  ASSERT_EQ( expected.size(), static_cast<size_t>( query.rows ) );
  ASSERT_EQ( first.size(), expected.size() );
  ASSERT_EQ( second.size(), expected.size() );
  size_t total = 0;
  for ( size_t i = 0; i < expected.size(); i++ )
  {
    ASSERT_EQ( first[i].size(), expected[i].size() ) << "query " << i;
    ASSERT_EQ( second[i].size(), expected[i].size() ) << "query " << i;
    for ( size_t j = 0; j < expected[i].size(); j++ )
    {
      EXPECT_EQ( first[i][j].trainIdx, expected[i][j].trainIdx );
      EXPECT_EQ( first[i][j].imgIdx, 0 );
      EXPECT_EQ( first[i][j].distance, expected[i][j].distance );
      EXPECT_EQ( second[i][j].trainIdx, expected[i][j].trainIdx );
      EXPECT_EQ( second[i][j].distance, expected[i][j].distance );
    }
    total += expected[i].size();
  }
  EXPECT_GT( total, 0u );
}

TEST( BinaryDescriptor_Matcher, knnMatch_more_neighbours_than_descriptors )
{
  Mat query( 40, 32, CV_8UC1 ), train( 3, 32, CV_8UC1 );
  theRNG().fill( query, RNG::UNIFORM, 0, 256 );
  theRNG().fill( train, RNG::UNIFORM, 0, 256 );
  Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

  std::vector<std::vector<DMatch> > pairMatches, datasetMatches;
  bdm->knnMatch( query, train, pairMatches, 5 );
  ASSERT_EQ( pairMatches.size(), static_cast<size_t>( query.rows ) );
  for ( size_t i = 0; i < pairMatches.size(); i++ )
    EXPECT_EQ( pairMatches[i].size(), 3u );

  bdm->add( std::vector<Mat>( 1, train ) );
  bdm->knnMatch( query, datasetMatches, 5 );
  ASSERT_EQ( datasetMatches.size(), static_cast<size_t>( query.rows ) );
  for ( size_t i = 0; i < datasetMatches.size(); i++ )
    EXPECT_EQ( datasetMatches[i].size(), 3u );
}

TEST( BinaryDescriptor_Matcher, empty_matcher )
{
  Mat query( 40, 32, CV_8UC1 );
  theRNG().fill( query, RNG::UNIFORM, 0, 256 );
  Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

  std::vector<DMatch> matches;
  std::vector<std::vector<DMatch> > kmatches, rmatches;
  bdm->match( query, matches );
  bdm->knnMatch( query, kmatches, 2 );
  bdm->radiusMatch( query, rmatches, 120.f );
  EXPECT_TRUE( matches.empty() );
  EXPECT_TRUE( kmatches.empty() );
  EXPECT_TRUE( rmatches.empty() );
}

// k = 0 asks for no neighbours; without the check Mihasher::query() writes into a zero-size result array
TEST( BinaryDescriptor_Matcher, knnMatch_zero_neighbours )
{
  Mat query( 16, 32, CV_8UC1 ), train( 40, 32, CV_8UC1 );
  theRNG().fill( query, RNG::UNIFORM, 0, 256 );
  theRNG().fill( train, RNG::UNIFORM, 0, 256 );
  Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

  std::vector<std::vector<DMatch> > kmatches;
  bdm->knnMatch( query, train, kmatches, 0 );
  EXPECT_TRUE( kmatches.empty() );

  bdm->add( std::vector<Mat>( 1, train ) );
  bdm->train();
  bdm->knnMatch( query, kmatches, 0 );
  EXPECT_TRUE( kmatches.empty() );
}

TEST( BinaryDescriptor_Matcher, reject_invalid_masks )
{
  Mat query( 64, 32, CV_8UC1 ), train( 60, 32, CV_8UC1 );
  theRNG().fill( query, RNG::UNIFORM, 0, 256 );
  theRNG().fill( train, RNG::UNIFORM, 0, 256 );
  Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

  // a mask needs one row per query and one column
  const Mat badMasks[] = { Mat::ones( 5, 1, CV_8UC1 ), Mat::ones( query.rows, train.rows, CV_8UC1 ) };
  for ( const Mat& mask : badMasks )
  {
    std::vector<DMatch> matches;
    std::vector<std::vector<DMatch> > kmatches, rmatches;
    bdm->match( query, train, matches, mask );
    bdm->knnMatch( query, train, kmatches, 2, mask );
    bdm->radiusMatch( query, train, rmatches, 120.f, mask );
    EXPECT_TRUE( matches.empty() ) << mask.size;
    EXPECT_TRUE( kmatches.empty() ) << mask.size;
    EXPECT_TRUE( rmatches.empty() ) << mask.size;
  }
}

static void expectMatchesEqual( const std::vector<DMatch>& expected, const std::vector<DMatch>& actual )
{
  ASSERT_EQ( actual.size(), expected.size() );
  for ( size_t i = 0; i < expected.size(); i++ )
  {
    EXPECT_EQ( actual[i].queryIdx, expected[i].queryIdx );
    EXPECT_EQ( actual[i].trainIdx, expected[i].trainIdx );
    EXPECT_EQ( actual[i].imgIdx, expected[i].imgIdx );
    EXPECT_EQ( actual[i].distance, expected[i].distance );
  }
}

static void expectMatchVectorsEqual( std::vector<std::vector<DMatch> > expected, std::vector<std::vector<DMatch> > actual )
{
  sortMatchesByTrainIdx( expected );
  sortMatchesByTrainIdx( actual );
  ASSERT_EQ( actual.size(), expected.size() );
  for ( size_t i = 0; i < expected.size(); i++ )
    expectMatchesEqual( expected[i], actual[i] );
}

TEST( BinaryDescriptor_Matcher, incremental_add_and_train )
{
  const int cases[][2] = { { 100, 5 }, { 5, 100 } };
  for ( int c = 0; c < 2; c++ )
  {
    const int n1 = cases[c][0], n2 = cases[c][1];
    Mat first( n1, 32, CV_8UC1 ), second( n2, 32, CV_8UC1 ), extraQuery( 10, 32, CV_8UC1 );
    theRNG().fill( first, RNG::UNIFORM, 0, 256 );
    theRNG().fill( second, RNG::UNIFORM, 0, 256 );
    theRNG().fill( extraQuery, RNG::UNIFORM, 0, 256 );

    std::vector<int> expectedBatch, expectedRow;
    Mat query;
    for ( int i = 0; i < 5; i++ )
    {
      int r = i * ( n1 / 5 );
      query.push_back( first.row( r ) );
      expectedBatch.push_back( 0 );
      expectedRow.push_back( r );
    }
    for ( int i = 0; i < 5; i++ )
    {
      int r = i * ( n2 / 5 );
      query.push_back( second.row( r ) );
      expectedBatch.push_back( 1 );
      expectedRow.push_back( n1 + r );
    }
    query.push_back( extraQuery );

    Ptr<BinaryDescriptorMatcher> refBdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    refBdm->add( std::vector<Mat>{ first, second } );
    refBdm->train();

    std::vector<DMatch> refM;
    std::vector<std::vector<DMatch> > refK, refR;
    refBdm->match( query, refM );
    refBdm->knnMatch( query, refK, 3 );
    refBdm->radiusMatch( query, refR, 120.f );

    ASSERT_EQ( refM.size(), static_cast<size_t>( query.rows ) );
    for ( size_t q = 0; q < expectedBatch.size(); q++ )
    {
      EXPECT_EQ( refM[q].imgIdx, expectedBatch[q] );
      EXPECT_EQ( refM[q].trainIdx, expectedRow[q] );
      EXPECT_EQ( refM[q].distance, 0.f );
    }

    // Explicit train() between add() calls, plus no-op train() with no pending descriptors
    Ptr<BinaryDescriptorMatcher> bdm1 = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    bdm1->add( std::vector<Mat>( 1, first ) );
    bdm1->train();
    bdm1->add( std::vector<Mat>( 1, second ) );
    bdm1->train();
    bdm1->train();

    std::vector<DMatch> m1;
    std::vector<std::vector<DMatch> > k1, r1;
    bdm1->match( query, m1 );
    bdm1->knnMatch( query, k1, 3 );
    bdm1->radiusMatch( query, r1, 120.f );
    expectMatchesEqual( refM, m1 );
    expectMatchVectorsEqual( refK, k1 );
    expectMatchVectorsEqual( refR, r1 );

    // Implicit train() triggered by match() between add() calls
    Ptr<BinaryDescriptorMatcher> bdm2 = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    bdm2->add( std::vector<Mat>( 1, first ) );
    std::vector<DMatch> warmupM;
    bdm2->match( query.rowRange( 0, 2 ), warmupM );
    EXPECT_EQ( warmupM.size(), 2u );
    bdm2->add( std::vector<Mat>( 1, second ) );

    std::vector<DMatch> m2;
    std::vector<std::vector<DMatch> > k2, r2;
    bdm2->match( query, m2 );
    bdm2->knnMatch( query, k2, 3 );
    bdm2->radiusMatch( query, r2, 120.f );
    expectMatchesEqual( refM, m2 );
    expectMatchVectorsEqual( refK, k2 );
    expectMatchVectorsEqual( refR, r2 );

    // Per-image masks across incremental batches
    std::vector<Mat> masks = { Mat::zeros( query.rows, 1, CV_8UC1 ), Mat::zeros( query.rows, 1, CV_8UC1 ) };
    for ( int q = 0; q < query.rows; q++ )
      masks[q % 2].at<uchar>( q ) = 1;
    std::vector<DMatch> refMasked, gotMasked;
    refBdm->match( query, refMasked, masks );
    bdm1->match( query, gotMasked, masks );
    expectMatchesEqual( refMasked, gotMasked );

    // Non-continuous ROI submatrices (64-col parent sliced to 32 cols) across train()
    Mat wideFirst( first.rows, 64, CV_8UC1 ), wideSecond( second.rows, 64, CV_8UC1 ), wideQuery( query.rows, 64, CV_8UC1 );
    theRNG().fill( wideFirst, RNG::UNIFORM, 0, 256 );
    theRNG().fill( wideSecond, RNG::UNIFORM, 0, 256 );
    theRNG().fill( wideQuery, RNG::UNIFORM, 0, 256 );
    first.copyTo( wideFirst.colRange( 0, 32 ) );
    second.copyTo( wideSecond.colRange( 0, 32 ) );
    query.copyTo( wideQuery.colRange( 0, 32 ) );
    Ptr<BinaryDescriptorMatcher> roiBdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    roiBdm->add( std::vector<Mat>( 1, wideFirst.colRange( 0, 32 ) ) );
    roiBdm->train();
    roiBdm->add( std::vector<Mat>( 1, wideSecond.colRange( 0, 32 ) ) );
    std::vector<DMatch> roiM;
    roiBdm->match( wideQuery.colRange( 0, 32 ), roiM );
    expectMatchesEqual( refM, roiM );
  }

  // Empty (0x32 and 0x0) images before and between non-empty batches preserve imgIdx and masks
  {
    Mat batchA( 5, 32, CV_8UC1 ), batchB( 5, 32, CV_8UC1 );
    theRNG().fill( batchA, RNG::UNIFORM, 0, 256 );
    theRNG().fill( batchB, RNG::UNIFORM, 0, 256 );
    Mat empty32( 0, 32, CV_8UC1 ), empty0;

    Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    bdm->add( std::vector<Mat>( 1, empty32 ) ); // imgIdx 0
    bdm->add( std::vector<Mat>( 1, batchA ) );  // imgIdx 1
    bdm->train();
    bdm->add( std::vector<Mat>( 1, empty0 ) );  // imgIdx 2
    bdm->train();
    bdm->add( std::vector<Mat>( 1, batchB ) );  // imgIdx 3
    bdm->train();

    std::vector<DMatch> matchesA, matchesB;
    bdm->match( batchA, matchesA );
    bdm->match( batchB, matchesB );
    ASSERT_EQ( matchesA.size(), 5u );
    ASSERT_EQ( matchesB.size(), 5u );
    for ( size_t i = 0; i < 5; i++ )
    {
      EXPECT_EQ( matchesA[i].imgIdx, 1 );
      EXPECT_EQ( matchesA[i].trainIdx, static_cast<int>( i ) );
      EXPECT_EQ( matchesA[i].distance, 0.f );
      EXPECT_EQ( matchesB[i].imgIdx, 3 );
      EXPECT_EQ( matchesB[i].trainIdx, static_cast<int>( 5 + i ) );
      EXPECT_EQ( matchesB[i].distance, 0.f );
    }

    std::vector<Mat> masks = {
      Mat::zeros( 5, 1, CV_8UC1 ), Mat::ones( 5, 1, CV_8UC1 ),
      Mat::zeros( 5, 1, CV_8UC1 ), Mat::ones( 5, 1, CV_8UC1 )
    };
    std::vector<DMatch> maskedA, maskedB;
    bdm->match( batchA, maskedA, masks );
    bdm->match( batchB, maskedB, masks );
    expectMatchesEqual( matchesA, maskedA );
    expectMatchesEqual( matchesB, maskedB );
  }
}

TEST( BinaryDescriptor_Matcher, reject_invalid_descriptor_width_and_type )
{
  Mat q32( 10, 32, CV_8UC1 ), t32( 10, 32, CV_8UC1 );
  theRNG().fill( q32, RNG::UNIFORM, 0, 256 );
  theRNG().fill( t32, RNG::UNIFORM, 0, 256 );

  Ptr<BinaryDescriptorMatcher> trained = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
  trained->add( std::vector<Mat>( 1, t32 ) );
  trained->train();
  std::vector<DMatch> baseMatches;
  trained->match( q32, baseMatches );

  const int badWidths[] = { 8, 16, 31, 33, 64 };
  std::vector<Mat> badMats;
  for ( int w : badWidths )
    badMats.push_back( Mat::zeros( 10, w, CV_8UC1 ) );
  badMats.push_back( Mat::zeros( 0, 16, CV_8UC1 ) );
  badMats.push_back( Mat::zeros( 10, 0, CV_8UC1 ) );
  badMats.push_back( Mat::zeros( 10, 32, CV_32FC1 ) );
  badMats.push_back( Mat::zeros( 10, 32, CV_8UC2 ) );
  const int dims1 = 32;
  badMats.push_back( Mat::zeros( 1, &dims1, CV_8UC1 ) );
  const int dims3[] = { 2, 5, 32 };
  badMats.push_back( Mat::zeros( 3, dims3, CV_8UC1 ) );

  for ( const Mat& bad : badMats )
  {
    Ptr<BinaryDescriptorMatcher> fresh = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    std::vector<DMatch> m;
    std::vector<std::vector<DMatch> > mm;

    EXPECT_THROW( fresh->add( std::vector<Mat>( 1, bad ) ), cv::Exception ) << bad.size << " type=" << bad.type();
    EXPECT_THROW( fresh->match( bad, t32, m ), cv::Exception ) << bad.size << " type=" << bad.type();
    EXPECT_THROW( fresh->match( q32, bad, m ), cv::Exception ) << bad.size << " type=" << bad.type();
    EXPECT_THROW( fresh->knnMatch( bad, t32, mm, 2 ), cv::Exception ) << bad.size << " type=" << bad.type();
    EXPECT_THROW( fresh->knnMatch( q32, bad, mm, 2 ), cv::Exception ) << bad.size << " type=" << bad.type();
    EXPECT_THROW( fresh->radiusMatch( bad, t32, mm, 120.f ), cv::Exception ) << bad.size << " type=" << bad.type();
    EXPECT_THROW( fresh->radiusMatch( q32, bad, mm, 120.f ), cv::Exception ) << bad.size << " type=" << bad.type();

    EXPECT_THROW( trained->match( bad, m ), cv::Exception ) << bad.size << " type=" << bad.type();
    EXPECT_THROW( trained->knnMatch( bad, mm, 2 ), cv::Exception ) << bad.size << " type=" << bad.type();
    EXPECT_THROW( trained->radiusMatch( bad, mm, 120.f ), cv::Exception ) << bad.size << " type=" << bad.type();
  }

  // Failed multi-matrix add() leaves existing dataset unchanged
  EXPECT_THROW( trained->add( std::vector<Mat>{ t32, Mat::zeros( 5, 16, CV_8UC1 ) } ), cv::Exception );
  std::vector<DMatch> afterFailedAdd;
  trained->match( q32, afterFailedAdd );
  expectMatchesEqual( baseMatches, afterFailedAdd );

  // Regression guard (passes on base): valid empty (0 x 32) CV_8UC1 descriptor matrix returns cleanly
  Mat empty32( 0, 32, CV_8UC1 );
  std::vector<DMatch> emptyM;
  trained->match( empty32, t32, emptyM );
  trained->match( empty32, emptyM );
  EXPECT_TRUE( emptyM.empty() );
}

}} // namespace
