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
  Mat query( 40, 32, CV_8UC1 ), train0( 35, 32, CV_8UC1 ), train1( 25, 32, CV_8UC1 );
  theRNG().fill( query, RNG::UNIFORM, 0, 256 );
  theRNG().fill( train0, RNG::UNIFORM, 0, 256 );
  theRNG().fill( train1, RNG::UNIFORM, 0, 256 );

  Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
  std::vector<std::vector<DMatch> > expected, first, second;
  bdm->radiusMatch( query, train0, expected, 120.f );

  bdm->add( std::vector<Mat>( 1, train0 ) );
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

  // Incremental add() + train() after an earlier train() must match a single batched add()
  bdm->add( std::vector<Mat>( 1, train1 ) );
  bdm->train();
  std::vector<std::vector<DMatch> > incrRadius, batchRadius;
  bdm->radiusMatch( query, incrRadius, 120.f );

  Ptr<BinaryDescriptorMatcher> bdmBatch = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
  std::vector<Mat> both( 2 );
  both[0] = train0;
  both[1] = train1;
  bdmBatch->add( both );
  bdmBatch->train();
  bdmBatch->radiusMatch( query, batchRadius, 120.f );

  sortMatchesByTrainIdx( incrRadius );
  sortMatchesByTrainIdx( batchRadius );
  ASSERT_EQ( incrRadius.size(), batchRadius.size() );
  for ( size_t i = 0; i < batchRadius.size(); i++ )
  {
    ASSERT_EQ( incrRadius[i].size(), batchRadius[i].size() ) << "query " << i;
    for ( size_t j = 0; j < batchRadius[i].size(); j++ )
    {
      EXPECT_EQ( incrRadius[i][j].trainIdx, batchRadius[i][j].trainIdx );
      EXPECT_EQ( incrRadius[i][j].imgIdx, batchRadius[i][j].imgIdx );
      EXPECT_EQ( incrRadius[i][j].distance, batchRadius[i][j].distance );
    }
  }

  // knnMatch with k larger than the train set size must clamp safely without reading past k_distances
  Mat tinyTrain = train0.rowRange( 0, 3 ).clone();
  std::vector<std::vector<DMatch> > knnPair, knnDataset;
  bdm->knnMatch( query, tinyTrain, knnPair, 5 );
  ASSERT_EQ( knnPair.size(), static_cast<size_t>( query.rows ) );
  for ( size_t i = 0; i < knnPair.size(); i++ )
    EXPECT_EQ( knnPair[i].size(), 3u );

  bdm->clear();
  std::vector<DMatch> emptySingle;
  std::vector<std::vector<DMatch> > emptyMulti;
  bdm->match( query, emptySingle );
  EXPECT_TRUE( emptySingle.empty() );
  bdm->knnMatch( query, emptyMulti, 2 );
  EXPECT_TRUE( emptyMulti.empty() );
  bdm->radiusMatch( query, emptyMulti, 120.f );
  EXPECT_TRUE( emptyMulti.empty() );

  bdm->add( std::vector<Mat>( 1, tinyTrain ) );
  bdm->knnMatch( query, knnDataset, 5 );
  ASSERT_EQ( knnDataset.size(), static_cast<size_t>( query.rows ) );
  for ( size_t i = 0; i < knnDataset.size(); i++ )
    EXPECT_EQ( knnDataset[i].size(), 3u );
}

TEST( BinaryDescriptor_Matcher, reject_invalid_masks )
{
  Mat query( 64, 32, CV_8UC1 ), train( 60, 32, CV_8UC1 );
  theRNG().fill( query, RNG::UNIFORM, 0, 256 );
  theRNG().fill( train, RNG::UNIFORM, 0, 256 );
  Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

  Mat shortMask = Mat::ones( 5, 1, CV_8UC1 );
  std::vector<DMatch> matches;
  bdm->match( query, train, matches, shortMask );
  EXPECT_TRUE( matches.empty() );

  std::vector<std::vector<DMatch> > kmatches;
  bdm->knnMatch( query, train, kmatches, 2, shortMask );
  EXPECT_TRUE( kmatches.empty() );

  std::vector<std::vector<DMatch> > rmatches;
  bdm->radiusMatch( query, train, rmatches, 120.f, shortMask );
  EXPECT_TRUE( rmatches.empty() );

  Mat wideMask = Mat::ones( query.rows, train.rows, CV_8UC1 );
  bdm->match( query, train, matches, wideMask );
  EXPECT_TRUE( matches.empty() );
  bdm->knnMatch( query, train, kmatches, 2, wideMask );
  EXPECT_TRUE( kmatches.empty() );
  bdm->radiusMatch( query, train, rmatches, 120.f, wideMask );
  EXPECT_TRUE( rmatches.empty() );

  std::vector<Mat> ds( 2 );
  ds[0] = train.rowRange( 0, 30 );
  ds[1] = train.rowRange( 30, 60 );
  bdm->add( ds );
  bdm->train();

  std::vector<Mat> badMasks( 2 );
  badMasks[0] = Mat::ones( query.rows, 1, CV_8UC1 );
  badMasks[1] = shortMask;
  bdm->match( query, matches, badMasks );
  EXPECT_TRUE( matches.empty() );
  bdm->knnMatch( query, kmatches, 2, badMasks );
  EXPECT_TRUE( kmatches.empty() );
  bdm->radiusMatch( query, rmatches, 120.f, badMasks );
  EXPECT_TRUE( rmatches.empty() );
}

}} // namespace
