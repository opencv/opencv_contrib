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
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

//write sanity: ./bin/opencv_perf_tracking --perf_write_sanity=true --perf_min_samples=1
//verify sanity: ./bin/opencv_perf_tracking --perf_min_samples=1

#define TESTSET_NAMES testing::Values("david","dudek","faceocc2")
//#define TESTSET_NAMES testing::internal::ValueArray1<string>("david")
#define SEGMENTS testing::Values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

const string TRACKING_DIR = "cv/tracking";
const string FOLDER_IMG = "data";

typedef perf::TestBaseWithParam<tuple<string, int> > tracking;

std::vector<std::string> splitString( std::string s, std::string delimiter )
{
  std::vector<string> token;
  size_t pos = 0;
  while ( ( pos = s.find( delimiter ) ) != std::string::npos )
  {
    token.push_back( s.substr( 0, pos ) );
    s.erase( 0, pos + delimiter.length() );
  }
  token.push_back( s );
  return token;
}

void checkData( const string& datasetMeta, int& startFrame, string& prefix, string& suffix )
{
  //get informations on the current test data
  FileStorage fs;
  fs.open( datasetMeta, FileStorage::READ );
  fs["start"] >> startFrame;
  fs["prefix"] >> prefix;
  fs["suffix"] >> suffix;
  fs.release();
}

bool getGroundTruth( const string& gtFile, vector<Rect>& gtBBs )
{
  std::ifstream gt;
  //open the ground truth
  gt.open( gtFile.c_str() );
  if( !gt.is_open() )
  {
    return false;
  }
  string line;
  Rect currentBB;
  while ( getline( gt, line ) )
  {
    vector<string> tokens = splitString( line, "," );

    if( tokens.size() != 4 )
    {
      return false;
    }

    gtBBs.push_back(
        Rect( atoi( tokens.at( 0 ).c_str() ), atoi( tokens.at( 1 ).c_str() ), atoi( tokens.at( 2 ).c_str() ), atoi( tokens.at( 3 ).c_str() ) ) );
  }
  return true;
}

void getSegment( int segmentId, int numSegments, int bbCounter, int& startFrame, int& endFrame )
{
  //compute the start and the and for each segment
  int gtStartFrame = startFrame;
  int numFrame = bbCounter / numSegments;
  startFrame += ( segmentId - 1 ) * numFrame;
  endFrame = startFrame + numFrame;

  if( ( segmentId ) == numSegments )
    endFrame = bbCounter + gtStartFrame - 1;
}

void getMatOfRects( const vector<Rect>& bbs, Mat& bbs_mat )
{
  for ( int b = 0, size = (int)bbs.size(); b < size; b++ )
  {
    bbs_mat.at<float>( b, 0 ) = (float)bbs[b].x;
    bbs_mat.at<float>( b, 1 ) = (float)bbs[b].y;
    bbs_mat.at<float>( b, 2 ) = (float)bbs[b].width;
    bbs_mat.at<float>( b, 3 ) = (float)bbs[b].height;
  }
}

PERF_TEST_P(tracking, mil, testing::Combine(TESTSET_NAMES, SEGMENTS))
{
  string video = get<0>( GetParam() );
  int segmentId = get<1>( GetParam() );

  int startFrame;
  string prefix;
  string suffix;
  string datasetMeta = getDataPath( TRACKING_DIR + "/" + video + "/" + video + ".yml" );
  checkData( datasetMeta, startFrame, prefix, suffix );
  int gtStartFrame = startFrame;

  vector<Rect> gtBBs;
  string gtFile = getDataPath( TRACKING_DIR + "/" + video + "/gt.txt" );
  if( !getGroundTruth( gtFile, gtBBs ) )
    FAIL()<< "Ground truth file " << gtFile << " can not be read" << endl;
  int bbCounter = (int)gtBBs.size();

  Mat frame;
  bool initialized = false;
  vector<Rect> bbs;

  Ptr<Tracker> tracker = TrackerMIL::create();
  string folder = TRACKING_DIR + "/" + video + "/" + FOLDER_IMG;
  int numSegments = ( sizeof ( SEGMENTS)/sizeof(int) );
  int endFrame = 0;
  getSegment( segmentId, numSegments, bbCounter, startFrame, endFrame );

  Rect currentBBi = gtBBs[startFrame - gtStartFrame];
  Rect2d currentBB(currentBBi);

  TEST_CYCLE_N(1)
  {
    VideoCapture c;
    c.open( getDataPath( TRACKING_DIR + "/" + video + "/" + FOLDER_IMG + "/" + video + ".webm" ) );
    c.set( CAP_PROP_POS_FRAMES, startFrame );

    for ( int frameCounter = startFrame; frameCounter < endFrame; frameCounter++ )
    {
      c >> frame;

      if( frame.empty() )
      {
        break;
      }

      if( !initialized )
      {
        if( !tracker->init( frame, currentBB ) )
        {
          FAIL()<< "Could not initialize tracker" << endl;
          return;
        }
        initialized = true;
      }
      else if( initialized )
      {
        tracker->update( frame, currentBB );
      }
      bbs.push_back( currentBB );

    }
  }
  //save the bounding boxes in a Mat
  Mat bbs_mat( (int)bbs.size(), 4, CV_32F );
  getMatOfRects( bbs, bbs_mat );

  SANITY_CHECK( bbs_mat, 15, ERROR_RELATIVE );

}

PERF_TEST_P(tracking, boosting, testing::Combine(TESTSET_NAMES, SEGMENTS))
{
  string video = get<0>( GetParam() );
  int segmentId = get<1>( GetParam() );

  int startFrame;
  string prefix;
  string suffix;
  string datasetMeta = getDataPath( TRACKING_DIR + "/" + video + "/" + video + ".yml" );
  checkData( datasetMeta, startFrame, prefix, suffix );
  int gtStartFrame = startFrame;

  vector<Rect> gtBBs;
  string gtFile = getDataPath( TRACKING_DIR + "/" + video + "/gt.txt" );
  if( !getGroundTruth( gtFile, gtBBs ) )
    FAIL()<< "Ground truth file " << gtFile << " can not be read" << endl;
  int bbCounter = (int)gtBBs.size();

  Mat frame;
  bool initialized = false;
  vector<Rect> bbs;

  Ptr<Tracker> tracker = TrackerBoosting::create();
  string folder = TRACKING_DIR + "/" + video + "/" + FOLDER_IMG;
  int numSegments = ( sizeof ( SEGMENTS)/sizeof(int) );
  int endFrame = 0;
  getSegment( segmentId, numSegments, bbCounter, startFrame, endFrame );

  Rect currentBBi = gtBBs[startFrame - gtStartFrame];
  Rect2d currentBB(currentBBi);

  TEST_CYCLE_N(1)
  {
    VideoCapture c;
    c.open( getDataPath( TRACKING_DIR + "/" + video + "/" + FOLDER_IMG + "/" + video + ".webm" ) );
    c.set( CAP_PROP_POS_FRAMES, startFrame );
    for ( int frameCounter = startFrame; frameCounter < endFrame; frameCounter++ )
    {
      c >> frame;

      if( frame.empty() )
      {
        break;
      }

      if( !initialized )
      {
        if( !tracker->init( frame, currentBB ) )
        {
          FAIL()<< "Could not initialize tracker" << endl;
          return;
        }
        initialized = true;
      }
      else if( initialized )
      {
        tracker->update( frame, currentBB );
      }
      bbs.push_back( currentBB );

    }
  }
  //save the bounding boxes in a Mat
  Mat bbs_mat( (int)bbs.size(), 4, CV_32F );
  getMatOfRects( bbs, bbs_mat );

  SANITY_CHECK( bbs_mat, 15, ERROR_RELATIVE );

}

PERF_TEST_P(tracking, tld, testing::Combine(TESTSET_NAMES, SEGMENTS))
{
  string video = get<0>( GetParam() );
  int segmentId = get<1>( GetParam() );

  int startFrame;
  string prefix;
  string suffix;
  string datasetMeta = getDataPath( TRACKING_DIR + "/" + video + "/" + video + ".yml" );
  checkData( datasetMeta, startFrame, prefix, suffix );
  int gtStartFrame = startFrame;

  vector<Rect> gtBBs;
  string gtFile = getDataPath( TRACKING_DIR + "/" + video + "/gt.txt" );
  if( !getGroundTruth( gtFile, gtBBs ) )
    FAIL()<< "Ground truth file " << gtFile << " can not be read" << endl;
  int bbCounter = (int)gtBBs.size();

  Mat frame;
  bool initialized = false;
  vector<Rect> bbs;

  Ptr<Tracker> tracker = TrackerTLD::create();
  string folder = TRACKING_DIR + "/" + video + "/" + FOLDER_IMG;
  int numSegments = ( sizeof ( SEGMENTS)/sizeof(int) );
  int endFrame = 0;
  getSegment( segmentId, numSegments, bbCounter, startFrame, endFrame );

  Rect currentBBi = gtBBs[startFrame - gtStartFrame];
  Rect2d currentBB(currentBBi);

  TEST_CYCLE_N(1)
  {
    VideoCapture c;
    c.open( getDataPath( TRACKING_DIR + "/" + video + "/" + FOLDER_IMG + "/" + video + ".webm" ) );
    c.set( CAP_PROP_POS_FRAMES, startFrame );
    for ( int frameCounter = startFrame; frameCounter < endFrame; frameCounter++ )
    {
      c >> frame;

      if( frame.empty() )
      {
        break;
      }

      if( !initialized )
      {
        if( !tracker->init( frame, currentBB ) )
        {
          FAIL()<< "Could not initialize tracker" << endl;
          return;
        }
        initialized = true;
      }
      else if( initialized )
      {
        tracker->update( frame, currentBB );
      }
      bbs.push_back( currentBB );

    }
  }
  //save the bounding boxes in a Mat
  Mat bbs_mat( (int)bbs.size(), 4, CV_32F );
  getMatOfRects( bbs, bbs_mat );

  SANITY_CHECK( bbs_mat, 15, ERROR_RELATIVE );

}

PERF_TEST_P(tracking, GOTURN, testing::Combine(TESTSET_NAMES, SEGMENTS))
{
  string video = get<0>(GetParam());
  int segmentId = get<1>(GetParam());

  int startFrame;
  string prefix;
  string suffix;
  string datasetMeta = getDataPath(TRACKING_DIR + "/" + video + "/" + video + ".yml");
  checkData(datasetMeta, startFrame, prefix, suffix);
  int gtStartFrame = startFrame;

  vector<Rect> gtBBs;
  string gtFile = getDataPath(TRACKING_DIR + "/" + video + "/gt.txt");
  if (!getGroundTruth(gtFile, gtBBs))
    FAIL() << "Ground truth file " << gtFile << " can not be read" << endl;
  int bbCounter = (int)gtBBs.size();

  Mat frame;
  bool initialized = false;
  vector<Rect> bbs;

  Ptr<Tracker> tracker = TrackerGOTURN::create();
  string folder = TRACKING_DIR + "/" + video + "/" + FOLDER_IMG;
  int numSegments = (sizeof(SEGMENTS) / sizeof(int));
  int endFrame = 0;
  getSegment(segmentId, numSegments, bbCounter, startFrame, endFrame);

  Rect currentBBi = gtBBs[startFrame - gtStartFrame];
  Rect2d currentBB(currentBBi);

  TEST_CYCLE_N(1)
  {
    VideoCapture c;
    c.open(getDataPath(TRACKING_DIR + "/" + video + "/" + FOLDER_IMG + "/" + video + ".webm"));
    c.set(CAP_PROP_POS_FRAMES, startFrame);
    for (int frameCounter = startFrame; frameCounter < endFrame; frameCounter++)
    {
      c >> frame;

      if (frame.empty())
      {
        break;
      }

      if (!initialized)
      {
        if (!tracker->init(frame, currentBB))
        {
          FAIL() << "Could not initialize tracker" << endl;
          return;
        }
        initialized = true;
      }
      else if (initialized)
      {
        tracker->update(frame, currentBB);
      }
      bbs.push_back(currentBB);

    }
  }
  //save the bounding boxes in a Mat
  Mat bbs_mat((int)bbs.size(), 4, CV_32F);
  getMatOfRects(bbs, bbs_mat);

  SANITY_CHECK(bbs_mat, 15, ERROR_RELATIVE);

}

}} // namespace
