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

#include "test_precomp.hpp"
#include "opencv2/tracking.hpp"
#include <fstream>

using namespace cv;
using namespace testing;
using namespace std;

#define PARAM_TEST_CASE(name, ...) struct name : testing::TestWithParam< std::tr1::tuple< __VA_ARGS__ > >
#define GET_PARAM(k) std::tr1::get< k >(GetParam())
#define TESTSET_NAMES testing::Values("david","dudek","faceocc2")
#define LOCATION_ERROR_THRESHOLD testing::Values(0, 10, 20, 30, 40, 50)
#define OVERLAP_THRESHOLD testing::Values(0, 0.2, 0.4, 0.6, 0.8, 1)

#define SPATIAL_SHIFTS testing::Values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12)

const string TRACKING_DIR = "tracking";
const string FOLDER_IMG = "data";

/*
 * The Evaluation Methodologies are partially based on:
 * ====================================================================================================================
 *  [OTB] Y. Wu, J. Lim, and M.-H. Yang, "Online object tracking: A benchmark," in Computer Vision and Pattern Recognition (CVPR), 2013
 *
 */

//Robustness Evaluation, see [OTB] chapter 4. SRE Spatial robustness evaluation
class TrackerSRETest
{
 public:
  enum
  {
    DISTANCE = 1,  // test trackers based on euclidean distance
    OVERLAP = 2    // test trackers based on the overlapping ratio
  };

  TrackerSRETest( const Ptr<Tracker> _tracker, int _testType, string _video, int _shift, float _threshold );
  virtual ~TrackerSRETest();
  virtual void run();
  string getRatioSucc() const;

 protected:
  void checkDataTest();

  void distanceTest();
  void overlapTest();

  Ptr<Tracker> tracker;
  int testType;
  string video;
  std::vector<Rect> bbs;
  int startFrame;
  string suffix;
  string prefix;
  float threshold;
  int shift;
  float ratioSucc;

 private:
  float calcDistance( Rect a, Rect b );
  float calcOverlap( Rect a, Rect b );
  std::vector<std::string> splitString( std::string s, std::string delimiter );

};

TrackerSRETest::TrackerSRETest( const Ptr<Tracker> _tracker, int _testType, string _video, int _shift, float _threshold ) :
    tracker( _tracker ),
    testType( _testType ),
    video( _video ),
    threshold( _threshold ),
    shift( _shift )
{
  startFrame = 1;
  ratioSucc = 0;
}

TrackerSRETest::~TrackerSRETest()
{

}

string TrackerSRETest::getRatioSucc() const
{
  stringstream ratio;
  ratio << ratioSucc;
  return ratio.str();
}

std::vector<std::string> TrackerSRETest::splitString( std::string s, std::string delimiter )
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

float TrackerSRETest::calcDistance( Rect a, Rect b )
{
  Point2f p_a( (float)(a.x + a.width / 2), (float)(a.y + a.height / 2) );
  Point2f p_b( (float)(b.x + b.width / 2), (float)(b.y + b.height / 2) );
  return sqrt( pow( p_a.x - p_b.x, 2 ) + pow( p_a.y - p_b.y, 2 ) );
}

float TrackerSRETest::calcOverlap( Rect a, Rect b )
{
  float aArea = (float)(a.width * a.height);
  float bArea = (float)(b.width * b.height);

  if( aArea < bArea )
  {
    a.x -= ( b.width - a.width ) / 2;
    a.y -= ( b.height - a.height ) / 2;
    a.width = b.width;
    a.height = b.height;
  }
  else
  {
    b.x -= ( a.width - b.width ) / 2;
    b.y -= ( a.height - b.height ) / 2;
    b.width = a.width;
    b.height = a.height;
  }

  Rect rectIntersection = a & b;
  Rect rectUnion = a | b;
  float iArea = (float)(rectIntersection.width * rectIntersection.height);
  float uArea = (float)(rectUnion.width * rectUnion.height);
  float overlap = iArea / uArea;
  return overlap;
}

void TrackerSRETest::distanceTest()
{
  Mat frame;
  bool initialized = false;

  Rect currentBBi = bbs.at( 0 );
  Rect2d currentBB(currentBBi);
  float sumDistance = 0;
  int frameCounter = 0;
  int frameCounterSucc = 0;
  string folder = cvtest::TS::ptr()->get_data_path() + TRACKING_DIR + "/" + video + "/" + FOLDER_IMG;

  VideoCapture c;
  c.open( cvtest::TS::ptr()->get_data_path() + "/" + TRACKING_DIR + "/" + video + "/" + FOLDER_IMG + "/" + video + ".webm" );
  c.set( CAP_PROP_POS_FRAMES, startFrame );

  for ( ;; )
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
      if( frameCounter >= (int) bbs.size() )
      break;
      tracker->update( frame, currentBB );
    }

    float curDistance = calcDistance( currentBB, bbs.at( frameCounter ) );
    if( curDistance <= threshold )
      frameCounterSucc++;
    sumDistance += curDistance;
    frameCounter++;
  }

  float distance = sumDistance / frameCounter;
  ratioSucc = (float) frameCounterSucc / (float) frameCounter;

  if( distance > threshold )
  {
    FAIL()<< "Incorrect distance: curr = " << distance << ", max = " << threshold << endl;
    return;
  }

}

void TrackerSRETest::overlapTest()
{
  Mat frame;
  bool initialized = false;
  Rect currentBBi = bbs.at( 0 );
  Rect2d currentBB(currentBBi);
  float sumOverlap = 0;
  string folder = cvtest::TS::ptr()->get_data_path() + TRACKING_DIR + "/" + video + "/" + FOLDER_IMG;

  int frameCounter = 0;
  int frameCounterSucc = 0;

  VideoCapture c;
  c.open( cvtest::TS::ptr()->get_data_path() + "/" + TRACKING_DIR + "/" + video + "/" + FOLDER_IMG + "/" + video + ".webm" );
  c.set( CAP_PROP_POS_FRAMES, startFrame );

  for ( ;; )
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
      if( frameCounter >= (int) bbs.size() )
      break;
      tracker->update( frame, currentBB );
    }
    float curOverlap = calcOverlap( currentBB, bbs.at( frameCounter ) );
    if( curOverlap >= threshold )
      frameCounterSucc++;

    sumOverlap += curOverlap;
    frameCounter++;
  }

  float overlap = sumOverlap / frameCounter;
  ratioSucc = (float) frameCounterSucc / (float) frameCounter;

  if( overlap < threshold )
  {
    FAIL()<< "Incorrect overlap: curr = " << overlap << ", min = " << threshold << endl;
    return;
  }
}

void TrackerSRETest::checkDataTest()
{
  string gtFile = cvtest::TS::ptr()->get_data_path() + TRACKING_DIR + "/" + video + "/gt.txt";

  ifstream gt;
  //open the ground truth
  gt.open( gtFile.c_str() );
  if( !gt.is_open() )
  {
    FAIL()<< "Ground truth file " << gtFile << " can not be read" << endl;
  }
  string line;
  int bbCounter = 0;
  while ( getline( gt, line ) )
  {
    vector<string> tokens = splitString( line, "," );
    Rect bb( atoi( tokens.at( 0 ).c_str() ), atoi( tokens.at( 1 ).c_str() ), atoi( tokens.at( 2 ).c_str() ), atoi( tokens.at( 3 ).c_str() ) );
    if( tokens.size() != 4 )
    {
      FAIL()<< "Incorrect ground truth file";
    }

    //apply the shift
    if( bbCounter == 0 )
    {
      Point center( bb.x + ( bb.width / 2 ), bb.y + ( bb.height / 2 ) );

      int xLimit = bb.x + bb.width - 1;
      int yLimit = bb.y + bb.height - 1;

      int h = 0;
      int w = 0;
      float ratio = 1.0;

      switch ( shift )
      {
        case 1:
          //center shift left
          bb.x = bb.x - (int)ceil( 0.1 * bb.width );
          break;
        case 2:
          //center shift right
          bb.x = bb.x + (int)ceil( 0.1 * bb.width );
          break;
        case 3:
          //center shift up
          bb.y = bb.y - (int)ceil( 0.1 * bb.height );
          break;
        case 4:
          //center shift down
          bb.y = bb.y + (int)ceil( 0.1 * bb.height );
          break;
        case 5:
          //corner shift top-left
          bb.x = (int)cvRound( bb.x -  0.1 * bb.width );
          bb.y = (int)cvRound( bb.y - 0.1 * bb.height );

          bb.width = xLimit - bb.x + 1;
          bb.height = yLimit - bb.y + 1;
          break;
        case 6:
          //corner shift top-right
          xLimit = (int)cvRound( xLimit + 0.1 * bb.width );

          bb.y = (int)cvRound( bb.y - 0.1 * bb.height );
          bb.width = xLimit - bb.x + 1;
          bb.height = yLimit - bb.y + 1;
          break;
        case 7:
          //corner shift bottom-left
          bb.x = (int)cvRound( bb.x - 0.1 * bb.width );
          yLimit = (int)cvRound( yLimit + 0.1 * bb.height );

          bb.width = xLimit - bb.x + 1;
          bb.height = yLimit - bb.y + 1;
          break;
        case 8:
          //corner shift bottom-right
          xLimit = (int)cvRound( xLimit + 0.1 * bb.width );
          yLimit = (int)cvRound( yLimit + 0.1 * bb.height );

          bb.width = xLimit - bb.x + 1;
          bb.height = yLimit - bb.y + 1;
          break;
        case 9:
          //scale 0.8
          ratio = 0.8f;
          w = (int)(ratio * bb.width);
          h = (int)(ratio * bb.height);

          bb = Rect( center.x - ( w / 2 ), center.y - ( h / 2 ), w, h );
          break;
        case 10:
          //scale 0.9
          ratio = 0.9f;
          w = (int)(ratio * bb.width);
          h = (int)(ratio * bb.height);

          bb = Rect( center.x - ( w / 2 ), center.y - ( h / 2 ), w, h );
          break;
        case 11:
          //scale 1.1
          ratio = 1.1f;
          w = (int)(ratio * bb.width);
          h = (int)(ratio * bb.height);

          bb = Rect( center.x - ( w / 2 ), center.y - ( h / 2 ), w, h );
          break;
        case 12:
          //scale 1.2
          ratio = 1.2f;
          w = (int)(ratio * bb.width);
          h = (int)(ratio * bb.height);

          bb = Rect( center.x - ( w / 2 ), center.y - ( h / 2 ), w, h );
          break;
        default:
          break;
      }
    }
    bbs.push_back( bb );
    bbCounter++;
  }

  FileStorage fs;
  fs.open( cvtest::TS::ptr()->get_data_path() + TRACKING_DIR + "/" + video + "/" + video + ".yml", FileStorage::READ );
  fs["start"] >> startFrame;
  fs["prefix"] >> prefix;
  fs["suffix"] >> suffix;
  fs.release();

}

void TrackerSRETest::run()
{
  srand( 1 );
  SCOPED_TRACE( "A" );

  if( !tracker )
  {
    FAIL()<< "Error in the instantiation of the tracker" << endl;
    return;
  }

  checkDataTest();

  //check for failure
  if( ::testing::Test::HasFatalFailure() )
    return;

  if( testType == DISTANCE )
  {
    distanceTest();
  }
  else if( testType == OVERLAP )
  {
    overlapTest();
  }
  else
  {
    FAIL()<< "Test type unknown" << endl;
    return;
  }

}

/****************************************************************************************\
*                                Tests registrations                                     *
 \****************************************************************************************/

//[TESTDATA] [#SHIFT] [LOCATION ERROR THRESHOLD]
PARAM_TEST_CASE(SRE_Distance, string, int, float)
{
  string dataset;
  int shift;
  float threshold;
  virtual void SetUp()
  {
    dataset = GET_PARAM(0);
    shift = GET_PARAM(1);
    threshold = GET_PARAM(2);
  }
};

//[TESTDATA] [#SHIFT] [OVERLAP THRESHOLD]
PARAM_TEST_CASE(SRE_Overlap, string, int, float)
{
  string dataset;
  int shift;
  float threshold;
  virtual void SetUp()
  {
    dataset = GET_PARAM(0);
    shift = GET_PARAM(1);
    threshold = GET_PARAM(2);
  }
};

TEST_P(SRE_Distance, MIL)
{
  TrackerSRETest test( Tracker::create( "MIL" ), TrackerSRETest::DISTANCE, dataset, shift, threshold );
  test.run();
  RecordProperty( "ratioSuccess", test.getRatioSucc() );
}

TEST_P(SRE_Overlap, MIL)
{
  TrackerSRETest test( Tracker::create( "MIL" ), TrackerSRETest::OVERLAP, dataset, shift, threshold );
  test.run();
  RecordProperty( "ratioSuccess", test.getRatioSucc() );
}

TEST_P(SRE_Distance, Boosting)
{
  TrackerSRETest test( Tracker::create( "BOOSTING" ), TrackerSRETest::DISTANCE, dataset, shift, threshold );
  test.run();
  RecordProperty( "ratioSuccess", test.getRatioSucc() );
}

TEST_P(SRE_Overlap, Boosting)
{
  TrackerSRETest test( Tracker::create( "BOOSTING" ), TrackerSRETest::OVERLAP, dataset, shift, threshold );
  test.run();
  RecordProperty( "ratioSuccess", test.getRatioSucc() );
}

TEST_P(SRE_Distance, TLD)
{
  TrackerSRETest test( Tracker::create( "TLD" ), TrackerSRETest::DISTANCE, dataset, shift, threshold );
  test.run();
  RecordProperty( "ratioSuccess", test.getRatioSucc() );
}

TEST_P(SRE_Overlap, TLD)
{
  TrackerSRETest test( Tracker::create( "TLD" ), TrackerSRETest::OVERLAP, dataset, shift, threshold );
  test.run();
  RecordProperty( "ratioSuccess", test.getRatioSucc() );
}

TEST_P(SRE_Distance, GOTURN)
{
  TrackerSRETest test(Tracker::create("GOTURN"), TrackerSRETest::DISTANCE, dataset, shift, threshold);
  test.run();
  RecordProperty("ratioSuccess", test.getRatioSucc());
}

TEST_P(SRE_Overlap, GOTURN)
{
  TrackerSRETest test(Tracker::create("GOTURN"), TrackerSRETest::OVERLAP, dataset, shift, threshold);
  test.run();
  RecordProperty("ratioSuccess", test.getRatioSucc());
}

INSTANTIATE_TEST_CASE_P( Tracking, SRE_Distance, testing::Combine( TESTSET_NAMES, SPATIAL_SHIFTS, LOCATION_ERROR_THRESHOLD ) );

INSTANTIATE_TEST_CASE_P( Tracking, SRE_Overlap, testing::Combine( TESTSET_NAMES, SPATIAL_SHIFTS, OVERLAP_THRESHOLD ) );

/* End of file. */
