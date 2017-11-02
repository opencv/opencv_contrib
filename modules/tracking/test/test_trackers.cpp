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

#define TESTSET_NAMES testing::Values("david","dudek","faceocc2")

const string TRACKING_DIR = "tracking";
const string FOLDER_IMG = "data";
const string FOLDER_OMIT_INIT = "initOmit";

/*
 * The Evaluation Methodologies are partially based on:
 * ====================================================================================================================
 *  [OTB] Y. Wu, J. Lim, and M.-H. Yang, "Online object tracking: A benchmark," in Computer Vision and Pattern Recognition (CVPR), 2013
 *
 */

enum BBTransformations
{
    NoTransform = 0,
    CenterShiftLeft = 1,
    CenterShiftRight = 2,
    CenterShiftUp = 3,
    CenterShiftDown = 4,
    CornerShiftTopLeft = 5,
    CornerShiftTopRight = 6,
    CornerShiftBottomLeft = 7,
    CornerShiftBottomRight = 8,
    Scale_0_8 = 9,
    Scale_0_9 = 10,
    Scale_1_1 = 11,
    Scale_1_2 = 12
};

class TrackerTest
{
 public:

  TrackerTest(Ptr<Tracker> _tracker, string _video, float _distanceThreshold,
                 float _overlapThreshold, int _shift = NoTransform, int _segmentIdx = 1, int _numSegments = 10 );
  virtual ~TrackerTest();
  virtual void run();

 protected:
  void checkDataTest();

  void distanceAndOvrerlapTest();

  Ptr<Tracker> tracker;
  string video;
  std::vector<Rect> bbs;
  int startFrame;
  string suffix;
  string prefix;
  float overlapThreshold;
  float distanceThreshold;
  int segmentIdx;
  int shift;
  int numSegments;

  int gtStartFrame;
  int endFrame;
  vector<int> validSequence;

 private:
  float calcDistance( Rect a, Rect b );
  float calcOverlap( Rect a, Rect b );
  Rect applyShift(Rect bb);
  std::vector<std::string> splitString( std::string s, std::string delimiter );

};

TrackerTest::TrackerTest(Ptr<Tracker> _tracker, string _video, float _distanceThreshold,
                                float _overlapThreshold, int _shift, int _segmentIdx, int _numSegments ) :
    tracker( _tracker ),
    video( _video ),
    overlapThreshold( _overlapThreshold ),
    distanceThreshold( _distanceThreshold ),
    segmentIdx(_segmentIdx),
    shift(_shift),
    numSegments(_numSegments)
{
}

TrackerTest::~TrackerTest()
{

}

std::vector<std::string> TrackerTest::splitString( std::string s, std::string delimiter )
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

float TrackerTest::calcDistance( Rect a, Rect b )
{
  Point2f p_a( (float)(a.x + a.width / 2), (float)(a.y + a.height / 2) );
  Point2f p_b( (float)(b.x + b.width / 2), (float)(b.y + b.height / 2) );
  return sqrt( pow( p_a.x - p_b.x, 2 ) + pow( p_a.y - p_b.y, 2 ) );
}

float TrackerTest::calcOverlap( Rect a, Rect b )
{
  float rectIntersectionArea = (float)(a & b).area();
  return rectIntersectionArea / (a.area() + b.area() - rectIntersectionArea);
}

Rect TrackerTest::applyShift(Rect bb)
{
  Point center( bb.x + ( bb.width / 2 ), bb.y + ( bb.height / 2 ) );

  int xLimit = bb.x + bb.width - 1;
  int yLimit = bb.y + bb.height - 1;

  int h = 0;
  int w = 0;
  float ratio = 1.0;

  switch ( shift )
  {
    case CenterShiftLeft:
      bb.x = bb.x - (int)ceil( 0.1 * bb.width );
      break;
    case CenterShiftRight:
      bb.x = bb.x + (int)ceil( 0.1 * bb.width );
      break;
    case CenterShiftUp:
      bb.y = bb.y - (int)ceil( 0.1 * bb.height );
      break;
    case CenterShiftDown:
      bb.y = bb.y + (int)ceil( 0.1 * bb.height );
      break;
    case CornerShiftTopLeft:
      bb.x = (int)cvRound( bb.x -  0.1 * bb.width );
      bb.y = (int)cvRound( bb.y - 0.1 * bb.height );

      bb.width = xLimit - bb.x + 1;
      bb.height = yLimit - bb.y + 1;
      break;
    case CornerShiftTopRight:
      xLimit = (int)cvRound( xLimit + 0.1 * bb.width );

      bb.y = (int)cvRound( bb.y - 0.1 * bb.height );
      bb.width = xLimit - bb.x + 1;
      bb.height = yLimit - bb.y + 1;
      break;
    case CornerShiftBottomLeft:
      bb.x = (int)cvRound( bb.x - 0.1 * bb.width );
      yLimit = (int)cvRound( yLimit + 0.1 * bb.height );

      bb.width = xLimit - bb.x + 1;
      bb.height = yLimit - bb.y + 1;
      break;
    case CornerShiftBottomRight:
      xLimit = (int)cvRound( xLimit + 0.1 * bb.width );
      yLimit = (int)cvRound( yLimit + 0.1 * bb.height );

      bb.width = xLimit - bb.x + 1;
      bb.height = yLimit - bb.y + 1;
      break;
    case Scale_0_8:
      ratio = 0.8f;
      w = (int)(ratio * bb.width);
      h = (int)(ratio * bb.height);

      bb = Rect( center.x - ( w / 2 ), center.y - ( h / 2 ), w, h );
      break;
    case Scale_0_9:
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

  return bb;
}

void TrackerTest::distanceAndOvrerlapTest()
{
  Mat frame;
  bool initialized = false;

  int fc = ( startFrame - gtStartFrame );

  bbs.at( fc ) = applyShift(bbs.at( fc ));
  Rect currentBBi = bbs.at( fc );
  Rect2d currentBB(currentBBi);
  float sumDistance = 0;
  float sumOverlap = 0;

  string folder = cvtest::TS::ptr()->get_data_path() + "/" + TRACKING_DIR + "/" + video + "/" + FOLDER_IMG;

  VideoCapture c;
  c.open( folder + "/" + video + ".webm" );
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
      if( frameCounter >= (int) bbs.size() )
      break;
      tracker->update( frame, currentBB );
    }
    float curDistance = calcDistance( currentBB, bbs.at( fc ) );
    float curOverlap = calcOverlap( currentBB, bbs.at( fc ) );

    sumDistance += curDistance;
    sumOverlap += curOverlap;
    fc++;
  }

  float meanDistance = sumDistance / (endFrame - startFrame);
  float meanOverlap = sumOverlap / (endFrame - startFrame);

  if( meanDistance > distanceThreshold )
  {
    FAIL()<< "Incorrect distance: curr = " << meanDistance << ", max = " << distanceThreshold << endl;
    return;
  }

  if( meanOverlap < overlapThreshold )
  {
    FAIL()<< "Incorrect overlap: curr = " << meanOverlap << ", min = " << overlapThreshold << endl;
    return;
  }
}

void TrackerTest::checkDataTest()
{

  FileStorage fs;
  fs.open( cvtest::TS::ptr()->get_data_path() + TRACKING_DIR + "/" + video + "/" + video + ".yml", FileStorage::READ );
  fs["start"] >> startFrame;
  fs["prefix"] >> prefix;
  fs["suffix"] >> suffix;
  fs.release();

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
    bbCounter++;
  }
  gt.close();

  int seqLength = bbCounter;
  for ( int i = startFrame; i < seqLength; i++ )
  {
    validSequence.push_back( i );
  }

  //exclude from the images sequence, the frames where the target is occluded or out of view
  string omitFile = cvtest::TS::ptr()->get_data_path() + TRACKING_DIR + "/" + video + "/" + FOLDER_OMIT_INIT + "/" + video + ".txt";
  ifstream omit;
  omit.open( omitFile.c_str() );
  if( omit.is_open() )
  {
    string omitLine;
    while ( getline( omit, omitLine ) )
    {
      vector<string> tokens = splitString( omitLine, " " );
      int s_start = atoi( tokens.at( 0 ).c_str() );
      int s_end = atoi( tokens.at( 1 ).c_str() );
      for ( int k = s_start; k <= s_end; k++ )
      {
        std::vector<int>::iterator position = std::find( validSequence.begin(), validSequence.end(), k );
        if( position != validSequence.end() )
          validSequence.erase( position );
      }
    }
  }
  omit.close();
  gtStartFrame = startFrame;
  //compute the start and the and for each segment
  int numFrame = (int)(validSequence.size() / numSegments);
  startFrame += ( segmentIdx - 1 ) * numFrame;
  endFrame = startFrame + numFrame;

  ifstream gt2;
  //open the ground truth
  gt2.open( gtFile.c_str() );
  if( !gt2.is_open() )
  {
    FAIL()<< "Ground truth file " << gtFile << " can not be read" << endl;
  }
  string line2;
  int bbCounter2 = 0;
  while ( getline( gt2, line2 ) )
  {
    vector<string> tokens = splitString( line2, "," );
    Rect bb( atoi( tokens.at( 0 ).c_str() ), atoi( tokens.at( 1 ).c_str() ), atoi( tokens.at( 2 ).c_str() ), atoi( tokens.at( 3 ).c_str() ) );
    if( tokens.size() != 4 )
    {
      FAIL()<< "Incorrect ground truth file";
    }

    bbs.push_back( bb );
    bbCounter2++;
  }
  gt2.close();

  if( segmentIdx == numSegments )
    endFrame = (int)bbs.size();
}

void TrackerTest::run()
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

  distanceAndOvrerlapTest();
}

/****************************************************************************************\
*                                Tests registrations                                     *
 \****************************************************************************************/

//[TESTDATA]
PARAM_TEST_CASE(DistanceAndOverlap, string)
{
  string dataset;
  virtual void SetUp()
  {
    dataset = GET_PARAM(0);
  }
};

TEST_P(DistanceAndOverlap, MedianFlow)
{
  TrackerTest test( TrackerMedianFlow::create(), dataset, 35, .5f, NoTransform, 1, 1);
  test.run();
}

TEST_P(DistanceAndOverlap, MIL)
{
  TrackerTest test( TrackerMIL::create(), dataset, 30, .65f, NoTransform);
  test.run();
}

TEST_P(DistanceAndOverlap, Boosting)
{
  TrackerTest test( TrackerBoosting::create(), dataset, 70, .7f, NoTransform);
  test.run();
}

TEST_P(DistanceAndOverlap, KCF)
{
  TrackerTest test( TrackerKCF::create(), dataset, 20, .35f, NoTransform, 5);
  test.run();
}

TEST_P(DistanceAndOverlap, TLD)
{
  TrackerTest test( TrackerTLD::create(), dataset, 40, .45f, NoTransform);
  test.run();
}

TEST_P(DistanceAndOverlap, MOSSE)
{
  TrackerTest test( TrackerMOSSE::create(), dataset, 22, .7f, NoTransform);
  test.run();
}

/***************************************************************************************/
//Tests with shifted initial window
TEST_P(DistanceAndOverlap, Shifted_Data_MedianFlow)
{
  TrackerTest test( TrackerMedianFlow::create(), dataset, 80, .2f, CenterShiftLeft, 1, 1);
  test.run();
}

TEST_P(DistanceAndOverlap, Shifted_Data_MIL)
{
  TrackerTest test( TrackerMIL::create(), dataset, 30, .6f, CenterShiftLeft);
  test.run();
}

TEST_P(DistanceAndOverlap, Shifted_Data_Boosting)
{
  TrackerTest test( TrackerBoosting::create(), dataset, 80, .65f, CenterShiftLeft);
  test.run();
}

TEST_P(DistanceAndOverlap, Shifted_Data_KCF)
{
  TrackerTest test( TrackerKCF::create(), dataset, 20, .4f, CenterShiftLeft, 5);
  test.run();
}

TEST_P(DistanceAndOverlap, Shifted_Data_TLD)
{
  TrackerTest test( TrackerTLD::create(), dataset, 30, .35f, CenterShiftLeft);
  test.run();
}

TEST_P(DistanceAndOverlap, Shifted_Data_MOSSE)
{
  TrackerTest test( TrackerMOSSE::create(), dataset, 13, .69f, CenterShiftLeft);
  test.run();
}
/***************************************************************************************/
//Tests with scaled initial window
TEST_P(DistanceAndOverlap, Scaled_Data_MedianFlow)
{
  TrackerTest test( TrackerMedianFlow::create(), dataset, 25, .5f, Scale_1_1, 1, 1);
  test.run();
}

TEST_P(DistanceAndOverlap, Scaled_Data_MIL)
{
  TrackerTest test( TrackerMIL::create(), dataset, 30, .7f, Scale_1_1);
  test.run();
}

TEST_P(DistanceAndOverlap, Scaled_Data_Boosting)
{
  TrackerTest test( TrackerBoosting::create(), dataset, 80, .7f, Scale_1_1);
  test.run();
}

TEST_P(DistanceAndOverlap, Scaled_Data_KCF)
{
  TrackerTest test( TrackerKCF::create(), dataset, 20, .4f, Scale_1_1, 5);
  test.run();
}

TEST_P(DistanceAndOverlap, Scaled_Data_TLD)
{
  TrackerTest test( TrackerTLD::create(), dataset, 30, .45f, Scale_1_1);
  test.run();
}


TEST_P(DistanceAndOverlap, DISABLED_GOTURN)
{
  TrackerTest test(TrackerGOTURN::create(), dataset, 18, .5f, NoTransform);
  test.run();
}

TEST_P(DistanceAndOverlap, Scaled_Data_MOSSE)
{
  TrackerTest test( TrackerMOSSE::create(), dataset, 22, 0.69f, Scale_1_1, 1);
  test.run();
}


INSTANTIATE_TEST_CASE_P( Tracking, DistanceAndOverlap, TESTSET_NAMES);

/* End of file. */
