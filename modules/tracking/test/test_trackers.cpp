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

#define TEST_LEGACY
#include <opencv2/tracking/tracking_legacy.hpp>

//#define DEBUG_TEST
#ifdef DEBUG_TEST
#include <opencv2/highgui.hpp>
#endif

namespace opencv_test { namespace {
//using namespace cv::tracking;

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

namespace {

std::vector<std::string> splitString(const std::string& s_, const std::string& delimiter)
{
  std::string s = s_;
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

float calcDistance(const Rect& a, const Rect& b)
{
  Point2f p_a( (float)(a.x + a.width / 2), (float)(a.y + a.height / 2) );
  Point2f p_b( (float)(b.x + b.width / 2), (float)(b.y + b.height / 2) );
  return sqrt( pow( p_a.x - p_b.x, 2 ) + pow( p_a.y - p_b.y, 2 ) );
}

float calcOverlap(const Rect& a, const Rect& b)
{
  float rectIntersectionArea = (float)(a & b).area();
  return rectIntersectionArea / (a.area() + b.area() - rectIntersectionArea);
}

}  // namespace


template<typename Tracker, typename ROI_t = Rect2d>
class TrackerTest
{
public:

  TrackerTest(const Ptr<Tracker>& tracker, const string& video, float distanceThreshold,
              float overlapThreshold, int shift = NoTransform, int segmentIdx = 1, int numSegments = 10);
  ~TrackerTest() {}
  void run();

protected:
  void checkDataTest();

  void distanceAndOverlapTest();

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
  Rect applyShift(const Rect& bb);
};

template<typename Tracker, typename ROI_t>
TrackerTest<Tracker, ROI_t>::TrackerTest(const Ptr<Tracker>& _tracker, const string& _video, float _distanceThreshold,
                         float _overlapThreshold, int _shift, int _segmentIdx, int _numSegments ) :
    tracker( _tracker ),
    video( _video ),
    overlapThreshold( _overlapThreshold ),
    distanceThreshold( _distanceThreshold ),
    segmentIdx(_segmentIdx),
    shift(_shift),
    numSegments(_numSegments)
{
    // nothing
}

template<typename Tracker, typename ROI_t>
Rect TrackerTest<Tracker, ROI_t>::applyShift(const Rect& bb_)
{
  Rect bb = bb_;
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

template<typename Tracker, typename ROI_t>
void TrackerTest<Tracker, ROI_t>::distanceAndOverlapTest()
{
  bool initialized = false;

  int fc = ( startFrame - gtStartFrame );

  bbs.at( fc ) = applyShift(bbs.at( fc ));
  Rect currentBBi = bbs.at( fc );
  ROI_t currentBB(currentBBi);
  float sumDistance = 0;
  float sumOverlap = 0;

  string folder = cvtest::TS::ptr()->get_data_path() + "/" + TRACKING_DIR + "/" + video + "/" + FOLDER_IMG;
  string videoPath = folder + "/" + video + ".webm";

  VideoCapture c;
  c.open(videoPath);
  ASSERT_TRUE(c.isOpened()) << videoPath;
#if 0
  c.set(CAP_PROP_POS_FRAMES, startFrame);
#else
  if (startFrame)
      std::cout << "startFrame = " << startFrame << std::endl;
  for (int i = 0; i < startFrame; i++)
  {
      Mat dummy_frame;
      c >> dummy_frame;
      ASSERT_FALSE(dummy_frame.empty()) << i << ": " << videoPath;
  }
#endif

  for ( int frameCounter = startFrame; frameCounter < endFrame; frameCounter++ )
  {
    Mat frame;
    c >> frame;

    ASSERT_FALSE(frame.empty()) << "frameCounter=" << frameCounter << " video=" << videoPath;
    if( !initialized )
    {
#if 0
      if( !tracker->init( frame, currentBB ) )
      {
        FAIL()<< "Could not initialize tracker" << endl;
        return;
      }
#else
      tracker->init(frame, currentBB);
#endif
      std::cout << "frame size = " << frame.size() << std::endl;
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

#ifdef DEBUG_TEST
    Mat result;
    repeat(frame, 1, 2, result);
    rectangle(result, currentBB, Scalar(0, 255, 0), 1);
    Rect roi2(frame.cols, 0, frame.cols, frame.rows);
    rectangle(result(roi2), bbs.at(fc), Scalar(0, 0, 255), 1);
    imshow("result", result);
    waitKey(1);
#endif

    sumDistance += curDistance;
    sumOverlap += curOverlap;
    fc++;
  }

  float meanDistance = sumDistance / (endFrame - startFrame);
  float meanOverlap = sumOverlap / (endFrame - startFrame);

  EXPECT_LE(meanDistance, distanceThreshold);
  EXPECT_GE(meanOverlap, overlapThreshold);
}

template<typename Tracker, typename ROI_t>
void TrackerTest<Tracker, ROI_t>::checkDataTest()
{

  FileStorage fs;
  fs.open( cvtest::TS::ptr()->get_data_path() + TRACKING_DIR + "/" + video + "/" + video + ".yml", FileStorage::READ );
  fs["start"] >> startFrame;
  fs["prefix"] >> prefix;
  fs["suffix"] >> suffix;
  fs.release();

  string gtFile = cvtest::TS::ptr()->get_data_path() + TRACKING_DIR + "/" + video + "/gt.txt";
  std::ifstream gt;
  //open the ground truth
  gt.open( gtFile.c_str() );
  ASSERT_TRUE(gt.is_open()) << gtFile;
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
  std::ifstream omit;
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

  std::ifstream gt2;
  //open the ground truth
  gt2.open( gtFile.c_str() );
  ASSERT_TRUE(gt2.is_open()) << gtFile;
  string line2;
  int bbCounter2 = 0;
  while ( getline( gt2, line2 ) )
  {
    vector<string> tokens = splitString( line2, "," );
    Rect bb( atoi( tokens.at( 0 ).c_str() ), atoi( tokens.at( 1 ).c_str() ), atoi( tokens.at( 2 ).c_str() ), atoi( tokens.at( 3 ).c_str() ) );
    ASSERT_EQ((size_t)4, tokens.size()) << "Incorrect ground truth file " << gtFile;

    bbs.push_back( bb );
    bbCounter2++;
  }
  gt2.close();

  if( segmentIdx == numSegments )
    endFrame = (int)bbs.size();
}

template<typename Tracker, typename ROI_t>
void TrackerTest<Tracker, ROI_t>::run()
{
  srand( 1 );  // FIXIT remove that, ensure that there is no "rand()" in implementation

  ASSERT_TRUE(tracker);

  checkDataTest();

  //check for failure
  if( ::testing::Test::HasFatalFailure() )
    return;

  distanceAndOverlapTest();
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
  TrackerTest<legacy::Tracker> test(legacy::TrackerMedianFlow::create(), dataset, 35, .5f, NoTransform, 1, 1);
  test.run();
}

TEST_P(DistanceAndOverlap, MIL)
{
  TrackerTest<Tracker, Rect> test(TrackerMIL::create(), dataset, 30, .65f, NoTransform);
  test.run();
}
#ifdef TEST_LEGACY
TEST_P(DistanceAndOverlap, MIL_legacy)
{
  TrackerTest<legacy::Tracker> test(legacy::TrackerMIL::create(), dataset, 30, .65f, NoTransform);
  test.run();
}
#endif

TEST_P(DistanceAndOverlap, Boosting)
{
  TrackerTest<legacy::Tracker> test(legacy::TrackerBoosting::create(), dataset, 70, .7f, NoTransform);
  test.run();
}

TEST_P(DistanceAndOverlap, KCF)
{
  TrackerTest<Tracker, Rect> test(TrackerKCF::create(), dataset, 20, .35f, NoTransform, 5);
  test.run();
}
#ifdef TEST_LEGACY
TEST_P(DistanceAndOverlap, KCF_legacy)
{
  TrackerTest<legacy::Tracker> test(legacy::TrackerKCF::create(), dataset, 20, .35f, NoTransform, 5);
  test.run();
}
#endif

TEST_P(DistanceAndOverlap, TLD)
{
  TrackerTest<legacy::Tracker> test(legacy::TrackerTLD::create(), dataset, 40, .45f, NoTransform);
  test.run();
}

TEST_P(DistanceAndOverlap, MOSSE)
{
  TrackerTest<legacy::Tracker> test(legacy::TrackerMOSSE::create(), dataset, 22, .7f, NoTransform);
  test.run();
}

TEST_P(DistanceAndOverlap, CSRT)
{
  TrackerTest<Tracker, Rect> test(TrackerCSRT::create(), dataset, 22, .7f, NoTransform);
  test.run();
}
#ifdef TEST_LEGACY
TEST_P(DistanceAndOverlap, CSRT_legacy)
{
  TrackerTest<legacy::Tracker> test(legacy::TrackerCSRT::create(), dataset, 22, .7f, NoTransform);
  test.run();
}
#endif

/***************************************************************************************/
//Tests with shifted initial window
TEST_P(DistanceAndOverlap, Shifted_Data_MedianFlow)
{
  TrackerTest<legacy::Tracker> test(legacy::TrackerMedianFlow::create(), dataset, 80, .2f, CenterShiftLeft, 1, 1);
  test.run();
}

TEST_P(DistanceAndOverlap, Shifted_Data_MIL)
{
  TrackerTest<Tracker, Rect> test(TrackerMIL::create(), dataset, 30, .6f, CenterShiftLeft);
  test.run();
}
#ifdef TEST_LEGACY
TEST_P(DistanceAndOverlap, Shifted_Data_MIL_legacy)
{
  TrackerTest<legacy::Tracker> test(legacy::TrackerMIL::create(), dataset, 30, .6f, CenterShiftLeft);
  test.run();
}
#endif

TEST_P(DistanceAndOverlap, Shifted_Data_Boosting)
{
  TrackerTest<legacy::Tracker> test(legacy::TrackerBoosting::create(), dataset, 80, .65f, CenterShiftLeft);
  test.run();
}

TEST_P(DistanceAndOverlap, Shifted_Data_KCF)
{
  TrackerTest<Tracker, Rect> test(TrackerKCF::create(), dataset, 20, .4f, CenterShiftLeft, 5);
  test.run();
}
#ifdef TEST_LEGACY
TEST_P(DistanceAndOverlap, Shifted_Data_KCF_legacy)
{
  TrackerTest<legacy::Tracker> test(legacy::TrackerKCF::create(), dataset, 20, .4f, CenterShiftLeft, 5);
  test.run();
}
#endif

TEST_P(DistanceAndOverlap, Shifted_Data_TLD)
{
  TrackerTest<legacy::Tracker> test(legacy::TrackerTLD::create(), dataset, 30, .35f, CenterShiftLeft);
  test.run();
}

TEST_P(DistanceAndOverlap, Shifted_Data_MOSSE)
{
  TrackerTest<legacy::Tracker> test(legacy::TrackerMOSSE::create(), dataset, 13, .69f, CenterShiftLeft);
  test.run();
}

TEST_P(DistanceAndOverlap, Shifted_Data_CSRT)
{
  TrackerTest<Tracker, Rect> test(TrackerCSRT::create(), dataset, 13, .69f, CenterShiftLeft);
  test.run();
}
#ifdef TEST_LEGACY
TEST_P(DistanceAndOverlap, Shifted_Data_CSRT_legacy)
{
  TrackerTest<legacy::Tracker> test(legacy::TrackerCSRT::create(), dataset, 13, .69f, CenterShiftLeft);
  test.run();
}
#endif

/***************************************************************************************/
//Tests with scaled initial window
TEST_P(DistanceAndOverlap, Scaled_Data_MedianFlow)
{
  TrackerTest<legacy::Tracker> test(legacy::TrackerMedianFlow::create(), dataset, 25, .5f, Scale_1_1, 1, 1);
  test.run();
}

TEST_P(DistanceAndOverlap, Scaled_Data_MIL)
{
  TrackerTest<Tracker, Rect> test(TrackerMIL::create(), dataset, 30, .7f, Scale_1_1);
  test.run();
}
#ifdef TEST_LEGACY
TEST_P(DistanceAndOverlap, Scaled_Data_MIL_legacy)
{
  TrackerTest<legacy::Tracker> test(legacy::TrackerMIL::create(), dataset, 30, .7f, Scale_1_1);
  test.run();
}
#endif

TEST_P(DistanceAndOverlap, Scaled_Data_Boosting)
{
  TrackerTest<legacy::Tracker> test(legacy::TrackerBoosting::create(), dataset, 80, .7f, Scale_1_1);
  test.run();
}

TEST_P(DistanceAndOverlap, Scaled_Data_KCF)
{
  TrackerTest<Tracker, Rect> test(TrackerKCF::create(), dataset, 20, .4f, Scale_1_1, 5);
  test.run();
}
#ifdef TEST_LEGACY
TEST_P(DistanceAndOverlap, Scaled_Data_KCF_legacy)
{
  TrackerTest<legacy::Tracker> test(legacy::TrackerKCF::create(), dataset, 20, .4f, Scale_1_1, 5);
  test.run();
}
#endif

TEST_P(DistanceAndOverlap, Scaled_Data_TLD)
{
  TrackerTest<legacy::Tracker> test(legacy::TrackerTLD::create(), dataset, 30, .45f, Scale_1_1);
  test.run();
}

TEST_P(DistanceAndOverlap, Scaled_Data_MOSSE)
{
  TrackerTest<legacy::Tracker> test(legacy::TrackerMOSSE::create(), dataset, 22, 0.69f, Scale_1_1, 1);
  test.run();
}

TEST_P(DistanceAndOverlap, Scaled_Data_CSRT)
{
  TrackerTest<Tracker, Rect> test(TrackerCSRT::create(), dataset, 22, 0.69f, Scale_1_1, 1);
  test.run();
}
#ifdef TEST_LEGACY
TEST_P(DistanceAndOverlap, Scaled_Data_CSRT_legacy)
{
  TrackerTest<Tracker, Rect> test(TrackerCSRT::create(), dataset, 22, 0.69f, Scale_1_1, 1);
  test.run();
}
#endif

TEST_P(DistanceAndOverlap, GOTURN)
{
    std::string model = cvtest::findDataFile("dnn/gsoc2016-goturn/goturn.prototxt");
    std::string weights = cvtest::findDataFile("dnn/gsoc2016-goturn/goturn.caffemodel", false);
    cv::TrackerGOTURN::Params params;
    params.modelTxt = model;
    params.modelBin = weights;
    TrackerTest<Tracker, Rect> test(TrackerGOTURN::create(params), dataset, 35, .35f, NoTransform);
    test.run();
}

INSTANTIATE_TEST_CASE_P(Tracking, DistanceAndOverlap, TESTSET_NAMES);



TEST(GOTURN, memory_usage)
{
  cv::Rect roi(145, 70, 85, 85);

  std::string model = cvtest::findDataFile("dnn/gsoc2016-goturn/goturn.prototxt");
  std::string weights = cvtest::findDataFile("dnn/gsoc2016-goturn/goturn.caffemodel", false);
  cv::TrackerGOTURN::Params params;
  params.modelTxt = model;
  params.modelBin = weights;
  cv::Ptr<Tracker> tracker = TrackerGOTURN::create(params);

  string inputVideo = cvtest::findDataFile("tracking/david/data/david.webm");
  cv::VideoCapture video(inputVideo);
  ASSERT_TRUE(video.isOpened()) << inputVideo;

  cv::Mat frame;
  video >> frame;
  ASSERT_FALSE(frame.empty()) << inputVideo;
  tracker->init(frame, roi);
  string ground_truth_bb;
  for (int nframes = 0; nframes < 15; ++nframes)
  {
    std::cout << "Frame: " << nframes << std::endl;
    video >> frame;
    bool res = tracker->update(frame, roi);
    ASSERT_TRUE(res);
    std::cout << "Predicted ROI: " << roi << std::endl;
  }
}

}} // namespace
