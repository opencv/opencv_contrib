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

// Check used "cmake" version in case of errors
// Check compiler command line options for <opencv>/modules include
#include "video/test/test_trackers.impl.hpp"


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

INSTANTIATE_TEST_CASE_P(Tracking, DistanceAndOverlap, TESTSET_NAMES);

TEST(Tracking_CSRT, synthetic_sequence_and_border_roi)
{
    Mat base(240, 320, CV_8UC3);
    RNG& rng = theRNG();
    rng.fill(base, RNG::UNIFORM, 20, 80);
    GaussianBlur(base, base, Size(9, 9), 2.0);

    Mat patch(64, 64, CV_8UC3);
    rng.fill(patch, RNG::UNIFORM, 40, 250);
    GaussianBlur(patch, patch, Size(3, 3), 0.8);
    rectangle(patch, Rect(8, 8, 48, 48), Scalar(250, 220, 40), 3);
    circle(patch, Point(32, 32), 16, Scalar(30, 60, 240), -1);
    line(patch, Point(12, 32), Point(52, 32), Scalar(240, 250, 240), 2);
    line(patch, Point(32, 12), Point(32, 52), Scalar(240, 250, 240), 2);

    Ptr<TrackerCSRT> tracker = TrackerCSRT::create();
    Rect bbox(110, 70, 64, 64);
    Mat frame0 = base.clone();
    patch.copyTo(frame0(bbox));
    tracker->init(frame0, bbox);

    for (int step = 1; step <= 5; ++step)
    {
        Mat frame = base.clone();
        Rect gt(bbox.x + step * 2, bbox.y + step, bbox.width, bbox.height);
        patch.copyTo(frame(gt));
        Rect pred;
        ASSERT_TRUE(tracker->update(frame, pred));
        EXPECT_NEAR(pred.x, gt.x, 3);
        EXPECT_NEAR(pred.y, gt.y, 3);
    }

    // Also test target touching and partially outside image border (exercises segmentation_roi clamping)
    Ptr<TrackerCSRT> borderTracker = TrackerCSRT::create();
    Rect borderBox(0, 0, 48, 48);
    Mat borderFrame = base.clone();
    patch(Rect(0, 0, 48, 48)).copyTo(borderFrame(borderBox));
    borderTracker->init(borderFrame, borderBox);
    Rect borderPred;
    EXPECT_TRUE(borderTracker->update(borderFrame, borderPred));
}

}} // namespace
