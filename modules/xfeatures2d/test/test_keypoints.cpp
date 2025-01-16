/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

const string FEATURES2D_DIR = "features2d";
const string IMAGE_FILENAME = "tsukuba.png";

/****************************************************************************************\
*                                     Test for KeyPoint                                  *
\****************************************************************************************/

class CV_FeatureDetectorKeypointsTest : public cvtest::BaseTest
{
public:
    explicit CV_FeatureDetectorKeypointsTest(const Ptr<Feature2D>& _detector) :
        detector(_detector) {}

protected:
    virtual void run(int)
    {
        CV_Assert(detector);
        string imgFilename = string(ts->get_data_path()) + FEATURES2D_DIR + "/" + IMAGE_FILENAME;

        // Read the test image.
        Mat image = imread(imgFilename);
        if(image.empty())
        {
            ts->printf(cvtest::TS::LOG, "Image %s can not be read.\n", imgFilename.c_str());
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }

        vector<KeyPoint> keypoints;
        detector->detect(image, keypoints);

        if(keypoints.empty())
        {
            ts->printf(cvtest::TS::LOG, "Detector can't find keypoints in image.\n");
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
            return;
        }

        Rect r(0, 0, image.cols, image.rows);
        for(size_t i = 0; i < keypoints.size(); i++)
        {
            const KeyPoint& kp = keypoints[i];

            // Workaround for https://github.com/opencv/opencv/issues/26016
            // To keep its behaviour, kp.pt casts to Point_<int>.
            if(!r.contains(Point_<int>(kp.pt)))
            {
                ts->printf(cvtest::TS::LOG, "KeyPoint::pt is out of image (x=%f, y=%f).\n", kp.pt.x, kp.pt.y);
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                return;
            }

            if(kp.size <= 0.f)
            {
                ts->printf(cvtest::TS::LOG, "KeyPoint::size is not positive (%f).\n", kp.size);
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                return;
            }

            if((kp.angle < 0.f && kp.angle != -1.f) || kp.angle >= 360.f)
            {
                ts->printf(cvtest::TS::LOG, "KeyPoint::angle is out of range [0, 360). It's %f.\n", kp.angle);
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                return;
            }
        }
        ts->set_failed_test_info(cvtest::TS::OK);
    }

    Ptr<Feature2D> detector;
};


// Registration of tests
#ifdef OPENCV_ENABLE_NONFREE
TEST(Features2d_Detector_Keypoints_SURF, validation)
{
    CV_FeatureDetectorKeypointsTest test(xfeatures2d::SURF::create());
    test.safe_run();
}
#endif // NONFREE


TEST(Features2d_Detector_Keypoints_Star, validation)
{
    CV_FeatureDetectorKeypointsTest test(xfeatures2d::StarDetector::create());
    test.safe_run();
}


TEST(Features2d_Detector_Keypoints_MSDDetector, validation)
{
    CV_FeatureDetectorKeypointsTest test(xfeatures2d::MSDDetector::create());
    test.safe_run();
}

TEST(Features2d_Detector_Keypoints_TBMRDetector, validation)
{
    CV_FeatureDetectorKeypointsTest test(xfeatures2d::TBMR::create());
    test.safe_run();
}

}} // namespace
