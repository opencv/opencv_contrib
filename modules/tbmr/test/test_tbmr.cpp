/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
license.
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
// Redistribution and use in source and binary forms, with or without
modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright
notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote
products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is"
and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are
disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any
direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "cvconfig.h"
#include "test_precomp.hpp"

namespace opencv_test
{

class CV_TBMRTest : public cvtest::BaseTest
{
  public:
    CV_TBMRTest();
    ~CV_TBMRTest();

  protected:
    void run(int /* idx */);
};

CV_TBMRTest::CV_TBMRTest() {}
CV_TBMRTest::~CV_TBMRTest() {}

void CV_TBMRTest::run(int)
{
    using namespace tbmr;

    Mat image(100, 100, CV_8UC1);
    image.setTo(Scalar(0));
    Point2f feature(50, 50);
    double feature_angle = 5 * CV_PI / 180.;

    ellipse(image, feature, Size2f(40, 30), feature_angle * 180. / CV_PI, 0,
            360, Scalar(255), -1);
    ellipse(image, Point2f(40, 50), Size2f(10, 5), 0, 0, 360, Scalar(128), -1);
    ellipse(image, Point2f(70, 50), Size2f(10, 5), 0, 0, 360, Scalar(98), -1);

    Ptr<TBMR> tbmr = TBMR::create(30, 0.01);

    // set the corresponding parameters
    tbmr->setMinArea(30);

    // For normal images we want to set maxAreaRelative to around 0.01. Here we
    // have a feature bigger than half of the image.
    tbmr->setMaxAreaRelative(0.9);

    std::vector<KeyPoint> tbmrs;
    tbmr->detect(image, tbmrs);

    if (tbmrs.size() != 1)

    {
        ts->printf(cvtest::TS::LOG, "Invalid result \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        return;
    }

    double feature_diff = cv::norm(tbmrs.at(0).pt - feature);
    double angle_diff = (tbmrs.at(0).angle - feature_angle);

    if (feature_diff > 0.1 || angle_diff > 0.01)

    {
        ts->printf(cvtest::TS::LOG, "Incorrect result \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        return;
    }
}

TEST(tbmr_simple_test, accuracy)
{
    CV_TBMRTest test;
    test.safe_run();
}

} // namespace opencv_test
