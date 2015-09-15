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

#include <string>

using namespace std;
using namespace cv;

/* ///////////////////// sparsetodenseflow_test ///////////////////////// */

class CV_SparseToDenseFlowTest : public cvtest::BaseTest
{
protected:
    void run(int);
};

static bool isFlowCorrect(float u) {
  return !cvIsNaN(u) && (fabs(u) < 1e9);
}

static float calc_rmse(Mat flow1, Mat flow2) {
  float sum = 0;
  int counter = 0;
  const int rows = flow1.rows;
  const int cols = flow1.cols;

  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      Vec2f flow1_at_point = flow1.at<Vec2f>(y, x);
      Vec2f flow2_at_point = flow2.at<Vec2f>(y, x);

      float u1 = flow1_at_point[0];
      float v1 = flow1_at_point[1];
      float u2 = flow2_at_point[0];
      float v2 = flow2_at_point[1];

      if (isFlowCorrect(u1) && isFlowCorrect(u2) && isFlowCorrect(v1) && isFlowCorrect(v2)) {
        sum += (u1-u2)*(u1-u2) + (v1-v2)*(v1-v2);
        counter++;
      }
    }
  }
  return (float)sqrt(sum / (1e-9 + counter));
}

void CV_SparseToDenseFlowTest::run(int) {
    const float MAX_RMSE = 0.6f;
    const string frame1_path = ts->get_data_path() + "optflow/RubberWhale1.png";
    const string frame2_path = ts->get_data_path() + "optflow/RubberWhale2.png";
    const string gt_flow_path = ts->get_data_path() + "optflow/RubberWhale.flo";

    Mat frame1 = imread(frame1_path);
    Mat frame2 = imread(frame2_path);

    if (frame1.empty()) {
      ts->printf(cvtest::TS::LOG, "could not read image %s\n", frame2_path.c_str());
      ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
      return;
    }

    if (frame2.empty()) {
      ts->printf(cvtest::TS::LOG, "could not read image %s\n", frame2_path.c_str());
      ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
      return;
    }

    if (frame1.rows != frame2.rows && frame1.cols != frame2.cols) {
      ts->printf(cvtest::TS::LOG, "images should be of equal sizes (%s and %s)",
                 frame1_path.c_str(), frame2_path.c_str());
      ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
      return;
    }

    if (frame1.type() != 16 || frame2.type() != 16) {
      ts->printf(cvtest::TS::LOG, "images should be of equal type CV_8UC3 (%s and %s)",
                 frame1_path.c_str(), frame2_path.c_str());
      ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
      return;
    }

    Mat flow_gt = optflow::readOpticalFlow(gt_flow_path);
    if(flow_gt.empty()) {
      ts->printf(cvtest::TS::LOG, "error while reading flow data from file %s",
                 gt_flow_path.c_str());
      ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
      return;
    }

    Mat flow;
    optflow::calcOpticalFlowSparseToDense(frame1, frame2, flow);

    float rmse = calc_rmse(flow_gt, flow);

    ts->printf(cvtest::TS::LOG, "Optical flow estimation RMSE for SparseToDenseFlow algorithm : %lf\n",
               rmse);

    if (rmse > MAX_RMSE) {
      ts->printf( cvtest::TS::LOG,
                 "Too big rmse error : %lf ( >= %lf )\n", rmse, MAX_RMSE);
      ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
      return;
    }
}


TEST(Video_OpticalFlowSparseToDenseFlow, accuracy) { CV_SparseToDenseFlowTest test; test.safe_run(); }
