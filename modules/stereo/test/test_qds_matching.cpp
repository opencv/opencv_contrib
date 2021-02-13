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

class CV_QdsMatchingTest : public cvtest::BaseTest
{
public:
    CV_QdsMatchingTest();
    ~CV_QdsMatchingTest();
protected:
    void run(int /* idx */);
};

CV_QdsMatchingTest::CV_QdsMatchingTest(){}
CV_QdsMatchingTest::~CV_QdsMatchingTest(){}

static float disparity_MAE(const Mat &reference, const Mat &estimation)
{
    int elems=0;
    float error=0;
    float ref_invalid=0;
    for (int row=0; row< reference.rows; row++)
    {
        for (int col=0; col<reference.cols; col++)
        {
            float ref_val = reference.at<float>(row, col);
            float estimated_val = estimation.at<float>(row, col);
            if (ref_val == 0){
                ref_invalid++;
            }
            // filter out pixels with unknown reference value and pixels whose disparity did not get estimated.
            if (estimated_val == 0 || ref_val == 0 || std::isnan(estimated_val))
            {
                continue;
            }
            else{
                error+=abs(ref_val - estimated_val);
                elems+=1;
            }
        }
    }
    return error/elems;
}


void CV_QdsMatchingTest::run(int)
{
    //load data
    Mat image1, image2, gt;
    image1 = imread(ts->get_data_path() + "stereomatching/datasets/cones/im2.png", IMREAD_GRAYSCALE);
    image2 = imread(ts->get_data_path() + "stereomatching/datasets/cones/im6.png", IMREAD_GRAYSCALE);
    gt = imread(ts->get_data_path() + "stereomatching/datasets/cones/disp2.png", IMREAD_GRAYSCALE);

    // reference scale factor is based on this https://github.com/opencv/opencv_extra/blob/master/testdata/cv/stereomatching/datasets/datasets.xml
    gt.convertTo(gt, CV_32F);
    gt =gt/4;

    //test inputs
    if(image1.empty() || image2.empty() || gt.empty())
    {
        ts->printf(cvtest::TS::LOG, "Wrong input data \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    if(image1.rows != image2.rows || image1.cols != image2.cols || gt.cols != image1.cols || gt.rows != image1.rows)
    {
        ts->printf(cvtest::TS::LOG, "Wrong input / output dimension \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    //configure disparity algorithm
    cv::Size frameSize = image1.size();
    Ptr<stereo::QuasiDenseStereo> qds_matcher = stereo::QuasiDenseStereo::create(frameSize);


    //compute disparity
    qds_matcher->process(image1, image2);
    Mat outDisp = qds_matcher->getDisparity();


    // test input output size consistency
    if(gt.rows != outDisp.rows || gt.cols != outDisp.cols)
    {
        ts->printf(cvtest::TS::LOG, "Missmatch input output dimension \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        return;
    }

    // test error level, for this sample/hyperparameters, EPE(MAE) should be ~1.1 in version 4.5.1
    double error_mae = disparity_MAE(gt, outDisp);
    if(error_mae > 2)
    {
        ts->printf( cvtest::TS::LOG,("Disparity Mean Absolute Error: "+std::to_string(error_mae)+" pixels > 2\n").c_str());
        ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        return;
    }
}


TEST(qds_matching_simple_test, accuracy) { CV_QdsMatchingTest test; test.safe_run(); }


}} // namespace