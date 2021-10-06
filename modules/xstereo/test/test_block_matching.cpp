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

class CV_BlockMatchingTest : public cvtest::BaseTest
{
public:
    CV_BlockMatchingTest();
    ~CV_BlockMatchingTest();
protected:
    void run(int /* idx */);
};

CV_BlockMatchingTest::CV_BlockMatchingTest(){}
CV_BlockMatchingTest::~CV_BlockMatchingTest(){}

static double errorLevel(const Mat &ideal, Mat &actual)
{
    uint8_t *date, *harta;
    harta = actual.data;
    date = ideal.data;
    int stride, h;
    stride = (int)ideal.step;
    h = ideal.rows;
    int error = 0;
    for (int i = 0; i < ideal.rows; i++)
    {
        for (int j = 0; j < ideal.cols; j++)
        {
            if (date[i * stride + j] != 0)
                if (abs(date[i * stride + j] - harta[i * stride + j]) > 2 * 16)
                {
                    error += 1;
                }
        }
    }
    return ((double)((error * 100) * 1.0) / (stride * h));
}
void CV_BlockMatchingTest::run(int )
{
    Mat image1, image2, gt;
    image1 = imread(ts->get_data_path() + "stereomatching/datasets/tsukuba/im2.png", IMREAD_GRAYSCALE);
    image2 = imread(ts->get_data_path() + "stereomatching/datasets/tsukuba/im6.png", IMREAD_GRAYSCALE);
    gt = imread(ts->get_data_path() + "stereomatching/datasets/tsukuba/disp2.png", IMREAD_GRAYSCALE);

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

    RNG range;
    //set the parameters
    int binary_descriptor_type = range.uniform(0,8);
    int kernel_size, aggregation_window;
    if(binary_descriptor_type == 0)
        kernel_size = 5;
    else if(binary_descriptor_type == 2 || binary_descriptor_type == 3)
        kernel_size = 7;
    else if(binary_descriptor_type == 1)
        kernel_size = 11;
    else
        kernel_size = 9;
    if(binary_descriptor_type == 3)
        aggregation_window = 13;
    else
        aggregation_window = 11;
    Mat test = Mat(image1.rows, image1.cols, CV_8UC1);
    Ptr<StereoBinaryBM> sbm = StereoBinaryBM::create(16, kernel_size);
    //we set the corresponding parameters
    sbm->setPreFilterCap(31);
    sbm->setMinDisparity(0);
    sbm->setTextureThreshold(10);
    sbm->setUniquenessRatio(0);
    sbm->setSpeckleWindowSize(400);//speckle size
    sbm->setSpeckleRange(200);
    sbm->setDisp12MaxDiff(0);
    sbm->setScalleFactor(16);//the scaling factor
    sbm->setBinaryKernelType(binary_descriptor_type);//binary descriptor kernel
    sbm->setAgregationWindowSize(aggregation_window);
    //speckle removal algorithm the user can choose between the average speckle removal algorithm
    //or the classical version that was implemented in open cv
    sbm->setSpekleRemovalTechnique(CV_SPECKLE_REMOVAL_AVG_ALGORITHM);
    sbm->setUsePrefilter(false);//pre-filter or not the images prior to making the transformations
    //-- calculate the disparity image
    sbm->compute(image1, image2, test);
    if(test.empty())
    {
        ts->printf(cvtest::TS::LOG, "Wrong input / output dimension \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        return;
    }
    if(errorLevel(gt,test) > 20)
    {
        ts->printf( cvtest::TS::LOG,
            "Too big error\n");
        ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        return;
    }
}

class CV_SGBlockMatchingTest : public cvtest::BaseTest
{
public:
    CV_SGBlockMatchingTest();
    ~CV_SGBlockMatchingTest();
protected:
    void run(int /* idx */);
};

CV_SGBlockMatchingTest::CV_SGBlockMatchingTest(){}
CV_SGBlockMatchingTest::~CV_SGBlockMatchingTest(){}

void CV_SGBlockMatchingTest::run(int )
{
    Mat image1, image2, gt;
    image1 = imread(ts->get_data_path() + "stereomatching/datasets/tsukuba/im2.png", IMREAD_GRAYSCALE);
    image2 = imread(ts->get_data_path() + "stereomatching/datasets/tsukuba/im6.png", IMREAD_GRAYSCALE);
    gt = imread(ts->get_data_path() + "stereomatching/datasets/tsukuba/disp2.png", IMREAD_GRAYSCALE);

    ts->printf(cvtest::TS::LOG,(ts->get_data_path() + "stereomatching/datasets/tsukuba/im2.png").c_str());
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

    RNG range;
    //set the parameters
    int binary_descriptor_type = range.uniform(0,8);
    int kernel_size;
    if(binary_descriptor_type == 0)
        kernel_size = 5;
    else if(binary_descriptor_type == 2 || binary_descriptor_type == 3)
        kernel_size = 7;
    else if(binary_descriptor_type == 1)
        kernel_size = 11;
    else
        kernel_size = 9;

    Mat test = Mat(image1.rows, image1.cols, CV_8UC1);
    Mat imgDisparity16S2 = Mat(image1.rows, image1.cols, CV_16S);
    Ptr<StereoBinarySGBM> sgbm = StereoBinarySGBM::create(0, 16, kernel_size);
    //setting the penalties for sgbm
    sgbm->setP1(10);
    sgbm->setP2(100);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(16);//set disparity number
    sgbm->setUniquenessRatio(1);
    sgbm->setSpeckleWindowSize(400);
    sgbm->setSpeckleRange(200);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setBinaryKernelType(binary_descriptor_type);//set the binary descriptor
    sgbm->setSpekleRemovalTechnique(CV_SPECKLE_REMOVAL_AVG_ALGORITHM); //the avg speckle removal algorithm
    sgbm->setSubPixelInterpolationMethod(CV_SIMETRICV_INTERPOLATION);// the SIMETRIC V interpolation method
    sgbm->compute(image1, image2, imgDisparity16S2);
    double minVal; double maxVal;
    minMaxLoc(imgDisparity16S2, &minVal, &maxVal);

    imgDisparity16S2.convertTo(test, CV_8UC1, 255 / (maxVal - minVal));

    if(test.empty())
    {
        ts->printf(cvtest::TS::LOG, "Wrong input / output dimension \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        return;
    }
    double error = errorLevel(gt,test);
    if(error > 10)
    {
        ts->printf( cvtest::TS::LOG,
            "Too big error\n");
        ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        return;
    }
}
TEST(block_matching_simple_test, accuracy) { CV_BlockMatchingTest test; test.safe_run(); }
TEST(SG_block_matching_simple_test, accuracy) { CV_SGBlockMatchingTest test; test.safe_run(); }


}} // namespace