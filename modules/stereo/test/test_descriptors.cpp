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

class CV_DescriptorBaseTest : public cvtest::BaseTest
{
public:
    CV_DescriptorBaseTest();
    ~CV_DescriptorBaseTest();
protected:
    virtual void imageTransformation(const Mat &img1, const Mat &img2, Mat &out1, Mat &out2) = 0;
    virtual void imageTransformation(const Mat &img1, Mat &out1) = 0;
    void testROI(const Mat &img);
    void testMonotonicity(const Mat &img, Mat &out);
    void run(int );
    Mat censusImage[2];
    Mat censusImageSingle[2];
    Mat left;
    Mat right;
    int kernel_size, descriptor_type;
};
//we test to see if the descriptor applied on a roi
//has the same value with the descriptor from the original image
//tested at the roi boundaries
void CV_DescriptorBaseTest::testROI(const Mat &img)
{
    int pt, pb,w,h;
    //initialize random values for the roi top and bottom
    pt = rand() % 100;
    pb = rand() % 100;
    //calculate the new width and height
    w = img.cols;
    h = img.rows - pt - pb;
    int start = pt + kernel_size / 2 + 1;
    int stop = h - kernel_size/2 - 1;
    //set the region of interest according to above values
    Rect region_of_interest = Rect(0, pt, w, h);
    Mat image_roi1 = img(region_of_interest);
    Mat p1,p2;
    //create 2 images where to put our output
    p1.create(image_roi1.rows, image_roi1.cols, CV_32SC4);
    p2.create(img.rows, img.cols, CV_32SC4);
    imageTransformation(image_roi1,p1);
    imageTransformation(img,p2);
    int *roi_data = (int *)p1.data;
    int *img_data = (int *)p2.data;
    //verify result
    for(int i = start; i < stop; i++)
    {
        for(int j = 0; j < w ; j++)
        {
            if(roi_data[(i - pt) * w + j] != img_data[(i) * w + j])
            {
                ts->printf(cvtest::TS::LOG, "Something wrong with ROI \n");
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                return;
            }
        }

    }
}
CV_DescriptorBaseTest::~CV_DescriptorBaseTest()
{
    left.release();
    right.release();
    censusImage[0].release();
    censusImage[1].release();
    censusImageSingle[0].release();
    censusImageSingle[1].release();
}
CV_DescriptorBaseTest::CV_DescriptorBaseTest()
{
    //read 2 images from file
    left = imread(ts->get_data_path() + "stereomatching/datasets/tsukuba/im2.png", IMREAD_GRAYSCALE);
    right = imread(ts->get_data_path() + "stereomatching/datasets/tsukuba/im6.png", IMREAD_GRAYSCALE);

    if(left.empty() || right.empty())
    {
        ts->printf(cvtest::TS::LOG, "Wrong input data \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    ts->printf(cvtest::TS::LOG, "Data loaded \n");
}
//verify if we don't have an image with all pixels the same( except when all input pixels are equal)
void CV_DescriptorBaseTest::testMonotonicity(const Mat &img, Mat &out)
{
    //verify if input data is correct
    if(img.rows != out.rows || img.cols != out.cols || img.empty() || out.empty())
    {
        ts->printf(cvtest::TS::LOG, "Wrong input / output dimension \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    //verify that for an input image with different pxels the values of the
    //output pixels are not the same
    int same = 0;
    uint8_t *data = img.data;
    uint8_t val = data[1];
    int stride = (int)img.step;
    for(int i = 0 ; i < img.rows && !same; i++)
    {
        for(int j = 0; j < img.cols; j++)
        {
            if(val != data[i * stride + j])
            {
                same = 1;
                break;
            }
        }
    }
    int value_descript = out.data[1];
    int accept = 0;
    uint8_t *outData = out.data;
    for(int i = 0 ; i < img.rows && !accept; i++)
    {
        for(int j = 0; j < img.cols; j++)
        {
            //we verify for the output image if the iage pixels are not all the same of an input
            //image with different pixels
            if(value_descript != outData[i * stride + j] && same)
            {
                //if we found a value that is different we accept
                accept = 1;
                break;
            }
        }
    }
    if(accept == 1 && same == 0)
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        ts->printf(cvtest::TS::LOG, "The image has all values the same \n");
        return;
    }
    if(accept == 0 && same == 1)
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        ts->printf(cvtest::TS::LOG, "For correct image we get all descriptor values the same \n");
        return;
    }
    ts->set_failed_test_info(cvtest::TS::OK);
}

///////////////////////////////////
//census transform

class CV_CensusTransformTest: public CV_DescriptorBaseTest
{
public:
    CV_CensusTransformTest();
protected:
    void imageTransformation(const Mat &img1, const Mat &img2, Mat &out1, Mat &out2);
    void imageTransformation(const Mat &img1, Mat &out1);
};

CV_CensusTransformTest::CV_CensusTransformTest()
{
    kernel_size = 11;
    descriptor_type = CV_SPARSE_CENSUS;
}
void CV_CensusTransformTest::imageTransformation(const Mat &img1, const Mat &img2, Mat &out1, Mat &out2)
{
    //verify if input data is correct
    if(img1.rows != out1.rows || img1.cols != out1.cols || img1.empty() || out1.empty()
        || img2.rows != out2.rows || img2.cols != out2.cols || img2.empty() || out2.empty())
    {
        ts->printf(cvtest::TS::LOG, "Wrong input / output data \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    if(kernel_size % 2 == 0)
    {
        ts->printf(cvtest::TS::LOG, "Wrong kernel size;Kernel should be odd \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    censusTransform(img1,img2,kernel_size,out1,out2,descriptor_type);

}
void CV_CensusTransformTest::imageTransformation(const Mat &img1, Mat &out1)
{
    //verify if input data is correct
    if(img1.rows != out1.rows || img1.cols != out1.cols || img1.empty() || out1.empty())
    {
        ts->printf(cvtest::TS::LOG, "Wrong input / output data \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    if(kernel_size % 2 == 0)
    {
        ts->printf(cvtest::TS::LOG, "Wrong kernel size;Kernel should be odd \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    censusTransform(img1,kernel_size,out1,descriptor_type);
}
//////////////////////////////////
//symetric census

class CV_SymetricCensusTest: public CV_DescriptorBaseTest
{
public:
    CV_SymetricCensusTest();
protected:
    void imageTransformation(const Mat &img1, const Mat &img2, Mat &out1, Mat &out2);
    void imageTransformation(const Mat &img1, Mat &out1);
};
CV_SymetricCensusTest::CV_SymetricCensusTest()
{
    kernel_size = 7;
    descriptor_type = CV_CS_CENSUS;
}
void CV_SymetricCensusTest::imageTransformation(const Mat &img1, const Mat &img2, Mat &out1, Mat &out2)
{
    //verify if input data is correct
    if(img1.rows != out1.rows || img1.cols != out1.cols || img1.empty() || out1.empty()
        || img2.rows != out2.rows || img2.cols != out2.cols || img2.empty() || out2.empty())
    {
        ts->printf(cvtest::TS::LOG, "Wrong input / output data \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    if(kernel_size % 2 == 0)
    {
        ts->printf(cvtest::TS::LOG, "Wrong kernel size;Kernel should be odd \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    symetricCensusTransform(img1,img2,kernel_size,out1,out2,descriptor_type);
}
void CV_SymetricCensusTest::imageTransformation(const Mat &img1, Mat &out1)
{
    //verify if input data is correct
    if(img1.rows != out1.rows || img1.cols != out1.cols || img1.empty() || out1.empty())
    {
        ts->printf(cvtest::TS::LOG, "Wrong input / output data \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    if(kernel_size % 2 == 0)
    {
        ts->printf(cvtest::TS::LOG, "Wrong kernel size;Kernel should be odd \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    symetricCensusTransform(img1,kernel_size,out1,descriptor_type);
}
//////////////////////////////////
//modified census transform
class CV_ModifiedCensusTransformTest: public CV_DescriptorBaseTest
{
public:
    CV_ModifiedCensusTransformTest();
protected:
    void imageTransformation(const Mat &img1, const Mat &img2, Mat &out1, Mat &out2);
    void imageTransformation(const Mat &img1, Mat &out1);
};
CV_ModifiedCensusTransformTest::CV_ModifiedCensusTransformTest()
{
    kernel_size = 9;
    descriptor_type = CV_MODIFIED_CENSUS_TRANSFORM;
}
void CV_ModifiedCensusTransformTest::imageTransformation(const Mat &img1, const Mat &img2, Mat &out1, Mat &out2)
{
    //verify if input data is correct
    if(img1.rows != out1.rows || img1.cols != out1.cols || img1.empty() || out1.empty()
        || img2.rows != out2.rows || img2.cols != out2.cols || img2.empty() || out2.empty())
    {
        ts->printf(cvtest::TS::LOG, "Wrong input / output data \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    if(kernel_size % 2 == 0)
    {
        ts->printf(cvtest::TS::LOG, "Wrong kernel size;Kernel should be odd \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    modifiedCensusTransform(img1,img2,kernel_size,out1,out2,descriptor_type);
}
void CV_ModifiedCensusTransformTest::imageTransformation(const Mat &img1, Mat &out1)
{
    if(img1.rows != out1.rows || img1.cols != out1.cols || img1.empty() || out1.empty())
    {
        ts->printf(cvtest::TS::LOG, "Wrong input / output data \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    if(kernel_size % 2 == 0)
    {
        ts->printf(cvtest::TS::LOG, "Wrong kernel size;Kernel should be odd \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    modifiedCensusTransform(img1,kernel_size,out1,descriptor_type);
}
//////////////////////////////////
//star kernel census
class CV_StarKernelCensusTest: public CV_DescriptorBaseTest
{
public:
    CV_StarKernelCensusTest();
protected:
    void imageTransformation(const Mat &img1, const Mat &img2, Mat &out1, Mat &out2);
    void imageTransformation(const Mat &img1, Mat &out1);
};
CV_StarKernelCensusTest :: CV_StarKernelCensusTest()
{
    kernel_size = 9;
    descriptor_type = CV_STAR_KERNEL;
}
void CV_StarKernelCensusTest :: imageTransformation(const Mat &img1, const Mat &img2, Mat &out1, Mat &out2)
{
    //verify if input data is correct
    if(img1.rows != out1.rows || img1.cols != out1.cols || img1.empty() || out1.empty()
        || img2.rows != out2.rows || img2.cols != out2.cols || img2.empty() || out2.empty())
    {
        ts->printf(cvtest::TS::LOG, "Wrong input / output data \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    if(kernel_size % 2 == 0)
    {
        ts->printf(cvtest::TS::LOG, "Wrong kernel size;Kernel should be odd \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    starCensusTransform(img1,img2,kernel_size,out1,out2);
}
void CV_StarKernelCensusTest::imageTransformation(const Mat &img1, Mat &out1)
{
    if(img1.rows != out1.rows || img1.cols != out1.cols || img1.empty() || out1.empty())
    {
        ts->printf(cvtest::TS::LOG, "Wrong input / output data \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    if(kernel_size % 2 == 0)
    {
        ts->printf(cvtest::TS::LOG, "Wrong kernel size;Kernel should be odd \n");
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    starCensusTransform(img1,kernel_size,out1);
}

void CV_DescriptorBaseTest::run(int )
{
    if (left.empty() || right.empty())
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        ts->printf(cvtest::TS::LOG, "No input images detected\n");
        return;
    }
    testROI(left);

    censusImage[0].create(left.rows, left.cols, CV_32SC4);
    censusImage[1].create(left.rows, left.cols, CV_32SC4);
    censusImageSingle[0].create(left.rows, left.cols, CV_32SC4);
    censusImageSingle[1].create(left.rows, left.cols, CV_32SC4);
    censusImage[0].setTo(0);
    censusImage[1].setTo(0);
    censusImageSingle[0].setTo(0);
    censusImageSingle[1].setTo(0);

    imageTransformation(left, right, censusImage[0], censusImage[1]);
    imageTransformation(left, censusImageSingle[0]);
    imageTransformation(right, censusImageSingle[1]);
    testMonotonicity(left,censusImage[0]);
    testMonotonicity(right,censusImage[1]);
    testMonotonicity(left,censusImageSingle[0]);
    testMonotonicity(right,censusImageSingle[1]);

    if (censusImage[0].empty() || censusImage[1].empty() || censusImageSingle[0].empty() || censusImageSingle[1].empty())
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        ts->printf(cvtest::TS::LOG, "The descriptor images are empty \n");
        return;
    }
    int *datl1 = (int *)censusImage[0].data;
    int *datr1 = (int *)censusImage[1].data;
    int *datl2 = (int *)censusImageSingle[0].data;
    int *datr2 = (int *)censusImageSingle[1].data;
    for(int i = 0; i < censusImage[0].rows - kernel_size/ 2; i++)
    {
        for(int j = 0; j < censusImage[0].cols; j++)
        {
            if(datl1[i * censusImage[0].cols + j] != datl2[i * censusImage[0].cols + j])
            {
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                ts->printf(cvtest::TS::LOG, "Mismatch for left images %d \n",descriptor_type);
                return;
            }
            if(datr1[i * censusImage[0].cols + j] != datr2[i * censusImage[0].cols + j])
            {
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                ts->printf(cvtest::TS::LOG, "Mismatch for right images %d \n",descriptor_type);
                return;
            }
        }
    }
    int min = std::numeric_limits<int>::min();
    int max = std::numeric_limits<int>::max();
    //check if all values are between int min and int max and not NAN
    if (0 != cvtest::check(censusImage[0], min, max, 0))
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }
    //check if all values are between int min and int max and not NAN
    if (0 != cvtest::check(censusImage[1], min, max, 0))
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return ;
    }
}

TEST(DISABLED_census_transform_testing, accuracy) { CV_CensusTransformTest test; test.safe_run(); }
TEST(DISABLED_symetric_census_testing, accuracy) { CV_SymetricCensusTest test; test.safe_run(); }
TEST(DISABLED_Dmodified_census_testing, accuracy) { CV_ModifiedCensusTransformTest test; test.safe_run(); }
TEST(DISABLED_Dstar_kernel_testing, accuracy) { CV_StarKernelCensusTest test; test.safe_run(); }


}} // namespace
