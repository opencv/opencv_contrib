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

using namespace cv;

/**
 *The expected results of this test case are come from the phash library,
 *I use it as golden model
 */
class cv::img_hash::RadialVarHashTester
{
public:
    cv::Mat getPixPerLine(cv::Mat const &input,
                          cv::img_hash::RadialVarianceHash &rvh) const
    {
        rvh.radialProjections(input);
        return rvh.pixPerLine_;
    }

    cv::Mat getProjection(cv::img_hash::RadialVarianceHash const &rvh) const
    {
        return rvh.projections_;
    }
};

class CV_RadialVarianceHashTest : public cvtest::BaseTest
{
public:
    CV_RadialVarianceHashTest();
    ~CV_RadialVarianceHashTest();
protected:
    void run(int /* idx */);

    void testPixPerLine();
    void testProjection();
    
    cv::Mat input;
    cv::img_hash::RadialVarianceHash rvh;
    cv::img_hash::RadialVarHashTester tester;
};

CV_RadialVarianceHashTest::CV_RadialVarianceHashTest() : 
    rvh(1,1,10)
{
    uchar *inPtr = input.ptr<uchar>(0);
    for(size_t i = 0; i != input.total(); ++i)
    {
        inPtr[i] = i;
    }
}
CV_RadialVarianceHashTest::~CV_RadialVarianceHashTest(){}

void CV_RadialVarianceHashTest::testPixPerLine()
{  
  tester.getPixPerLine(input, rvh);
  uchar const expectResult[] =
  {
    8,8,8,0,8,15,7,5,8,8,
  };
  bool const equal =
          std::equal(expectResult, expectResult + input.total(),
                     input.ptr<uchar>(0));
  if(equal == false)
  {
    ts->printf(cvtest::TS::LOG, "Wrong pixel per line value \n");
    ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
  }
}

void CV_RadialVarianceHashTest::testProjection()
{
    cv::Mat const proj = tester.getProjection(rvh);
    uchar const expectResult[] =
    {
      32,  16,   0,   0,   2,   4,   0,   0,  56,  40,
      33,  17,   9,   0,  10,  59,  58,   0,  49,  41,
      34,  18,  18,   0,  18,  51,  51,  58,  42,  42,
      35,  27,  19,   0,  27,  44,  43,  43,  43,  35,
      36,  36,  36,   0,  36,  36,  36,  36,  36,  36,
      37,  37,  45,   0,  44,  29,  30,  21,  21,  29,
      38,  46,  46,   0,  53,  22,  22,   6,  22,  22,
      39,  47,  55,   0,  61,  14,  15,   0,  15,  23
    };
    bool const equal =
            std::equal(expectResult, expectResult + proj.total(),
                       proj.ptr<uchar>(0));
    if(equal == false)
    {
      ts->printf(cvtest::TS::LOG, "Wrong projection value \n");
      ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
    }
}

void CV_RadialVarianceHashTest::run(int )
{    
    testPixPerLine();
}

TEST(radial_variance_hash_test, accuracy) { CV_RadialVarianceHashTest test; test.safe_run(); }
