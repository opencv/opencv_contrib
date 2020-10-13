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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
#include "cvconfig.h"
#include "opencv2/ts/ocl_test.hpp"

namespace opencv_test {

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
        Mat image;
        image = imread(ts->get_data_path() + "../cv/stereomatching/datasets/tsukuba/im2.png", IMREAD_GRAYSCALE);

        if (image.empty())
        {
            ts->printf(cvtest::TS::LOG, "Wrong input data \n");
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }

        Ptr<TBMR> tbmr = TBMR::create(30, 0.01);
        // set the corresponding parameters
        tbmr->setMinArea(30);
        tbmr->setMaxAreaRelative(0.01);

        std::vector<KeyPoint> tbmrs;
        tbmr->detectRegions(image, tbmrs);

        if (tbmrs.size() != 351)
        {
            ts->printf(cvtest::TS::LOG, "Invalid result \n");
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
            return;
        }
    }

    TEST(tbmr_simple_test, accuracy) { CV_TBMRTest test; test.safe_run(); }
#ifdef HAVE_OPENCL

    namespace ocl {

        //OCL_TEST_F(cv::tbmr::TBMR, ABC1)
        //{
        //    // RunTest<cv::UMat>(cv::superres::createSuperResolution_BTVL1());
        //}

    } // namespace opencv_test::ocl

#endif

} // namespace
