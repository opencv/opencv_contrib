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
// Copyright (C) 2010-2013, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
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
// This software is provided by the copyright holders and contributors as is and
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
#include "opencv2/ts/ocl_test.hpp"

#define RETINA_ITERATIONS 5

namespace opencv_test { namespace {

PARAM_TEST_CASE(Retina_OCL, bool, int, bool, double, double)
{
    bool colorMode;
    int colorSamplingMethod;
    bool useLogSampling;
    float reductionFactor;
    float samplingStrength;

    virtual void SetUp()
    {
        colorMode           = GET_PARAM(0);
        colorSamplingMethod = GET_PARAM(1);
        useLogSampling      = GET_PARAM(2);
        reductionFactor     = static_cast<float>(GET_PARAM(3));
        samplingStrength    = static_cast<float>(GET_PARAM(4));
    }
};

OCL_TEST_P(Retina_OCL, Accuracy)
{
    Mat input = imread(cvtest::TS::ptr()->get_data_path() + "shared/lena.png", colorMode);
    CV_Assert(!input.empty());

    Ptr<bioinspired::Retina> retina = bioinspired::Retina::create(
        input.size(),
        colorMode,
        colorSamplingMethod,
        useLogSampling,
        reductionFactor,
        samplingStrength);

    Mat gold_parvo;
    Mat gold_magno;
    UMat ocl_parvo;
    UMat ocl_magno;

    for(int i = 0; i < RETINA_ITERATIONS; i ++)
    {
        OCL_OFF(retina->run(input));
        OCL_OFF(retina->getParvo(gold_parvo));
        OCL_OFF(retina->getMagno(gold_magno));
        OCL_OFF(retina->clearBuffers());

        OCL_ON(retina->run(input));
        OCL_ON(retina->getParvo(ocl_parvo));
        OCL_ON(retina->getMagno(ocl_magno));
        OCL_ON(retina->clearBuffers());

        int eps = 1;

        EXPECT_MAT_NEAR(gold_parvo, ocl_parvo, eps);
        EXPECT_MAT_NEAR(gold_magno, ocl_magno, eps);
    }
}

OCL_INSTANTIATE_TEST_CASE_P(Contrib, Retina_OCL, testing::Combine(
                            testing::Bool(),
                            testing::Values((int)cv::bioinspired::RETINA_COLOR_BAYER),
                            testing::Values(false/*,true*/),
                            testing::Values(1.0, 0.5),
                            testing::Values(10.0, 5.0)));

}} // namespace
