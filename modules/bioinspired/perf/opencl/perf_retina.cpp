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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Fangfang Bai, fangfang@multicorewareinc.com
//    Jin Ma,       jin@multicorewareinc.com
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

#include "perf_precomp.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/ocl.hpp"

#ifdef HAVE_OPENCV_OCL

#include "opencv2/ocl.hpp"

using namespace std::tr1;
using namespace cv;
using namespace perf;

namespace cvtest {
namespace ocl {

///////////////////////// Retina ////////////////////////

typedef tuple<bool, int, double, double> RetinaParams;
typedef TestBaseWithParam<RetinaParams> RetinaFixture;

#define OCL_TEST_CYCLE() for(; startTimer(), next(); cv::ocl::finish(), stopTimer())

PERF_TEST_P(RetinaFixture, Retina,
            ::testing::Combine(testing::Bool(), testing::Values((int)cv::bioinspired::RETINA_COLOR_BAYER),
                               testing::Values(1.0, 0.5), testing::Values(10.0, 5.0)))
{
    if (!cv::ocl::haveOpenCL())
        throw TestBase::PerfSkipTestException();

    RetinaParams params = GetParam();
    bool colorMode = get<0>(params), useLogSampling = false;
    int colorSamplingMethod = get<1>(params);
    double reductionFactor = get<2>(params), samplingStrength = get<3>(params);

    Mat input = cv::imread(cvtest::TS::ptr()->get_data_path() + "shared/lena.png", colorMode);
    ASSERT_FALSE(input.empty());

    Mat gold_parvo, gold_magno;

    if (getSelectedImpl() == "plain")
    {
        Ptr<bioinspired::Retina> gold_retina = bioinspired::createRetina(
            input.size(), colorMode, colorSamplingMethod,
            useLogSampling, reductionFactor, samplingStrength);

        TEST_CYCLE()
        {
            gold_retina->run(input);

            gold_retina->getParvo(gold_parvo);
            gold_retina->getMagno(gold_magno);
        }
    }
    else if (getSelectedImpl() == "ocl")
    {
        cv::ocl::oclMat ocl_input(input), ocl_parvo, ocl_magno;

        Ptr<cv::bioinspired::Retina> ocl_retina = cv::bioinspired::createRetina_OCL(
            input.size(), colorMode, colorSamplingMethod, useLogSampling,
            reductionFactor, samplingStrength);

        OCL_TEST_CYCLE()
        {
            ocl_retina->run(ocl_input);

            ocl_retina->getParvo(ocl_parvo);
            ocl_retina->getMagno(ocl_magno);
        }
    }
    else
        CV_TEST_FAIL_NO_IMPL();

    SANITY_CHECK_NOTHING();
}

} } // namespace cvtest::ocl

#endif // HAVE_OPENCV_OCL
