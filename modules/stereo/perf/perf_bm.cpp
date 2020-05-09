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

#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple<Size, MatType, MatDepth> s_bm_test_t;
typedef perf::TestBaseWithParam<s_bm_test_t> s_bm;

PERF_TEST_P( s_bm, sgm_perf,
            testing::Combine(
            testing::Values( cv::Size(512, 283),  cv::Size(320, 240)),
            testing::Values( CV_8U ),
            testing::Values( CV_8U,CV_16S )
            )
            )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());
    int sdepth = get<2>(GetParam());

    Mat left(sz, matType);
    Mat right(sz, matType);
    Mat out1(sz, sdepth);
    Ptr<StereoBinarySGBM> sgbm = StereoBinarySGBM::create(0, 16, 5);
    sgbm->setBinaryKernelType(CV_DENSE_CENSUS);
    declare
        .in(left, WARMUP_RNG)
        .in(right, WARMUP_RNG)
        .out(out1)
        .time(0.1)
        .iterations(20);
    TEST_CYCLE()
    {
        sgbm->compute(left, right, out1);
    }
    SANITY_CHECK_NOTHING();
}
PERF_TEST_P( s_bm, bm_perf,
            testing::Combine(
            testing::Values( cv::Size(512, 383),  cv::Size(320, 240) ),
            testing::Values( CV_8U ),
            testing::Values( CV_8U )
            )
            )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());
    int sdepth = get<2>(GetParam());

    Mat left(sz, matType);
    Mat right(sz, matType);
    Mat out1(sz, sdepth);
    Ptr<StereoBinaryBM> sbm = StereoBinaryBM::create(16, 9);
    // we set the corresponding parameters
    sbm->setPreFilterCap(31);
    sbm->setMinDisparity(0);
    sbm->setTextureThreshold(10);
    sbm->setUniquenessRatio(0);
    sbm->setSpeckleWindowSize(400);
    sbm->setDisp12MaxDiff(0);
    sbm->setAgregationWindowSize(11);
    // the user can choose between the average speckle removal algorithm or
    // the classical version that was implemented in OpenCV
    sbm->setSpekleRemovalTechnique(CV_SPECKLE_REMOVAL_AVG_ALGORITHM);
    sbm->setUsePrefilter(false);

    declare
        .in(left, WARMUP_RNG)
        .in(right, WARMUP_RNG)
        .out(out1)
        .time(0.1)
        .iterations(20);
    TEST_CYCLE()
    {
        sbm->compute(left, right, out1);
    }
    SANITY_CHECK_NOTHING();
}


}} // namespace
