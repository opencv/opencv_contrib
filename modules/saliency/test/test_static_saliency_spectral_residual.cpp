/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(CV_StaticSaliencySpectralResidual, should_not_contain_nan)
{
    Ptr<StaticSaliencySpectralResidual> saliencyAlgorithm = StaticSaliencySpectralResidual::create();
    Mat img = Mat::zeros(cv::Size(1, 1), CV_32F);
    Mat saliencyMap;

    saliencyAlgorithm->computeSaliency(img, saliencyMap);
    EXPECT_FALSE(std::isnan(saliencyMap.at<float>(0, 0)));
}

TEST(CV_MotionSaliencyBinWangApr2014, synthetic_sequence_detection_and_template_update)
{
    RNG& rng = theRNG();
    const int W = 64, H = 48;
    Ptr<MotionSaliencyBinWangApr2014> ms = MotionSaliencyBinWangApr2014::create();
    ms->setImagesize(W, H);
    ASSERT_TRUE(ms->init());

    Mat bg(H, W, CV_8UC1);
    rng.fill(bg, RNG::UNIFORM, 60, 100);

    Mat salMap;
    // Warm up background templates and trigger templateReplacement + templateOrdering
    for (int f = 0; f < 60; ++f)
    {
        Mat frame = bg.clone();
        ASSERT_TRUE(ms->computeSaliency(frame, salMap));
        ASSERT_EQ(Size(W, H), salMap.size());
        ASSERT_EQ(CV_8UC1, salMap.type());
    }

    // Introduce a bright moving square and verify salient foreground detection
    Mat moving = bg.clone();
    rectangle(moving, Rect(16, 12, 16, 16), Scalar(240), FILLED);
    ASSERT_TRUE(ms->computeSaliency(moving, salMap));
    EXPECT_GT(countNonZero(salMap(Rect(16, 12, 16, 16))), 64);
    EXPECT_EQ(0, countNonZero(salMap(Rect(0, 0, 8, 8))));
}

}} // namespace
