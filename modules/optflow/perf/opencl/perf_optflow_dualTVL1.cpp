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

#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

#undef VISUALIZE_FLOW
//#define VISUALIZE_FLOW 1

#ifdef VISUALIZE_FLOW
// [TODO] move it to opencv_video/tracking.hpp
// Standard visualization of the optical flow, according to Scharstein/Middlebury (ICCV 2007).
// flow — input flow of CV_32FC2 type (field of motion vectors)
// dst  — output visual optical flow representation of CV_8UC3 type
// maxFlow — max flow magnitude or (if <=0, it's computed automatically; use with care,
//                                  because it 'boosts' flow when there is almost no one).
//           pixels where the motion vector magnitude is equal or exceeds maxFlow will be
//           painted with maximum saturation.
// eps — maxFlow threshold for 'no motion/still scene' cases
// computedMaxFlow — the optional output value to hold the computed maxFlow, to let user
//                   gradually calibrate maxFlow parameter.
static void tsVisualizeFlow(InputArray flowarr, OutputArray dstarr,
                            float maxFlow0 = -1.f, float eps = 1e-3f,
                            float* computedMaxFlow=nullptr)
{
    Mat flow = flowarr.getMat();
    CV_Assert(flow.type() == CV_32FC2);

    // --- color ring (55 colors, RGB) ---
    // sectors: RY=15, YG=6, GC=4, CB=11, BM=13, MR=6
    constexpr int RY=15, YG=6, GC=4, CB=11, BM=13, MR=6;
    constexpr int NCOLS = RY+YG+GC+CB+BM+MR; // 55

    std::array<Vec3b, NCOLS+1> cwheel_; // RGB
    int k = 0;
    for (int i=0;i<RY;i++,k++) cwheel_[k]=Vec3b(0, uint8_t(255*i/RY), 255);
    for (int i=0;i<YG;i++,k++) cwheel_[k]=Vec3b(0, 255, uint8_t(255-255*i/YG));
    for (int i=0;i<GC;i++,k++) cwheel_[k]=Vec3b(uint8_t(255*i/GC), 255, 0);
    for (int i=0;i<CB;i++,k++) cwheel_[k]=Vec3b(255, uint8_t(255-255*i/CB), 0);
    for (int i=0;i<BM;i++,k++) cwheel_[k]=Vec3b(255, 0, uint8_t(255*i/BM));
    for (int i=0;i<MR;i++,k++) cwheel_[k]=Vec3b(uint8_t(255-255*i/MR), 0, 255);
    cwheel_[NCOLS] = cwheel_[0];

    std::vector<float> maxvals(flow.rows);

    // --- compute max flow automatically ---
    if (maxFlow0 <= 0.0f) {
        parallel_for_(cv::Range(0, flow.rows), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; y++) {
                float maxval = 0.f;
                const cv::Vec2f* row = flow.ptr<cv::Vec2f>(y);
                for (int x = 0; x < flow.cols; x++) {
                    float dx = row[x][0], dy = row[x][1];
                    float mag = std::hypot(dx, dy);
                    maxval = std::max(maxval, mag);
                }
                maxvals[y] = maxval;
            }
        });
        maxFlow0 = 0.f;
        for (int y = 0; y < flow.rows; y++)
            maxFlow0 = std::max(maxFlow0, maxvals[y]);
    }

    maxFlow0 = maxFlow0 > eps ? maxFlow0 : 1.f;
    dstarr.create(flow.size(), CV_8UC3);
    Mat dst = dstarr.getMat();

    // paint the optical flow map
    parallel_for_(cv::Range(0, flow.rows), [&](const cv::Range& range) {
        const Vec3b* cwheel = cwheel_.data();
        float maxval = 0.f, maxflow = maxFlow0;
        for (int y = range.start; y < range.end; y++) {
            const cv::Vec2f* src = flow.ptr<cv::Vec2f>(y);
            cv::Vec3b*       out = dst.ptr<cv::Vec3b>(y);

            for (int x = 0; x < flow.cols; x++) {
                float dx = src[x][0];
                float dy = src[x][1];
                float mag = std::hypot(dx, dy);
                maxval = std::max(maxval, mag);
                mag = std::min(mag / maxflow, 1.f);

                // compute the color from angle
                float a  = std::atan2(-dy, -dx) / static_cast<float>(M_PI);
                float f = (a + 1.0f) * 0.5f * (NCOLS - 1);
                int idx = (int)f;
                f -= idx;

                // 'white' means no motion,
                // the stronger the motion the more saturated the corresponding pixel is.
                Vec3b clr;
                for (int c = 0; c < 3; c++) {
                    float chval = cwheel[idx][c] * (1.f - f) + cwheel[idx + 1][c] * f;
                    chval = 255.f - mag * (255.f - chval);
                    clr[c] = saturate_cast<uint8_t>(chval);
                }
                out[x] = clr;
            }
            maxvals[y] = maxval;
        }
    });

    if (computedMaxFlow) {
        maxFlow0 = 0.f;
        for (int y = 0; y < flow.rows; y++)
            maxFlow0 = std::max(maxFlow0, maxvals[y]);
        *computedMaxFlow = maxFlow0;
    }
}
#endif

///////////// OpticalFlow Dual TVL1 ////////////////////////
typedef tuple< tuple<int, double>, bool> OpticalFlowDualTVL1Params;
typedef TestBaseWithParam<OpticalFlowDualTVL1Params> OpticalFlowDualTVL1Fixture;

OCL_PERF_TEST_P(OpticalFlowDualTVL1Fixture, OpticalFlowDualTVL1,
            ::testing::Combine(
                        ::testing::Values(make_tuple<int, double>(-1, 0.3),
                                          make_tuple<int, double>(3, 0.5)),
                        ::testing::Bool()
                                )
            )
    {
        Mat frame0 = imread(getDataPath("cv/optflow/RubberWhale1.png"), IMREAD_GRAYSCALE);
        ASSERT_FALSE(frame0.empty()) << "can't load RubberWhale1.png";

        Mat frame1 = imread(getDataPath("cv/optflow/RubberWhale2.png"), IMREAD_GRAYSCALE);
        ASSERT_FALSE(frame1.empty()) << "can't load RubberWhale2.png";

        const Size srcSize = frame0.size();

        const OpticalFlowDualTVL1Params params = GetParam();
        const tuple<int, double> filteringScale = get<0>(params);
            const int medianFiltering = get<0>(filteringScale);
            const double scaleStep = get<1>(filteringScale);
        const bool useInitFlow = get<1>(params);
        double eps = 0.9;

        UMat uFrame0; frame0.copyTo(uFrame0);
        UMat uFrame1; frame1.copyTo(uFrame1);
        UMat uFlow(srcSize, CV_32FC2);
        declare.in(uFrame0, uFrame1, WARMUP_READ).out(uFlow, WARMUP_READ);

        //create algorithm
        Ptr<DualTVL1OpticalFlow> alg = createOptFlow_DualTVL1();

        //set parameters
        alg->setScaleStep(scaleStep);
        alg->setMedianFiltering(medianFiltering);

        if (useInitFlow)
        {
            //calculate initial flow as result of optical flow
            alg->calc(uFrame0, uFrame1, uFlow);
        }

        //set flag to use initial flow
        alg->setUseInitialFlow(useInitFlow);
        OCL_TEST_CYCLE()
            alg->calc(uFrame0, uFrame1, uFlow);

    #ifdef VISUALIZE_FLOW
        imshow("frame0", uFrame0);
        UMat framediff;
        absdiff(uFrame0, uFrame1, framediff);
        imshow("framediff", framediff);

        Mat flow8u;
        tsVisualizeFlow(uFlow, flow8u);
        imshow("uFlow", flow8u);
        waitKey();
    #endif

        SANITY_CHECK(uFlow, eps, ERROR_RELATIVE);
    }
}

} // namespace opencv_test::ocl

#endif // HAVE_OPENCL
