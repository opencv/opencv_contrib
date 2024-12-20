/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "perf_precomp.hpp"

namespace opencv_test {

// we use such nested structure to combine test values
typedef std::tuple< std::tuple<bool /* useBboxes */, bool /* useContourData */>,
                    int  /* numNeighbors */, std::string /*file path*/> MSERPerfParams;
typedef perf::TestBaseWithParam<MSERPerfParams> MSERPerfTest;

PERF_TEST_P(MSERPerfTest, run,
    ::testing::Combine(::testing::Values(std::tuple<bool, bool> { true, false},
                                         std::tuple<bool, bool> {false, false},
                                         std::tuple<bool, bool> { true,  true}
                                        ), // useBboxes, useContourData
                       ::testing::Values(4, 8), // numNeighbors
                       ::testing::Values("cv/shared/baboon.png", "cv/mser/puzzle.png")
                      )
           )
{
    auto p = GetParam();
    bool useBboxes      = std::get<0>(std::get<0>(p));
    bool useContourData = std::get<1>(std::get<0>(p));
    int  numNeighbors   =             std::get<1>(p); // 4 or 8
    std::string imgPath =             std::get<2>(p);

    cv::Mat src = imread(cvtest::findDataFile(imgPath), cv::IMREAD_GRAYSCALE);

    uint32_t delta = 2;
    uint32_t minArea = 256;
    uint32_t maxArea = (int)src.total()/4;
    float        maxVariation = 0.15f;
    float        minDiversity = 0.2f;

    cv::Ptr<cv::fastcv::FCVMSER> mser;
    mser = cv::fastcv::FCVMSER::create(src.size(), numNeighbors, delta, minArea, maxArea,
                                       maxVariation, minDiversity);

    while(next())
    {
        std::vector<std::vector<Point>> contours;
        std::vector<cv::Rect> bboxes;
        std::vector<cv::fastcv::FCVMSER::ContourData> contourData;

        startTimer();
        if (useBboxes)
        {
            if (useContourData)
            {
                mser->detect(src, contours, bboxes, contourData);
            }
            else
            {
                mser->detect(src, contours, bboxes);
            }
        }
        else
        {
            mser->detect(src, contours);
        }
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
