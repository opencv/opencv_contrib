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

    unsigned int delta = 2;
    unsigned int minArea = 256;
    unsigned int maxArea = (int)src.total()/4;
    float        maxVariation = 0.15f;
    float        minDiversity = 0.2f;

    while(next())
    {
        std::vector<std::vector<Point>> contours;
        std::vector<cv::Rect> bboxes;
        std::vector<cv::fastcv::ContourData> contourData;

        startTimer();
        if (useBboxes)
        {
            if (useContourData)
            {
                cv::fastcv::MSER(src, contours, bboxes, contourData, numNeighbors,
                                 delta, minArea, maxArea, maxVariation, minDiversity);
            }
            else
            {
                cv::fastcv::MSER(src, contours, bboxes, numNeighbors,
                                 delta, minArea, maxArea, maxVariation, minDiversity);
            }
        }
        else
        {
            cv::fastcv::MSER(src, contours, numNeighbors,
                             delta, minArea, maxArea, maxVariation, minDiversity);
        }
        stopTimer();
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
