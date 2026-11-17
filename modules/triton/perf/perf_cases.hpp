// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright 2026, The FlagOS Contributors.
#ifndef OPENCV_TRITON_PERF_CASES_HPP
#define OPENCV_TRITON_PERF_CASES_HPP
#include "perf_precomp.hpp"
namespace opencv_test
{
namespace
{

typedef ::perf::TestBaseWithParam<testing::tuple<cv::Size, cv::Size, int, int, int>>
    Sz_Dsize_Cn_Interp;
typedef ::perf::TestBaseWithParam<testing::tuple<cv::Size, int, int, int>> Sz_Interp_Dtype;
typedef ::perf::TestBaseWithParam<testing::tuple<cv::Size, int>> Sz_Code;

// Candidate and optional native comparisons use the same geometry/type/mode grid.
inline auto resizePerfCases()
{
    return testing::Combine(
        testing::Values(cv::Size(1280, 720), cv::Size(1920, 1080), cv::Size(3840, 2160)),
        testing::Values(cv::Size(640, 360), cv::Size(1280, 720), cv::Size(2560, 1440)),
        testing::Values(1, 3, 4),
        testing::Values((int)cv::INTER_NEAREST, (int)cv::INTER_LINEAR, (int)cv::INTER_CUBIC,
                        (int)cv::INTER_AREA),
        testing::Values(CV_8U, CV_32F));
}

inline auto warpPerfCases()
{
    return testing::Combine(
        testing::Values(cv::Size(1280, 720), cv::Size(1920, 1080), cv::Size(3840, 2160)),
        testing::Values(0, 1, 2), testing::Values(CV_8U, CV_16U, CV_32F),
        testing::Values((int)cv::INTER_NEAREST, (int)cv::INTER_LINEAR, (int)cv::INTER_CUBIC));
}

inline auto colorPerfSizes()
{
    return testing::Values(cv::Size(640, 480), cv::Size(1920, 1080), cv::Size(3840, 2160));
}

// Identity, half-pixel translation and rotation/scale; indices remain part of case IDs.
inline cv::Mat warpPerfMatrix(int geometry)
{
    static const double matrices[][6] = {{1.0, 0.0, 0.0, 0.0, 1.0, 0.0},
                                         {1.0, 0.0, 0.5, 0.0, 1.0, 0.5},
                                         {0.9, -0.43589, 40.0, 0.43589, 0.9, -20.0}};
    CV_Assert(geometry >= 0 && geometry < 3);
    cv::Mat matrix(2, 3, CV_64F);
    for (int i = 0; i < 6; ++i)
        matrix.at<double>(i / 3, i % 3) = matrices[geometry][i];
    return matrix;
}

}
} // namespace
#endif
