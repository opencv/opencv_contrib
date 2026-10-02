// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test {
namespace {

typedef tuple<MorphTypes, MorphShapes> MorphTypes_MorphShapes_t;
typedef TestBaseWithParam<MorphTypes_MorphShapes_t> SparseTableMorphologyPerfTest;

PERF_TEST_P(SparseTableMorphologyPerfTest, perf,
    testing::Combine(
        testing::Values(
            MORPH_ERODE, MORPH_DILATE, MORPH_OPEN, MORPH_CLOSE,
            MORPH_GRADIENT, MORPH_TOPHAT, MORPH_BLACKHAT),
        testing::Values(MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE)
    ) )
{
    MorphTypes_MorphShapes_t params = GetParam();
    int seSize = 51;
    Size sz = sz1080p;
    MorphTypes op = std::get<0>(params);
    MorphShapes knType = std::get<1>(params);

    Mat src(sz, CV_8UC3), dst(sz, CV_8UC3);
    Mat kernel = getStructuringElement(knType, cv::Size(2 * seSize + 1, 2 * seSize + 1));

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE_N(5)
    {
        cv::ximgproc::stMorph::morphologyEx(src, dst, op, kernel);
    }

    SANITY_CHECK_NOTHING();
}

}} // opencv_test:: ::
