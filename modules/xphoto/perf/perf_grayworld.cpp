// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple<Size, float> Size_WBThresh_t;
typedef perf::TestBaseWithParam<Size_WBThresh_t> Size_WBThresh;

PERF_TEST_P( Size_WBThresh, autowbGrayworld,
    testing::Combine(
        SZ_ALL_HD,
        testing::Values( 0.1, 0.5, 1.0 )
    )
)
{
    Size size = get<0>(GetParam());
    float wb_thresh = get<1>(GetParam());

    Mat src(size, CV_8UC3);
    Mat dst(size, CV_8UC3);

    declare.in(src, WARMUP_RNG).out(dst);
    Ptr<xphoto::GrayworldWB> wb = xphoto::createGrayworldWB();
    wb->setSaturationThreshold(wb_thresh);

    TEST_CYCLE() wb->balanceWhite(src, dst);

    SANITY_CHECK(dst);
}


}} // namespace
