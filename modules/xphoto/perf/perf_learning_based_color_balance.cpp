// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple<Size, MatType> learningBasedWBParams;
typedef perf::TestBaseWithParam<learningBasedWBParams> learningBasedWBPerfTest;

PERF_TEST_P(learningBasedWBPerfTest, perf, Combine(SZ_ALL_HD, Values(CV_8UC3, CV_16UC3)))
{
    Size size = get<0>(GetParam());
    MatType t = get<1>(GetParam());
    Mat src(size, t);
    Mat dst(size, t);

    int range_max_val = 255, hist_bin_num = 64;
    if (t == CV_16UC3)
    {
        range_max_val = 65535;
        hist_bin_num = 256;
    }

    Mat src_dscl(Size(size.width / 16, size.height / 16), t);
    RNG rng(1234);
    rng.fill(src_dscl, RNG::UNIFORM, 0, range_max_val);
    resize(src_dscl, src, src.size(), 0, 0, INTER_LINEAR_EXACT);
    Ptr<xphoto::LearningBasedWB> wb = xphoto::createLearningBasedWB();
    wb->setRangeMaxVal(range_max_val);
    wb->setSaturationThreshold(0.98f);
    wb->setHistBinNum(hist_bin_num);

    TEST_CYCLE() wb->balanceWhite(src, dst);

    SANITY_CHECK_NOTHING();
}


}} // namespace
