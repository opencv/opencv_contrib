// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;

namespace opencv_test { namespace {

// sprt(-1)/ransac(0)/preemptive(500) mode, inlier size, outlier size
typedef perf::TestBaseWithParam< tuple<int,int,int> > ptcloud_sac;
#define PREC testing::Values(-1,0,500)
#define MODL testing::Values(64,1024)
#define RAND testing::Values(64,1024)

//
// if the inlier ratio is high, plain ransac is still faster than the preemptive version
// (i guess, there is a high bailout rate then)
//
PERF_TEST_P(ptcloud_sac, segment, testing::Combine(PREC, MODL, RAND))
{
    int pc = get<0>(GetParam());
    int s1 = get<1>(GetParam());
    int s2 = get<2>(GetParam());

    Mat cloud;
    ptcloud::generatePlane(cloud, std::vector<double>{0.98, 0.13, 0.13, -10.829}, s1);
    ptcloud::generateRandom(cloud, std::vector<double>{0,0,0,100}, s2);

    Ptr<ptcloud::SACModelFitting> fit = ptcloud::SACModelFitting::create(cloud);
    fit->set_threshold(0.0015);
    if (pc == -1) { // test sprt
        fit->set_use_sprt(true);
        fit->set_preemptive_count(0);
    } else {
        fit->set_preemptive_count(pc);
    }

    std::vector<ptcloud::SACModel> mdl;

    size_t count = 0; // each cycle should segment exactly 1 model
    TEST_CYCLE() { fit->segment(mdl); count++; }

    ASSERT_TRUE(mdl.size() == count);

    SANITY_CHECK_NOTHING();
}

}}