// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef perf::TestBaseWithParam<std::string> daisy;

#define DAISY_IMAGES \
    "cv/detectors_descriptors_evaluation/images_datasets/leuven/img1.png",\
    "stitching/a3.png"

PERF_TEST_P(daisy, extract, testing::Values(DAISY_IMAGES))
{
    string filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame.empty()) << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame).time(90);

    Ptr<DAISY> descriptor = DAISY::create();

    vector<KeyPoint> points;
    Mat_<float> descriptors;
    // compute all daisies in image
    TEST_CYCLE() descriptor->compute(frame, descriptors);

    SANITY_CHECK_NOTHING();
}

}} // namespace
