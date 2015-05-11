#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

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

    // use DAISY in COMP_FULL mode (every pixel, dense baseline mode)
    Ptr<DAISY> descriptor = DAISY::create(15, 3, 8, 8, DAISY::COMP_FULL, DAISY::NRM_NONE, noArray(), true, false);

    vector<KeyPoint> points;
    vector<float> descriptors;
    TEST_CYCLE() descriptor->compute(frame, points, descriptors);

    SANITY_CHECK(descriptors, 1e-4);
}
