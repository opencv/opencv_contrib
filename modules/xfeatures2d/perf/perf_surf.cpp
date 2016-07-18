#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef perf::TestBaseWithParam<std::string> surf;

#define SURF_IMAGES \
    "cv/detectors_descriptors_evaluation/images_datasets/leuven/img1.png",\
    "stitching/a3.png"

PERF_TEST_P(surf, detect, testing::Values(SURF_IMAGES))
{
    string filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame.empty()) << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame).time(90);
    Ptr<SURF> detector = SURF::create();
    vector<KeyPoint> points;

    TEST_CYCLE() detector->detect(frame, points, mask);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(surf, extract, testing::Values(SURF_IMAGES))
{
    string filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame.empty()) << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame).time(90);

    Ptr<SURF> detector = SURF::create();
    vector<KeyPoint> points;
    vector<float> descriptors;
    detector->detect(frame, points, mask);

    TEST_CYCLE() detector->compute(frame, points, descriptors);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(surf, full, testing::Values(SURF_IMAGES))
{
    string filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame.empty()) << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame).time(90);
    Ptr<SURF> detector = SURF::create();
    vector<KeyPoint> points;
    vector<float> descriptors;

    TEST_CYCLE() detector->detectAndCompute(frame, mask, points, descriptors, false);

    SANITY_CHECK_NOTHING();
}
