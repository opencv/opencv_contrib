using namespace std;
using namespace cv;
using namespace perf;

#include "perf_precomp.hpp"

namespace opencv_test
{

typedef std::tr1::tuple<Size, MatType, MatDepth> Size_MatType_OutMatDepth_t;
typedef perf::TestBaseWithParam<Size_MatType_OutMatDepth_t> Size_MatType_OutMatDepth;

/* 2. Declare the testsuite */
PERF_TEST_P( Size_MatType_OutMatDepth, integral1,
    testing::Combine(
            testing::Values( TYPICAL_MAT_SIZES ),
            testing::Values( CV_8UC1, CV_8UC4 ),
            testing::Values( CV_32S, CV_32F, CV_64F ) ) )
{
   
    string folder = "cv/alphamat/";
    string image_path = folder + "img/elephant.png";
    string trimap_path = folder + "trimap/elephant.png";
    string reference_path = folder + "reference/elephant.png";

    Mat image = imread(getDataPath(image_path), IMREAD_COLOR);
    Mat trimap = imread(getDataPath(trimap_path), IMREAD_COLOR);
    Mat reference = imread(getDataPath(reference_path), IMREAD_GRAYSCALE);

    Size sz = get<0>(GetParam());
    int inpaintingMethod = get<1>(GetParam());

    Mat result;
    declare.in(image, trimap).out(result).time(120);

    TEST_CYCLE() infoFlow(image, trimap, result, false, true);

    SANITY_CHECK_NOTHING();
}

} // namespace
