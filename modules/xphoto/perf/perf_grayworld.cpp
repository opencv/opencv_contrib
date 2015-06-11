#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;

typedef std::tr1::tuple<Size, float> Size_WBThresh_t;
typedef perf::TestBaseWithParam<Size_WBThresh_t> Size_WBThresh;

PERF_TEST_P( Size_WBThresh, autowbGrayworld,
    testing::Combine(
        SZ_ALL_HD,
        testing::Values( 0.1, 0.5, 1.0 )
    )
)
{
    Size size = std::tr1::get<0>(GetParam());
    float wb_thresh = std::tr1::get<1>(GetParam());

    Mat src(size, CV_8UC3);
    Mat dst(size, CV_8UC3);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() xphoto::autowbGrayworld(src, dst, wb_thresh);

    SANITY_CHECK(dst);
}

