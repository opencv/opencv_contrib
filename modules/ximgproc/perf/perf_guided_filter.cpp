#include "perf_precomp.hpp"

namespace cvtest
{

using std::tr1::tuple;
using std::tr1::get;
using namespace perf;
using namespace testing;
using namespace cv;
using namespace cv::ximgproc;

CV_ENUM(GuideTypes, CV_8UC1, CV_8UC2, CV_8UC3, CV_32FC1, CV_32FC2, CV_32FC3);
CV_ENUM(SrcTypes, CV_8UC1, CV_8UC3, CV_32FC1, CV_32FC3);
typedef tuple<GuideTypes, SrcTypes, Size> GFParams;
    
typedef TestBaseWithParam<GFParams> GuidedFilterPerfTest;
    
PERF_TEST_P( GuidedFilterPerfTest, perf, Combine(GuideTypes::all(), SrcTypes::all(), Values(sz1080p, sz2K)) )
{
    RNG rng(0);

    GFParams params = GetParam();
    int guideType   = get<0>(params);
    int srcType     = get<1>(params);
    Size sz         = get<2>(params);

    Mat guide(sz, guideType);
    Mat src(sz, srcType);
    Mat dst(sz, srcType);

    declare.in(guide, src, WARMUP_RNG).out(dst).tbb_threads(cv::getNumberOfCPUs());

    cv::setNumThreads(cv::getNumberOfCPUs());
    TEST_CYCLE_N(3)
    {
        int radius = rng.uniform(5, 30);
        double eps = rng.uniform(0.1, 1e5);
        guidedFilter(guide, src, dst, radius, eps);
    }

    SANITY_CHECK(dst);
}

}