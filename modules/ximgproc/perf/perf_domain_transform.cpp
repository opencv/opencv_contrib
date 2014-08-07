#include "perf_precomp.hpp"

using namespace std;
using namespace std::tr1;
using namespace cv;
using namespace perf;
using namespace testing;

namespace cvtest
{

CV_ENUM(GuideMatType, CV_8UC1, CV_8UC3, CV_32FC1, CV_32FC3) //reduced set
CV_ENUM(SourceMatType, CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4) //full supported set
CV_ENUM(DTFMode, DTF_NC, DTF_IC, DTF_RF)
typedef tuple<GuideMatType, SourceMatType, Size, double, double, DTFMode> DTTestParams;

typedef TestBaseWithParam<DTTestParams> DomainTransformTest;

PERF_TEST_P( DomainTransformTest, perf,
             Combine(
                      GuideMatType::all(),
                      SourceMatType::all(),
                      Values(szVGA, sz720p),
                      Values(10.0, 80.0),
                      Values(30.0, 50.0),
                      DTFMode::all()
                    )
           )
{
    int guideType       = get<0>(GetParam());
    int srcType         = get<1>(GetParam());
    Size size           = get<2>(GetParam());
    double sigmaSpatial = get<3>(GetParam());
    double sigmaColor   = get<4>(GetParam());
    int dtfType         = get<5>(GetParam());

    Mat guide(size, guideType);
    Mat src(size, srcType);
    Mat dst(size, srcType);

    declare.in(guide, src, WARMUP_RNG).out(dst).tbb_threads(cv::getNumberOfCPUs());

    cv::setNumThreads(cv::getNumberOfCPUs());
    TEST_CYCLE_N(5)
    {
        dtFilter(guide, src, dst, sigmaSpatial, sigmaColor, dtfType);
    }
    
    SANITY_CHECK(dst);
}

}