#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

using namespace std::tr1;
using namespace cv;
using namespace perf;

namespace cvtest {
namespace ocl {

///////////////////////// Retina ////////////////////////

typedef tuple<bool, int, double, double> RetinaParams;
typedef TestBaseWithParam<RetinaParams> RetinaFixture;

OCL_PERF_TEST_P(RetinaFixture, Retina,
                ::testing::Combine(testing::Bool(), testing::Values((int)cv::bioinspired::RETINA_COLOR_BAYER),
                                   testing::Values(1.0, 0.5), testing::Values(10.0, 5.0)))
{
    RetinaParams params = GetParam();
    bool colorMode = get<0>(params), useLogSampling = false;
    int colorSamplingMethod = get<1>(params);
    float reductionFactor = static_cast<float>(get<2>(params));
    float samplingStrength = static_cast<float>(get<3>(params));

    Mat input = imread(getDataPath("cv/shared/lena.png"), colorMode);
    ASSERT_FALSE(input.empty());

    UMat ocl_parvo, ocl_magno;

    {
        Ptr<cv::bioinspired::Retina> retina = cv::bioinspired::Retina::create(
            input.size(), colorMode, colorSamplingMethod, useLogSampling,
            reductionFactor, samplingStrength);

        OCL_TEST_CYCLE()
        {
            retina->run(input);
            retina->getParvo(ocl_parvo);
            retina->getMagno(ocl_magno);
        }
    }

    SANITY_CHECK_NOTHING();
}

} } // namespace cvtest::ocl
