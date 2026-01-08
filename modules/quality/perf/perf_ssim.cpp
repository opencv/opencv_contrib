// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test
{
namespace
{

typedef tuple<Size, MatType> Size_MatType_t;
typedef perf::TestBaseWithParam<Size_MatType_t> Size_MatType;

// SSIM performance test with different image sizes
PERF_TEST_P(Size_MatType, SSIM,
            testing::Combine(
                testing::Values(szVGA, sz720p, sz1080p),
                testing::Values(CV_8UC1, CV_8UC3)
            )
)
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());

    Mat ref(size, type);
    Mat cmp(size, type);

    declare.in(ref, WARMUP_RNG).in(cmp, WARMUP_RNG);

    TEST_CYCLE()
    {
        cv::Scalar result = QualitySSIM::compute(ref, cmp, noArray());
        (void)result;
    }

    SANITY_CHECK_NOTHING();
}

// SSIM with quality map output
PERF_TEST_P(Size_MatType, SSIM_with_map,
            testing::Combine(
                testing::Values(szVGA, sz720p),
                testing::Values(CV_8UC1, CV_8UC3)
            )
)
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());

    Mat ref(size, type);
    Mat cmp(size, type);

    declare.in(ref, WARMUP_RNG).in(cmp, WARMUP_RNG);

    Mat qualityMap;

    TEST_CYCLE()
    {
        cv::Scalar result = QualitySSIM::compute(ref, cmp, qualityMap);
        (void)result;
    }

    SANITY_CHECK_NOTHING();
}

// SSIM with pre-computed reference (typical use case)
PERF_TEST_P(Size_MatType, SSIM_precomputed_ref,
            testing::Combine(
                testing::Values(szVGA, sz720p, sz1080p),
                testing::Values(CV_8UC1, CV_8UC3)
            )
)
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());

    Mat ref(size, type);
    Mat cmp(size, type);

    randu(ref, 0, 255);
    randu(cmp, 0, 255);

    // Pre-compute reference image data (one-time cost)
    Ptr<QualitySSIM> ssim = QualitySSIM::create(ref);

    declare.in(cmp, WARMUP_RNG);

    TEST_CYCLE()
    {
        cv::Scalar result = ssim->compute(cmp);
        (void)result;
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
} // namespace opencv_test
