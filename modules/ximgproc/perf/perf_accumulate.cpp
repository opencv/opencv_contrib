// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

namespace opencv_test {

typedef Size_MatType Accumulate;

#define MAT_TYPES_ACCUMLATE CV_8UC1, CV_16UC1, CV_32FC1
#define MAT_TYPES_ACCUMLATE_C MAT_TYPES_ACCUMLATE, CV_8UC3, CV_16UC3, CV_32FC3
#define MAT_TYPES_ACCUMLATE_D MAT_TYPES_ACCUMLATE, CV_64FC1
#define MAT_TYPES_ACCUMLATE_D_C MAT_TYPES_ACCUMLATE_C, CV_64FC1, CV_64FC1

#define PERF_ACCUMULATE_INIT(_FLTC)                    \
    const Size srcSize = get<0>(GetParam());           \
    const int srcType = get<1>(GetParam());            \
    const int dstType = _FLTC(CV_MAT_CN(srcType));     \
    Mat src1(srcSize, srcType), dst(srcSize, dstType); \
    declare.in(src1, dst, WARMUP_RNG).out(dst);

#define PERF_ACCUMULATE_MASK_INIT(_FLTC) \
    PERF_ACCUMULATE_INIT(_FLTC)          \
    Mat mask(srcSize, CV_8UC1);          \
    declare.in(mask, WARMUP_RNG);

#define PERF_TEST_P_ACCUMULATE(_NAME, _TYPES, _INIT, _FUN)           \
    PERF_TEST_P(Accumulate, _NAME,                                   \
        testing::Combine(                                            \
            testing::Values(sz1080p, sz720p, szVGA, szQVGA, szODD),  \
            testing::Values(_TYPES)                                  \
        )                                                            \
    )                                                                \
    {                                                                \
        _INIT                                                        \
        TEST_CYCLE() _FUN;                                           \
        SANITY_CHECK_NOTHING();                                      \
    }

/////////////////////////////////// Accumulate ///////////////////////////////////

PERF_TEST_P_ACCUMULATE(Accumulate, MAT_TYPES_ACCUMLATE,
        PERF_ACCUMULATE_INIT(CV_32FC), cv::ximgproc::accumulate(src1, dst))

PERF_TEST_P_ACCUMULATE(AccumulateMask, MAT_TYPES_ACCUMLATE_C,
    PERF_ACCUMULATE_MASK_INIT(CV_32FC), cv::ximgproc::accumulate(src1, dst, mask))

PERF_TEST_P_ACCUMULATE(AccumulateMask32FC4, CV_32FC4,
    PERF_ACCUMULATE_MASK_INIT(CV_32FC), cv::ximgproc::accumulate(src1, dst, mask))

PERF_TEST_P_ACCUMULATE(AccumulateMask8UC4To32FC4, CV_8UC4,
    PERF_ACCUMULATE_MASK_INIT(CV_32FC), cv::ximgproc::accumulate(src1, dst, mask))

PERF_TEST_P_ACCUMULATE(AccumulateDouble, MAT_TYPES_ACCUMLATE_D,
    PERF_ACCUMULATE_INIT(CV_64FC), cv::ximgproc::accumulate(src1, dst))

PERF_TEST_P_ACCUMULATE(AccumulateDoubleMask, MAT_TYPES_ACCUMLATE_D_C,
    PERF_ACCUMULATE_MASK_INIT(CV_64FC), cv::ximgproc::accumulate(src1, dst, mask))

///////////////////////////// AccumulateSquare ///////////////////////////////////

PERF_TEST_P_ACCUMULATE(Square, MAT_TYPES_ACCUMLATE,
    PERF_ACCUMULATE_INIT(CV_32FC), cv::ximgproc::accumulateSquare(src1, dst))

PERF_TEST_P_ACCUMULATE(SquareMask, MAT_TYPES_ACCUMLATE_C,
    PERF_ACCUMULATE_MASK_INIT(CV_32FC), cv::ximgproc::accumulateSquare(src1, dst, mask))

PERF_TEST_P_ACCUMULATE(SquareDouble, MAT_TYPES_ACCUMLATE_D,
    PERF_ACCUMULATE_INIT(CV_64FC), cv::ximgproc::accumulateSquare(src1, dst))

PERF_TEST_P_ACCUMULATE(SquareDoubleMask, MAT_TYPES_ACCUMLATE_D_C,
    PERF_ACCUMULATE_MASK_INIT(CV_64FC), cv::ximgproc::accumulateSquare(src1, dst, mask))

///////////////////////////// AccumulateProduct ///////////////////////////////////

#define PERF_ACCUMULATE_INIT_2(_FLTC) \
    PERF_ACCUMULATE_INIT(_FLTC)       \
    Mat src2(srcSize, srcType);       \
    declare.in(src2);

#define PERF_ACCUMULATE_MASK_INIT_2(_FLTC) \
    PERF_ACCUMULATE_MASK_INIT(_FLTC)       \
    Mat src2(srcSize, srcType);            \
    declare.in(src2);

PERF_TEST_P_ACCUMULATE(Product, MAT_TYPES_ACCUMLATE,
    PERF_ACCUMULATE_INIT_2(CV_32FC), cv::ximgproc::accumulateProduct(src1, src2, dst))

PERF_TEST_P_ACCUMULATE(ProductMask, MAT_TYPES_ACCUMLATE_C,
    PERF_ACCUMULATE_MASK_INIT_2(CV_32FC), cv::ximgproc::accumulateProduct(src1, src2, dst, mask))

PERF_TEST_P_ACCUMULATE(ProductDouble, MAT_TYPES_ACCUMLATE_D,
    PERF_ACCUMULATE_INIT_2(CV_64FC), cv::ximgproc::accumulateProduct(src1, src2, dst))

PERF_TEST_P_ACCUMULATE(ProductDoubleMask, MAT_TYPES_ACCUMLATE_D_C,
    PERF_ACCUMULATE_MASK_INIT_2(CV_64FC), cv::ximgproc::accumulateProduct(src1, src2, dst, mask))

///////////////////////////// AccumulateWeighted ///////////////////////////////////

PERF_TEST_P_ACCUMULATE(Weighted, MAT_TYPES_ACCUMLATE,
    PERF_ACCUMULATE_INIT(CV_32FC), cv::ximgproc::accumulateWeighted(src1, dst, 0.123))

PERF_TEST_P_ACCUMULATE(WeightedMask, MAT_TYPES_ACCUMLATE_C,
    PERF_ACCUMULATE_MASK_INIT(CV_32FC), cv::ximgproc::accumulateWeighted(src1, dst, 0.123, mask))

PERF_TEST_P_ACCUMULATE(WeightedDouble, MAT_TYPES_ACCUMLATE_D,
    PERF_ACCUMULATE_INIT(CV_64FC), cv::ximgproc::accumulateWeighted(src1, dst, 0.123456))

PERF_TEST_P_ACCUMULATE(WeightedDoubleMask, MAT_TYPES_ACCUMLATE_D_C,
    PERF_ACCUMULATE_MASK_INIT(CV_64FC), cv::ximgproc::accumulateWeighted(src1, dst, 0.123456, mask))

} // namespace

//////////////////////////////// OpenCL perf //////////////////////////////////

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

typedef Size_MatType AccumulateFixture;

OCL_PERF_TEST_P(AccumulateFixture, Accumulate,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int srcType = get<1>(params), cn = CV_MAT_CN(srcType), dstType = CV_32FC(cn);

    checkDeviceMaxMemoryAllocSize(srcSize, dstType);

    UMat src(srcSize, srcType), dst(srcSize, dstType);
    declare.in(src, dst, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::ximgproc::accumulate(src, dst);

    SANITY_CHECK_NOTHING();
}

typedef Size_MatType AccumulateSquareFixture;

OCL_PERF_TEST_P(AccumulateSquareFixture, AccumulateSquare,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int srcType = get<1>(params), cn = CV_MAT_CN(srcType), dstType = CV_32FC(cn);

    checkDeviceMaxMemoryAllocSize(srcSize, dstType);

    UMat src(srcSize, srcType), dst(srcSize, dstType);
    declare.in(src, dst, WARMUP_RNG);

    OCL_TEST_CYCLE() cv::ximgproc::accumulateSquare(src, dst);

    SANITY_CHECK_NOTHING();
}

typedef Size_MatType AccumulateProductFixture;

OCL_PERF_TEST_P(AccumulateProductFixture, AccumulateProduct,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int srcType = get<1>(params), cn = CV_MAT_CN(srcType), dstType = CV_32FC(cn);

    checkDeviceMaxMemoryAllocSize(srcSize, dstType);

    UMat src1(srcSize, srcType), src2(srcSize, srcType), dst(srcSize, dstType);
    declare.in(src1, src2, dst, WARMUP_RNG);

    OCL_TEST_CYCLE() cv::ximgproc::accumulateProduct(src1, src2, dst);

    SANITY_CHECK_NOTHING();
}

typedef Size_MatType AccumulateWeightedFixture;

OCL_PERF_TEST_P(AccumulateWeightedFixture, AccumulateWeighted,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int srcType = get<1>(params), cn = CV_MAT_CN(srcType), dstType = CV_32FC(cn);

    checkDeviceMaxMemoryAllocSize(srcSize, dstType);

    UMat src(srcSize, srcType), dst(srcSize, dstType);
    declare.in(src, dst, WARMUP_RNG);

    OCL_TEST_CYCLE() cv::ximgproc::accumulateWeighted(src, dst, 2.0);

    SANITY_CHECK_NOTHING();
}

} } // namespace opencv_test::ocl

#endif // HAVE_OPENCL
