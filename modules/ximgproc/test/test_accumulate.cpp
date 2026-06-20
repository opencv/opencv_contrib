// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

namespace opencv_test { namespace {

//////////////////////////////// CPU accuracy /////////////////////////////////
// Multi-channel masked accumulate against a scalar reference implementation.

typedef testing::TestWithParam<tuple<Size, int, int> > Ximgproc_Accumulate;

TEST_P(Ximgproc_Accumulate, accuracy)
{
    const Size size = get<0>(GetParam());
    const int pattern = get<1>(GetParam());
    const int srcType = get<2>(GetParam());

    RNG& rng = theRNG();

    Mat src(size, srcType);
    Mat dst(size, CV_32FC4);
    Mat mask(size, CV_8UC1);

    if (srcType == CV_8UC4)
        rng.fill(src, RNG::UNIFORM, Scalar::all(0), Scalar::all(256));
    else
        rng.fill(src, RNG::UNIFORM, Scalar::all(-10.0), Scalar::all(10.0));

    rng.fill(dst, RNG::UNIFORM, Scalar::all(-1000.0), Scalar::all(1000.0));

    for (int y = 0; y < mask.rows; ++y)
    {
        uchar* row = mask.ptr<uchar>(y);

        for (int x = 0; x < mask.cols; ++x)
        {
            switch (pattern)
            {
            case 0:
                row[x] = 0;
                break;
            case 1:
                row[x] = 255;
                break;
            case 2:
                row[x] = ((x + y) % 2) ? 255 : 0;
                break;
            case 3:
                row[x] = ((x * 13 + y * 7) % 5) ? 255 : 0;
                break;
            default:
                row[x] = ((x * 17 + y * 11) % 3) ? 255 : 0;
                break;
            }
        }
    }

    Mat dstRef = dst.clone();

    if (srcType == CV_32FC4)
    {
        for (int y = 0; y < src.rows; ++y)
        {
            const Vec4f* srcRow = src.ptr<Vec4f>(y);
            Vec4f* dstRefRow = dstRef.ptr<Vec4f>(y);
            const uchar* maskRow = mask.ptr<uchar>(y);

            for (int x = 0; x < src.cols; ++x)
            {
                if (maskRow[x])
                {
                    for (int c = 0; c < 4; ++c)
                        dstRefRow[x][c] += srcRow[x][c];
                }
            }
        }
    }
    else
    {
        CV_Assert(srcType == CV_8UC4);

        for (int y = 0; y < src.rows; ++y)
        {
            const Vec4b* srcRow = src.ptr<Vec4b>(y);
            Vec4f* dstRefRow = dstRef.ptr<Vec4f>(y);
            const uchar* maskRow = mask.ptr<uchar>(y);

            for (int x = 0; x < src.cols; ++x)
            {
                if (maskRow[x])
                {
                    for (int c = 0; c < 4; ++c)
                        dstRefRow[x][c] += static_cast<float>(srcRow[x][c]);
                }
            }
        }
    }

    cv::ximgproc::accumulate(src, dst, mask);

    const double err = cv::norm(dst, dstRef, NORM_INF);

    EXPECT_EQ(0.0, err)
        << "size=" << size
        << ", pattern=" << pattern
        << ", srcType=" << srcType;
}

INSTANTIATE_TEST_CASE_P(, Ximgproc_Accumulate,
    testing::Combine(
        testing::Values(Size(1, 1),
                        Size(3, 5),
                        Size(17, 7),
                        Size(37, 19),
                        Size(128, 16),
                        Size(641, 37)),
        testing::Values(0, 1, 2, 3, 4),
        testing::Values(CV_32FC4, CV_8UC4)));

}} // namespace opencv_test::<anonymous>

//////////////////////////////// OpenCL accuracy //////////////////////////////

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

PARAM_TEST_CASE(AccumulateBase, std::pair<MatDepth, MatDepth>, Channels, bool)
{
    int sdepth, ddepth, channels;
    bool useRoi;
    double alpha;

    TEST_DECLARE_INPUT_PARAMETER(src);
    TEST_DECLARE_INPUT_PARAMETER(mask);
    TEST_DECLARE_INPUT_PARAMETER(src2);
    TEST_DECLARE_OUTPUT_PARAMETER(dst);

    virtual void SetUp()
    {
        const std::pair<MatDepth, MatDepth> depths = GET_PARAM(0);
        sdepth = depths.first, ddepth = depths.second;
        channels = GET_PARAM(1);
        useRoi = GET_PARAM(2);
    }

    void random_roi()
    {
        const int stype = CV_MAKE_TYPE(sdepth, channels),
                dtype = CV_MAKE_TYPE(ddepth, channels);

        Size roiSize = randomSize(1, 10);
        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, stype, -MAX_VALUE, MAX_VALUE);

        Border maskBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(mask, mask_roi, roiSize, maskBorder, CV_8UC1, -MAX_VALUE, MAX_VALUE);
        cvtest::threshold(mask, mask, 80, 255, THRESH_BINARY);

        Border src2Border = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src2, src2_roi, roiSize, src2Border, stype, -MAX_VALUE, MAX_VALUE);

        Border dstBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, dtype, -MAX_VALUE, MAX_VALUE);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_INPUT_PARAMETER(mask);
        UMAT_UPLOAD_INPUT_PARAMETER(src2);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);

        alpha = randomDouble(-5, 5);
    }
};

/////////////////////////////////// Accumulate ///////////////////////////////////

typedef AccumulateBase Accumulate;

OCL_TEST_P(Accumulate, Mat)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        random_roi();

        OCL_OFF(cv::ximgproc::accumulate(src_roi, dst_roi));
        OCL_ON(cv::ximgproc::accumulate(usrc_roi, udst_roi));

        OCL_EXPECT_MATS_NEAR(dst, 1e-6);
    }
}

OCL_TEST_P(Accumulate, Mask)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        random_roi();

        OCL_OFF(cv::ximgproc::accumulate(src_roi, dst_roi, mask_roi));
        OCL_ON(cv::ximgproc::accumulate(usrc_roi, udst_roi, umask_roi));

        OCL_EXPECT_MATS_NEAR(dst, 1e-6);
    }
}

/////////////////////////////////// AccumulateSquare ///////////////////////////////////

typedef AccumulateBase AccumulateSquare;

OCL_TEST_P(AccumulateSquare, Mat)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        random_roi();

        OCL_OFF(cv::ximgproc::accumulateSquare(src_roi, dst_roi));
        OCL_ON(cv::ximgproc::accumulateSquare(usrc_roi, udst_roi));

        OCL_EXPECT_MATS_NEAR(dst, 1e-2);
    }
}

OCL_TEST_P(AccumulateSquare, Mask)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        random_roi();

        OCL_OFF(cv::ximgproc::accumulateSquare(src_roi, dst_roi, mask_roi));
        OCL_ON(cv::ximgproc::accumulateSquare(usrc_roi, udst_roi, umask_roi));

        OCL_EXPECT_MATS_NEAR(dst, 1e-2);
    }
}

/////////////////////////////////// AccumulateProduct ///////////////////////////////////

typedef AccumulateBase AccumulateProduct;

OCL_TEST_P(AccumulateProduct, Mat)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        random_roi();

        OCL_OFF(cv::ximgproc::accumulateProduct(src_roi, src2_roi, dst_roi));
        OCL_ON(cv::ximgproc::accumulateProduct(usrc_roi, usrc2_roi, udst_roi));

        OCL_EXPECT_MATS_NEAR(dst, 1e-2);
    }
}

OCL_TEST_P(AccumulateProduct, Mask)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        random_roi();

        OCL_OFF(cv::ximgproc::accumulateProduct(src_roi, src2_roi, dst_roi, mask_roi));
        OCL_ON(cv::ximgproc::accumulateProduct(usrc_roi, usrc2_roi, udst_roi, umask_roi));

        OCL_EXPECT_MATS_NEAR(dst, 1e-2);
    }
}

/////////////////////////////////// AccumulateWeighted ///////////////////////////////////

typedef AccumulateBase AccumulateWeighted;

OCL_TEST_P(AccumulateWeighted, Mat)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        random_roi();

        OCL_OFF(cv::ximgproc::accumulateWeighted(src_roi, dst_roi, alpha));
        OCL_ON(cv::ximgproc::accumulateWeighted(usrc_roi, udst_roi, alpha));

        OCL_EXPECT_MATS_NEAR(dst, 1e-2);
    }
}

OCL_TEST_P(AccumulateWeighted, Mask)
{
    for (int i = 0; i < test_loop_times; ++i)
    {
        random_roi();

        OCL_OFF(cv::ximgproc::accumulateWeighted(src_roi, dst_roi, alpha));
        OCL_ON(cv::ximgproc::accumulateWeighted(usrc_roi, udst_roi, alpha));

        OCL_EXPECT_MATS_NEAR(dst, 1e-2);
    }
}

/////////////////////////////////// Instantiation ///////////////////////////////////

#define OCL_DEPTH_ALL_COMBINATIONS \
    testing::Values(std::make_pair<MatDepth, MatDepth>(CV_8U, CV_32F), \
    std::make_pair<MatDepth, MatDepth>(CV_16U, CV_32F), \
    std::make_pair<MatDepth, MatDepth>(CV_32F, CV_32F), \
    std::make_pair<MatDepth, MatDepth>(CV_8U, CV_64F), \
    std::make_pair<MatDepth, MatDepth>(CV_16U, CV_64F), \
    std::make_pair<MatDepth, MatDepth>(CV_32F, CV_64F), \
    std::make_pair<MatDepth, MatDepth>(CV_64F, CV_64F))

OCL_INSTANTIATE_TEST_CASE_P(Ximgproc, Accumulate, Combine(OCL_DEPTH_ALL_COMBINATIONS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Ximgproc, AccumulateSquare, Combine(OCL_DEPTH_ALL_COMBINATIONS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Ximgproc, AccumulateProduct, Combine(OCL_DEPTH_ALL_COMBINATIONS, OCL_ALL_CHANNELS, Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Ximgproc, AccumulateWeighted, Combine(OCL_DEPTH_ALL_COMBINATIONS, OCL_ALL_CHANNELS, Bool()));

} } // namespace opencv_test::ocl

#endif // HAVE_OPENCL
