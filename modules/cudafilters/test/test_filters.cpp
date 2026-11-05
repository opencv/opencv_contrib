/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

#ifdef HAVE_CUDA
#include "../src/cuda/wavelet_matrix_feature_support_checks.h"

namespace opencv_test { namespace {

namespace
{
    IMPLEMENT_PARAM_CLASS(KSize, cv::Size)
    IMPLEMENT_PARAM_CLASS(Anchor, cv::Point)
    IMPLEMENT_PARAM_CLASS(Deriv_X, int)
    IMPLEMENT_PARAM_CLASS(Deriv_Y, int)
    IMPLEMENT_PARAM_CLASS(Iterations, int)
    IMPLEMENT_PARAM_CLASS(KernelSize, int)
    IMPLEMENT_PARAM_CLASS(GaussianSigma, double)

    cv::Mat getInnerROI(cv::InputArray m_, cv::Size ksize)
    {
        cv::Mat m = getMat(m_);
        cv::Rect roi(ksize.width, ksize.height, m.cols - 2 * ksize.width, m.rows - 2 * ksize.height);
        return m(roi);
    }

    cv::Mat makeStructuredMat(cv::Size size, int type)
    {
        const int cn = CV_MAT_CN(type);
        cv::Mat noise = randomMat(size, type, 0.0, 64.0);
        cv::Mat ramp(size, CV_32FC1);
        for (int y = 0; y < size.height; ++y)
        {
            float* row = ramp.ptr<float>(y);
            const float fy = size.height > 1 ? (float)y / (float)(size.height - 1) : 0.0f;
            for (int x = 0; x < size.width; ++x)
            {
                const float fx = size.width > 1 ? (float)x / (float)(size.width - 1) : 0.0f;
                row[x] = 150.0f * fx + 35.0f * fy;
            }
        }
        std::vector<cv::Mat> ch(cn, ramp);
        cv::Mat merged, typed;
        cv::merge(ch, merged);
        merged.convertTo(typed, type);
        cv::add(noise, typed, noise);
        return noise;
    }

    // sepFilter2D has no BORDER_WRAP, so pad the source by hand for that mode.
    cv::Mat sepFilterGold(const cv::Mat& src, const cv::Mat& rowKernel, const cv::Mat& columnKernel, cv::Point anchor, int borderType)
    {
        cv::Mat dst;
        if (borderType != cv::BORDER_WRAP)
        {
            cv::sepFilter2D(src, dst, -1, rowKernel, columnKernel, anchor, 0, borderType);
            return dst;
        }

        const int ax = anchor.x < 0 ? rowKernel.cols / 2 : anchor.x;
        const int ay = anchor.y < 0 ? columnKernel.cols / 2 : anchor.y;
        cv::Mat padded;
        cv::copyMakeBorder(src, padded, ay, columnKernel.cols - 1 - ay, ax, rowKernel.cols - 1 - ax, cv::BORDER_WRAP);
        cv::sepFilter2D(padded, dst, -1, rowKernel, columnKernel, cv::Point(ax, ay), 0, cv::BORDER_CONSTANT);
        return dst(cv::Rect(ax, ay, src.cols, src.rows)).clone();
    }

    cv::Mat normalizedKernel(int ksize)
    {
        cv::Mat kernel = randomMat(cv::Size(ksize, 1), CV_32FC1, 0.05, 1.0);
        kernel /= cv::sum(kernel)[0];
        return kernel;
    }

    double sepFilterTolerance(int depth)
    {
        return depth < CV_32F ? 1.0 : 1e-3;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Blur

PARAM_TEST_CASE(Blur, cv::cuda::DeviceInfo, cv::Size, MatType, KSize, Anchor, BorderType, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int type;
    cv::Size ksize;
    cv::Point anchor;
    int borderType;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        ksize = GET_PARAM(3);
        anchor = GET_PARAM(4);
        borderType = GET_PARAM(5);
        useRoi = GET_PARAM(6);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Blur, Accuracy)
{
    cv::Mat src = randomMat(size, type);

    cv::Ptr<cv::cuda::Filter> blurFilter = cv::cuda::createBoxFilter(src.type(), -1, ksize, anchor, borderType);

    cv::cuda::GpuMat dst = createMat(size, type, useRoi);
    blurFilter->apply(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::blur(src, dst_gold, ksize, anchor, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, 1.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Filters, Blur, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4), MatType(CV_32FC1)),
    testing::Values(KSize(cv::Size(3, 3)), KSize(cv::Size(5, 5)), KSize(cv::Size(7, 7))),
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    testing::Values(BorderType(cv::BORDER_REFLECT101), BorderType(cv::BORDER_REPLICATE), BorderType(cv::BORDER_CONSTANT), BorderType(cv::BORDER_REFLECT)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Filter2D

PARAM_TEST_CASE(Filter2D, cv::cuda::DeviceInfo, cv::Size, MatType, KSize, Anchor, BorderType, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int type;
    cv::Size ksize;
    cv::Point anchor;
    int borderType;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        ksize = GET_PARAM(3);
        anchor = GET_PARAM(4);
        borderType = GET_PARAM(5);
        useRoi = GET_PARAM(6);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Filter2D, Accuracy)
{
    cv::Mat src = randomMat(size, type);
    cv::Mat kernel = randomMat(cv::Size(ksize.width, ksize.height), CV_32FC1, 0.0, 1.0);

    cv::Ptr<cv::cuda::Filter> filter2D = cv::cuda::createLinearFilter(src.type(), -1, kernel, anchor, borderType);

    cv::cuda::GpuMat dst = createMat(size, type, useRoi);
    filter2D->apply(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::filter2D(src, dst_gold, -1, kernel, anchor, 0, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, CV_MAT_DEPTH(type) == CV_32F ? 1e-1 : 1.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Filters, Filter2D, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4), MatType(CV_16UC1), MatType(CV_16UC4), MatType(CV_32FC1), MatType(CV_32FC4)),
    testing::Values(KSize(cv::Size(3, 3)), KSize(cv::Size(5, 5)), KSize(cv::Size(7, 7)), KSize(cv::Size(11, 11)), KSize(cv::Size(13, 13)), KSize(cv::Size(15, 15))),
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    testing::Values(BorderType(cv::BORDER_REFLECT101), BorderType(cv::BORDER_REPLICATE), BorderType(cv::BORDER_CONSTANT), BorderType(cv::BORDER_REFLECT)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Laplacian

PARAM_TEST_CASE(Laplacian, cv::cuda::DeviceInfo, cv::Size, MatType, KSize, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int type;
    cv::Size ksize;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        ksize = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Laplacian, Accuracy)
{
    cv::Mat src = randomMat(size, type);

    cv::Ptr<cv::cuda::Filter> laplacian = cv::cuda::createLaplacianFilter(src.type(), -1, ksize.width);

    cv::cuda::GpuMat dst = createMat(size, type, useRoi);
    laplacian->apply(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::Laplacian(src, dst_gold, -1, ksize.width);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() < CV_32F ? 0.0 : 1e-3);
}

INSTANTIATE_TEST_CASE_P(CUDA_Filters, Laplacian, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4), MatType(CV_32FC1)),
    testing::Values(KSize(cv::Size(1, 1)), KSize(cv::Size(3, 3))),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// SeparableLinearFilter

PARAM_TEST_CASE(SeparableLinearFilter, cv::cuda::DeviceInfo, cv::Size, MatDepth, Channels, KSize, Anchor, BorderType, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    int cn;
    cv::Size ksize;
    cv::Point anchor;
    int borderType;
    bool useRoi;

    int type;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        cn = GET_PARAM(3);
        ksize = GET_PARAM(4);
        anchor = GET_PARAM(5);
        borderType = GET_PARAM(6);
        useRoi = GET_PARAM(7);

        cv::cuda::setDevice(devInfo.deviceID());

        type = CV_MAKE_TYPE(depth, cn);
    }
};

CUDA_TEST_P(SeparableLinearFilter, Accuracy)
{
    cv::Mat src = randomMat(size, type);
    cv::Mat rowKernel = randomMat(Size(ksize.width, 1), CV_32FC1, 0.0, 1.0);
    cv::Mat columnKernel = randomMat(Size(ksize.height, 1), CV_32FC1, 0.0, 1.0);

    cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createSeparableLinearFilter(src.type(), -1, rowKernel, columnKernel, anchor, borderType);

    cv::cuda::GpuMat dst = createMat(size, type, useRoi);
    filter->apply(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::sepFilter2D(src, dst_gold, -1, rowKernel, columnKernel, anchor, 0, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() < CV_32F ? 1.0 : 1e-2);
}

INSTANTIATE_TEST_CASE_P(CUDA_Filters, SeparableLinearFilter, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_16S), MatDepth(CV_32F)),
    IMAGE_CHANNELS,
    testing::Values(KSize(cv::Size(3, 3)),
                    KSize(cv::Size(7, 7)),
                    KSize(cv::Size(13, 13)),
                    KSize(cv::Size(15, 15)),
                    KSize(cv::Size(17, 17)),
                    KSize(cv::Size(23, 15)),
                    KSize(cv::Size(31, 3))),
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    testing::Values(BorderType(cv::BORDER_REFLECT101),
                    BorderType(cv::BORDER_REPLICATE),
                    BorderType(cv::BORDER_CONSTANT),
                    BorderType(cv::BORDER_REFLECT)),
    WHOLE_SUBMAT));

PARAM_TEST_CASE(SeparableLinearFilterLargeKSize, cv::cuda::DeviceInfo, cv::Size, MatType, KSize, BorderType, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int type;
    cv::Size ksize;
    int borderType;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        ksize = GET_PARAM(3);
        borderType = GET_PARAM(4);
        useRoi = GET_PARAM(5);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(SeparableLinearFilterLargeKSize, Accuracy)
{
    cv::Mat src = makeStructuredMat(size, type);
    cv::Mat rowKernel = normalizedKernel(ksize.width);
    cv::Mat columnKernel = normalizedKernel(ksize.height);

    const cv::Point anchors[] = {cv::Point(-1, -1), cv::Point(0, ksize.height - 1), cv::Point(ksize.width - 1, 0)};
    for (const cv::Point& anchor : anchors)
    {
        SCOPED_TRACE(cv::format("anchor=(%d, %d)", anchor.x, anchor.y));

        cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createSeparableLinearFilter(src.type(), -1, rowKernel, columnKernel, anchor, borderType);

        cv::cuda::GpuMat dst = createMat(size, type, useRoi);
        filter->apply(loadMat(src, useRoi), dst);

        EXPECT_MAT_NEAR(sepFilterGold(src, rowKernel, columnKernel, anchor, borderType), dst, sepFilterTolerance(src.depth()));
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Filters, SeparableLinearFilterLargeKSize, testing::Combine(
    ALL_DEVICES,
    testing::Values(cv::Size(160, 140), cv::Size(48, 32)),
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_16SC3), MatType(CV_32FC1), MatType(CV_32FC4)),
    testing::Values(KSize(cv::Size(33, 33)), KSize(cv::Size(65, 15)), KSize(cv::Size(15, 65)), KSize(cv::Size(133, 65))),
    testing::Values(BorderType(cv::BORDER_REFLECT101),
                    BorderType(cv::BORDER_REPLICATE),
                    BorderType(cv::BORDER_CONSTANT),
                    BorderType(cv::BORDER_REFLECT),
                    BorderType(cv::BORDER_WRAP)),
    WHOLE_SUBMAT));

// Kernels that reach more than one image length past an edge, including those of 32 or fewer coefficients.
INSTANTIATE_TEST_CASE_P(CUDA_Filters_WideKernel, SeparableLinearFilterLargeKSize, testing::Combine(
    ALL_DEVICES,
    testing::Values(cv::Size(8, 8), cv::Size(5, 3), cv::Size(20, 31), cv::Size(1, 16), cv::Size(16, 1), cv::Size(1, 1)),
    testing::Values(MatType(CV_8UC3), MatType(CV_32FC1)),
    testing::Values(KSize(cv::Size(3, 3)), KSize(cv::Size(9, 9)), KSize(cv::Size(15, 15)), KSize(cv::Size(16, 16)),
                    KSize(cv::Size(31, 31)), KSize(cv::Size(32, 32)), KSize(cv::Size(33, 33))),
    testing::Values(BorderType(cv::BORDER_REFLECT101),
                    BorderType(cv::BORDER_REPLICATE),
                    BorderType(cv::BORDER_CONSTANT),
                    BorderType(cv::BORDER_REFLECT),
                    BorderType(cv::BORDER_WRAP)),
    testing::Values(UseRoi(false))));

// One launch holds 65535 blocks of 8 rows along y, so this height needs a second pass of the y stride loop.
struct SeparableLinearFilterTallImage : testing::TestWithParam<cv::cuda::DeviceInfo>
{
    virtual void SetUp()
    {
        cv::cuda::setDevice(GetParam().deviceID());
    }
};

CUDA_TEST_P(SeparableLinearFilterTallImage, Accuracy)
{
    const cv::Size size(40, 524301);
    cv::Mat src = makeStructuredMat(size, CV_32FC1);
    cv::Mat rowKernel = normalizedKernel(33);
    cv::Mat columnKernel = normalizedKernel(65);

    cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createSeparableLinearFilter(src.type(), -1, rowKernel, columnKernel);

    cv::cuda::GpuMat dst;
    filter->apply(loadMat(src), dst);

    EXPECT_MAT_NEAR(sepFilterGold(src, rowKernel, columnKernel, cv::Point(-1, -1), cv::BORDER_DEFAULT), dst, sepFilterTolerance(src.depth()));
}

INSTANTIATE_TEST_CASE_P(CUDA_Filters, SeparableLinearFilterTallImage, ALL_DEVICES);

PARAM_TEST_CASE(SeparableLinearFilterWithEmptyKernels, cv::cuda::DeviceInfo, MatDepth, Channels, MatDepth, bool, bool, bool)
{
    cv::cuda::DeviceInfo devInfo;
    bool inPlace;
    bool useRowKernel;
    bool useColKernel;

    cv::Size size;
    int srcDepth;
    int cn;
    int dstDepth;
    cv::Size ksize;
    cv::Point anchor;
    int borderType;
    int srcType;
    int dstType;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        srcDepth = GET_PARAM(1);
        cn = GET_PARAM(2);
        dstDepth = GET_PARAM(3);
        inPlace = GET_PARAM(4);
        useRowKernel = GET_PARAM(5);
        useColKernel = GET_PARAM(6);

        size = cv::Size(640, 480);
        ksize = cv::Size(3, 1);
        anchor = cv::Point(-1, -1);
        borderType = cv::BORDER_REPLICATE;

        cv::cuda::setDevice(devInfo.deviceID());

        srcType = CV_MAKE_TYPE(srcDepth, cn);
        dstType = CV_MAKE_TYPE(dstDepth, cn);
    }
};

CUDA_TEST_P(SeparableLinearFilterWithEmptyKernels, Accuracy)
{
    cv::Mat src = randomMat(size, srcType);
    cv::Mat rowKernel = (cv::Mat_<float>(ksize) << -1, 0, 1);
    cv::Mat colKernel = rowKernel.t();
    cv::Mat oneKernel = cv::Mat::ones(cv::Size(1, 1), CV_32FC1);
    cv::Mat noKernel = cv::Mat();

    cv::Ptr<cv::cuda::Filter> sepFilterDummyKernels =
        cv::cuda::createSeparableLinearFilter(srcType, dstType,
            useRowKernel ? rowKernel : oneKernel,
            useColKernel ? colKernel : oneKernel,
            cv::Point(-1, -1), cv::BORDER_REPLICATE, cv::BORDER_REPLICATE);

    cv::Ptr<cv::cuda::Filter> sepFilterEmptyKernels =
        cv::cuda::createSeparableLinearFilter(srcType, dstType,
            useRowKernel ? rowKernel : noKernel,
            useColKernel ? colKernel : noKernel,
            cv::Point(-1, -1), cv::BORDER_REPLICATE, cv::BORDER_REPLICATE);

    cv::cuda::GpuMat src_sep_dummyK = loadMat(src);
    cv::cuda::GpuMat dst_sep_dummyK = inPlace ? src_sep_dummyK : cv::cuda::GpuMat();
    cv::cuda::GpuMat src_sep_emptyK = loadMat(src);
    cv::cuda::GpuMat dst_sep_emptyK = inPlace ? src_sep_emptyK : cv::cuda::GpuMat();

    sepFilterDummyKernels->apply(src_sep_dummyK, dst_sep_dummyK);
    sepFilterEmptyKernels->apply(src_sep_emptyK, dst_sep_emptyK);

    EXPECT_MAT_NEAR(dst_sep_dummyK, dst_sep_emptyK, src.depth() < CV_32F ? 1.0 : 1e-2);
}

INSTANTIATE_TEST_CASE_P(CUDA_Filters, SeparableLinearFilterWithEmptyKernels, testing::Combine(
    ALL_DEVICES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_16S), MatDepth(CV_32F)),
    IMAGE_CHANNELS,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_16S), MatDepth(CV_32F)),
    testing::Values(false, true),//in-place
    testing::Values(false, true),//use row kernel
    testing::Values(false, true)//use col kernel
    ));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Sobel

PARAM_TEST_CASE(Sobel, cv::cuda::DeviceInfo, cv::Size, MatDepth, Channels, KSize, Deriv_X, Deriv_Y, BorderType, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    int cn;
    cv::Size ksize;
    int dx;
    int dy;
    int borderType;
    bool useRoi;

    int type;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        cn = GET_PARAM(3);
        ksize = GET_PARAM(4);
        dx = GET_PARAM(5);
        dy = GET_PARAM(6);
        borderType = GET_PARAM(7);
        useRoi = GET_PARAM(8);

        cv::cuda::setDevice(devInfo.deviceID());

        type = CV_MAKE_TYPE(depth, cn);
    }
};

CUDA_TEST_P(Sobel, Accuracy)
{
    if (dx == 0 && dy == 0)
        return;

    cv::Mat src = randomMat(size, type);

    cv::Ptr<cv::cuda::Filter> sobel = cv::cuda::createSobelFilter(src.type(), -1, dx, dy, ksize.width, 1.0, borderType);

    cv::cuda::GpuMat dst = createMat(size, type, useRoi);
    sobel->apply(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::Sobel(src, dst_gold, -1, dx, dy, ksize.width, 1.0, 0.0, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() < CV_32F ? 0.0 : 0.1);
}

INSTANTIATE_TEST_CASE_P(CUDA_Filters, Sobel, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_16S), MatDepth(CV_32F)),
    IMAGE_CHANNELS,
    testing::Values(KSize(cv::Size(3, 3)), KSize(cv::Size(5, 5)), KSize(cv::Size(7, 7))),
    testing::Values(Deriv_X(0), Deriv_X(1), Deriv_X(2)),
    testing::Values(Deriv_Y(0), Deriv_Y(1), Deriv_Y(2)),
    testing::Values(BorderType(cv::BORDER_REFLECT101),
                    BorderType(cv::BORDER_REPLICATE),
                    BorderType(cv::BORDER_CONSTANT),
                    BorderType(cv::BORDER_REFLECT)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Scharr

PARAM_TEST_CASE(Scharr, cv::cuda::DeviceInfo, cv::Size, MatDepth, Channels, Deriv_X, Deriv_Y, BorderType, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    int cn;
    int dx;
    int dy;
    int borderType;
    bool useRoi;

    int type;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        cn = GET_PARAM(3);
        dx = GET_PARAM(4);
        dy = GET_PARAM(5);
        borderType = GET_PARAM(6);
        useRoi = GET_PARAM(7);

        cv::cuda::setDevice(devInfo.deviceID());

        type = CV_MAKE_TYPE(depth, cn);
    }
};

CUDA_TEST_P(Scharr, Accuracy)
{
    if (dx + dy != 1)
        return;

    cv::Mat src = randomMat(size, type);

    cv::Ptr<cv::cuda::Filter> scharr = cv::cuda::createScharrFilter(src.type(), -1, dx, dy, 1.0, borderType);

    cv::cuda::GpuMat dst = createMat(size, type, useRoi);
    scharr->apply(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::Scharr(src, dst_gold, -1, dx, dy, 1.0, 0.0, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() < CV_32F ? 0.0 : 0.1);
}

INSTANTIATE_TEST_CASE_P(CUDA_Filters, Scharr, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_16S), MatDepth(CV_32F)),
    IMAGE_CHANNELS,
    testing::Values(Deriv_X(0), Deriv_X(1)),
    testing::Values(Deriv_Y(0), Deriv_Y(1)),
    testing::Values(BorderType(cv::BORDER_REFLECT101),
                    BorderType(cv::BORDER_REPLICATE),
                    BorderType(cv::BORDER_CONSTANT),
                    BorderType(cv::BORDER_REFLECT)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// GaussianBlur

PARAM_TEST_CASE(GaussianBlur, cv::cuda::DeviceInfo, cv::Size, MatDepth, Channels, KSize, BorderType, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    int cn;
    cv::Size ksize;
    int borderType;
    bool useRoi;

    int type;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        cn = GET_PARAM(3);
        ksize = GET_PARAM(4);
        borderType = GET_PARAM(5);
        useRoi = GET_PARAM(6);

        cv::cuda::setDevice(devInfo.deviceID());

        type = CV_MAKE_TYPE(depth, cn);
    }
};

CUDA_TEST_P(GaussianBlur, Accuracy)
{
    cv::Mat src = randomMat(size, type);
    double sigma1 = randomDouble(0.0, 1.0);
    double sigma2 = randomDouble(0.0, 1.0);

    cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(src.type(), -1, ksize, sigma1, sigma2, borderType);

    cv::cuda::GpuMat dst = createMat(size, type, useRoi);
    gauss->apply(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::GaussianBlur(src, dst_gold, ksize, sigma1, sigma2, borderType);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() < CV_32F ? 4.0 : 1e-4);
}

INSTANTIATE_TEST_CASE_P(CUDA_Filters, GaussianBlur, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_16S), MatDepth(CV_32F)),
    IMAGE_CHANNELS,
    testing::Values(KSize(cv::Size(3, 3)),
                    KSize(cv::Size(5, 5)),
                    KSize(cv::Size(7, 7)),
                    KSize(cv::Size(9, 9)),
                    KSize(cv::Size(11, 11)),
                    KSize(cv::Size(13, 13)),
                    KSize(cv::Size(15, 15)),
                    KSize(cv::Size(17, 17)),
                    KSize(cv::Size(19, 19)),
                    KSize(cv::Size(21, 21)),
                    KSize(cv::Size(23, 23)),
                    KSize(cv::Size(25, 25)),
                    KSize(cv::Size(27, 27)),
                    KSize(cv::Size(29, 29)),
                    KSize(cv::Size(31, 31))),
    testing::Values(BorderType(cv::BORDER_REFLECT101),
                    BorderType(cv::BORDER_REPLICATE),
                    BorderType(cv::BORDER_CONSTANT),
                    BorderType(cv::BORDER_REFLECT)),
    WHOLE_SUBMAT));

PARAM_TEST_CASE(GaussianBlurLargeKSize, cv::cuda::DeviceInfo, cv::Size, MatType, KSize, GaussianSigma, BorderType, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int type;
    cv::Size ksize;
    double sigma;
    int borderType;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        ksize = GET_PARAM(3);
        sigma = GET_PARAM(4);
        borderType = GET_PARAM(5);
        useRoi = GET_PARAM(6);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(GaussianBlurLargeKSize, Accuracy)
{
    cv::Mat src = makeStructuredMat(size, type);

    cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(src.type(), -1, ksize, sigma, sigma, borderType);

    cv::cuda::GpuMat dst = createMat(size, type, useRoi);
    gauss->apply(loadMat(src, useRoi), dst);

    // For 8-bit input cv::GaussianBlur keeps 8 fractional bits per coefficient, too coarse for kernels this long.
    // This filter computes in float, so the reference does too.
    const int factor = src.depth() == CV_8U ? 3 : 4;
    const int kw = ksize.width > 0 ? ksize.width : cvRound(sigma * factor * 2 + 1) | 1;
    const int kh = ksize.height > 0 ? ksize.height : cvRound(sigma * factor * 2 + 1) | 1;
    cv::Mat src32F, dst_gold;
    src.convertTo(src32F, CV_32F);
    sepFilterGold(src32F, cv::getGaussianKernel(kw, sigma, CV_32F).t(), cv::getGaussianKernel(kh, sigma, CV_32F).t(), cv::Point(-1, -1), borderType)
        .convertTo(dst_gold, src.depth());

    EXPECT_MAT_NEAR(dst_gold, dst, sepFilterTolerance(src.depth()));
}

#define GAUSSIAN_LARGE_KSIZE_TYPES \
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_16UC1), MatType(CV_32FC1))

#define GAUSSIAN_LARGE_KSIZE_BORDERS \
    testing::Values(BorderType(cv::BORDER_REFLECT101), \
                    BorderType(cv::BORDER_REPLICATE), \
                    BorderType(cv::BORDER_CONSTANT), \
                    BorderType(cv::BORDER_REFLECT), \
                    BorderType(cv::BORDER_WRAP))

INSTANTIATE_TEST_CASE_P(CUDA_Filters, GaussianBlurLargeKSize, testing::Combine(
    ALL_DEVICES,
    testing::Values(cv::Size(160, 140), cv::Size(48, 32)),
    GAUSSIAN_LARGE_KSIZE_TYPES,
    testing::Values(KSize(cv::Size(65, 65)), KSize(cv::Size(133, 133))),
    testing::Values(GaussianSigma(0.0), GaussianSigma(22.0)),
    GAUSSIAN_LARGE_KSIZE_BORDERS,
    WHOLE_SUBMAT));

// The kernel size comes from sigma here, so sigma has to be set.
INSTANTIATE_TEST_CASE_P(CUDA_Filters_KSizeFromSigma, GaussianBlurLargeKSize, testing::Combine(
    ALL_DEVICES,
    testing::Values(cv::Size(160, 140), cv::Size(48, 32)),
    GAUSSIAN_LARGE_KSIZE_TYPES,
    testing::Values(KSize(cv::Size(0, 0))),
    testing::Values(GaussianSigma(22.0)),
    GAUSSIAN_LARGE_KSIZE_BORDERS,
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Erode

PARAM_TEST_CASE(Erode, cv::cuda::DeviceInfo, cv::Size, MatType, Anchor, Iterations, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int type;
    cv::Point anchor;
    int iterations;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        anchor = GET_PARAM(3);
        iterations = GET_PARAM(4);
        useRoi = GET_PARAM(5);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Erode, Accuracy)
{
    cv::Mat src = randomMat(size, type);
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);

    cv::Ptr<cv::cuda::Filter> erode = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, src.type(), kernel, anchor, iterations);

    cv::cuda::GpuMat dst = createMat(size, type, useRoi);
    erode->apply(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::erode(src, dst_gold, kernel, anchor, iterations);

    cv::Size ksize = cv::Size(kernel.cols + iterations * (kernel.cols - 1), kernel.rows + iterations * (kernel.rows - 1));

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, ksize), getInnerROI(dst, ksize), 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Filters, Erode, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4)),
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    testing::Values(Iterations(1), Iterations(2), Iterations(3)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Dilate

PARAM_TEST_CASE(Dilate, cv::cuda::DeviceInfo, cv::Size, MatType, Anchor, Iterations, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int type;
    cv::Point anchor;
    int iterations;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        anchor = GET_PARAM(3);
        iterations = GET_PARAM(4);
        useRoi = GET_PARAM(5);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Dilate, Accuracy)
{
    cv::Mat src = randomMat(size, type);
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);

    cv::Ptr<cv::cuda::Filter> dilate = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, src.type(), kernel, anchor, iterations);

    cv::cuda::GpuMat dst = createMat(size, type, useRoi);
    dilate->apply(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::dilate(src, dst_gold, kernel, anchor, iterations);

    cv::Size ksize = cv::Size(kernel.cols + iterations * (kernel.cols - 1), kernel.rows + iterations * (kernel.rows - 1));

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, ksize), getInnerROI(dst, ksize), 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Filters, Dilate, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4)),
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    testing::Values(Iterations(1), Iterations(2), Iterations(3)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// MorphEx

CV_ENUM(MorphOp, MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT, MORPH_TOPHAT, MORPH_BLACKHAT)

PARAM_TEST_CASE(MorphEx, cv::cuda::DeviceInfo, cv::Size, MatType, MorphOp, Anchor, Iterations, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int type;
    int morphOp;
    cv::Point anchor;
    int iterations;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        morphOp = GET_PARAM(3);
        anchor = GET_PARAM(4);
        iterations = GET_PARAM(5);
        useRoi = GET_PARAM(6);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(MorphEx, Accuracy)
{
    cv::Mat src = randomMat(size, type);
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);

    cv::Ptr<cv::cuda::Filter> morph = cv::cuda::createMorphologyFilter(morphOp, src.type(), kernel, anchor, iterations);

    cv::cuda::GpuMat dst = createMat(size, type, useRoi);
    morph->apply(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::morphologyEx(src, dst_gold, morphOp, kernel, anchor, iterations);

    cv::Size border = cv::Size(kernel.cols + (iterations + 1) * kernel.cols + 2, kernel.rows + (iterations + 1) * kernel.rows + 2);

    EXPECT_MAT_NEAR(getInnerROI(dst_gold, border), getInnerROI(dst, border), 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Filters, MorphEx, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC4)),
    MorphOp::all(),
    testing::Values(Anchor(cv::Point(-1, -1)), Anchor(cv::Point(0, 0)), Anchor(cv::Point(2, 2))),
    testing::Values(Iterations(1), Iterations(2), Iterations(3)),
    WHOLE_SUBMAT));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Median


PARAM_TEST_CASE(Median, cv::cuda::DeviceInfo, cv::Size, MatType, KernelSize, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int type;
    int kernel;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        kernel = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};



CUDA_TEST_P(Median, Accuracy)
{
    cv::Mat src = randomMat(size, type);

    cv::Ptr<cv::cuda::Filter> median = cv::cuda::createMedianFilter(src.type(), kernel);

    cv::cuda::GpuMat dst = createMat(size, type, useRoi);
    median->apply(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::medianBlur(src,dst_gold,kernel);

    cv::Rect rect(kernel/2, kernel/2, src.cols-(kernel-1), src.rows-(kernel-1));
    cv::Mat dst_gold_no_border = dst_gold(rect);
    cv::cuda::GpuMat dst_no_border = cv::cuda::GpuMat(dst, rect);

    EXPECT_MAT_NEAR(dst_gold_no_border, dst_no_border, 1);

}

INSTANTIATE_TEST_CASE_P(CUDA_Filters, Median, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatDepth(CV_8U)),
    testing::Values(KernelSize(3),
                    KernelSize(5),
                    KernelSize(7),
                    KernelSize(9),
                    KernelSize(11),
                    KernelSize(13),
                    KernelSize(15)),
    WHOLE_SUBMAT)
    );

INSTANTIATE_TEST_CASE_P(CUDA_Filters_Median_HDR, Median, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(
        MatType(CV_8UC3), MatType(CV_8UC4),
        MatType(CV_16U), MatType(CV_16UC3), MatType(CV_16UC4),
        MatType(CV_32F), MatType(CV_32FC3), MatType(CV_32FC4)),
    testing::Values(KernelSize(3), KernelSize(5)),
    WHOLE_SUBMAT)
    );


}} // namespace
#endif // HAVE_CUDA
