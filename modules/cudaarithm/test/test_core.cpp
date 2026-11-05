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

namespace opencv_test { namespace {

////////////////////////////////////////////////////////////////////////////////
// Merge

PARAM_TEST_CASE(Merge, cv::cuda::DeviceInfo, cv::Size, MatDepth, Channels, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    int channels;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        channels = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Merge, Accuracy)
{
    std::vector<cv::Mat> src;
    src.reserve(channels);
    for (int i = 0; i < channels; ++i)
        src.push_back(randomMat(size, CV_MAKE_TYPE(depth, 1)));

    std::vector<cv::cuda::GpuMat> d_src;
    for (int i = 0; i < channels; ++i)
        d_src.push_back(loadMat(src[i], useRoi));

    if (depth == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::merge(d_src, dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst;
        cv::cuda::merge(d_src, dst);

        cv::Mat dst_gold;
        cv::merge(src, dst_gold);

        ASSERT_EQ(dst_gold.type(), dst.type());
        ASSERT_EQ(dst_gold.size(), dst.size());
        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
}

CUDA_TEST_P(Merge, MultiChannelInputs)
{
    if (depth == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
        return;

    std::vector<cv::Mat> src;
    if (channels == 1)
    {
        src.push_back(randomMat(size, CV_MAKE_TYPE(depth, 3)));
    }
    else if (channels <= 3)
    {
        src.push_back(randomMat(size, CV_MAKE_TYPE(depth, 2)));
        src.push_back(randomMat(size, CV_MAKE_TYPE(depth, 1)));
    }
    else if (channels == 4)
    {
        src.push_back(randomMat(size, CV_MAKE_TYPE(depth, 3)));
        src.push_back(randomMat(size, CV_MAKE_TYPE(depth, 1)));
    }
    else
    {
        src.push_back(randomMat(size, CV_MAKE_TYPE(depth, 3)));
        src.push_back(randomMat(size, CV_MAKE_TYPE(depth, 1)));
        src.push_back(randomMat(size, CV_MAKE_TYPE(depth, 2)));
    }

    std::vector<cv::cuda::GpuMat> d_src;
    for (size_t i = 0; i < src.size(); ++i)
        d_src.push_back(loadMat(src[i], useRoi));

    cv::cuda::GpuMat dst;
    cv::cuda::merge(d_src, dst);

    cv::Mat dst_gold;
    cv::merge(src, dst_gold);

    ASSERT_EQ(dst_gold.type(), dst.type());
    ASSERT_EQ(dst_gold.size(), dst.size());
    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Merge, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    testing::Values(1, 2, 3, 4, 6),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Split

PARAM_TEST_CASE(Split, cv::cuda::DeviceInfo, cv::Size, MatDepth, Channels, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int depth;
    int channels;
    bool useRoi;

    int type;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        depth = GET_PARAM(2);
        channels = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::cuda::setDevice(devInfo.deviceID());

        type = CV_MAKE_TYPE(depth, channels);
    }
};

CUDA_TEST_P(Split, Accuracy)
{
    cv::Mat src;
    if (channels <= 4)
    {
        src = randomMat(size, type);
    }
    else
    {
        std::vector<cv::Mat> planes(channels);
        for (int i = 0; i < channels; ++i)
            planes[i] = randomMat(size, CV_MAKE_TYPE(depth, 1));
        cv::merge(planes, src);
    }

    if (depth == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            std::vector<cv::cuda::GpuMat> dst;
            cv::cuda::split(loadMat(src), dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        std::vector<cv::cuda::GpuMat> dst;
        cv::cuda::split(loadMat(src, useRoi), dst);

        std::vector<cv::Mat> dst_gold;
        cv::split(src, dst_gold);

        ASSERT_EQ(dst_gold.size(), dst.size());

        for (size_t i = 0; i < dst_gold.size(); ++i)
        {
            EXPECT_MAT_NEAR(dst_gold[i], dst[i], 0.0);
        }
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Split, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    ALL_DEPTH,
    testing::Values(1, 2, 3, 4, 6),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// MergeSplitEdgeCases

PARAM_TEST_CASE(MergeSplitEdgeCases, cv::cuda::DeviceInfo)
{
    cv::cuda::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(MergeSplitEdgeCases, Batch65PointerStreamAndHostOutput)
{
    const cv::Size sz(37, 29);
    const int num_ch = 65;

    std::vector<cv::Mat> h_planes(num_ch);
    std::vector<cv::cuda::GpuMat> d_planes(num_ch);
    for (int i = 0; i < num_ch; ++i)
    {
        h_planes[i] = randomMat(sz, CV_16UC1);
        d_planes[i] = loadMat(h_planes[i], true);
    }

    cv::cuda::Stream stream;
    cv::cuda::GpuMat d_merged;
    cv::cuda::merge(d_planes.data(), d_planes.size(), d_merged, stream);

    std::vector<cv::cuda::GpuMat> d_split(num_ch);
    cv::cuda::split(d_merged, d_split.data(), stream);
    stream.waitForCompletion();

    cv::Mat h_merged_gold;
    cv::merge(h_planes, h_merged_gold);
    EXPECT_MAT_NEAR(h_merged_gold, d_merged, 0.0);

    for (int i = 0; i < num_ch; ++i)
        EXPECT_MAT_NEAR(h_planes[i], d_split[i], 0.0);

    // Host cv::Mat OutputArray (both n == 1 and n > 4)
    cv::Mat h_out_single, h_out_multi;
    cv::cuda::merge(&d_planes[0], 1, h_out_single);
    cv::cuda::merge(d_planes.data(), 6, h_out_multi);
    EXPECT_MAT_NEAR(h_planes[0], h_out_single, 0.0);

    std::vector<cv::Mat> h_planes6(h_planes.begin(), h_planes.begin() + 6);
    cv::Mat h_gold6;
    cv::merge(h_planes6, h_gold6);
    EXPECT_MAT_NEAR(h_gold6, h_out_multi, 0.0);
}

CUDA_TEST_P(MergeSplitEdgeCases, FP16AliasingAndEmpty)
{
    const cv::Size sz(31, 19);
    std::vector<cv::Mat> h_fp16(6);
    std::vector<cv::cuda::GpuMat> d_fp16(6);
    for (int i = 0; i < 6; ++i)
    {
        randomMat(sz, CV_32FC1, -10.0, 10.0).convertTo(h_fp16[i], CV_16F);
        d_fp16[i] = loadMat(h_fp16[i], false);
    }

    cv::cuda::GpuMat d_merged16;
    cv::cuda::merge(d_fp16, d_merged16);
    std::vector<cv::cuda::GpuMat> d_split16;
    cv::cuda::split(d_merged16, d_split16);
    ASSERT_EQ(6u, d_split16.size());
    for (int i = 0; i < 6; ++i)
    {
        cv::Mat expected32, actual32;
        h_fp16[i].convertTo(expected32, CV_32F);
        cv::Mat(d_split16[i]).convertTo(actual32, CV_32F);
        EXPECT_MAT_NEAR(expected32, actual32, 0.0);
    }

    // In-place / aliased destination handling
    std::vector<cv::Mat> h_rgb(3);
    std::vector<cv::cuda::GpuMat> d_rgb(3);
    for (int i = 0; i < 3; ++i)
    {
        h_rgb[i] = randomMat(sz, CV_8UC1);
        d_rgb[i] = loadMat(h_rgb[i]);
    }
    cv::Mat h_rgb_gold;
    cv::merge(h_rgb, h_rgb_gold);
    cv::cuda::merge(d_rgb, d_rgb[0]);
    EXPECT_MAT_NEAR(h_rgb_gold, d_rgb[0], 0.0);

    cv::cuda::GpuMat d_single = loadMat(h_rgb[1]);
    cv::cuda::merge(&d_single, 1, d_single);
    cv::cuda::split(d_single, &d_single);
    EXPECT_MAT_NEAR(h_rgb[1], d_single, 0.0);

    // Empty handling
    std::vector<cv::cuda::GpuMat> non_empty_vec(3, d_single);
    cv::cuda::split(cv::cuda::GpuMat(), non_empty_vec);
    EXPECT_TRUE(non_empty_vec.empty());

    std::vector<cv::cuda::GpuMat> empty_src;
    cv::cuda::GpuMat dummy_dst;
    EXPECT_THROW(cv::cuda::merge(empty_src, dummy_dst), cv::Exception);
    cv::cuda::GpuMat empty_plane;
    EXPECT_THROW(cv::cuda::merge(&empty_plane, 1, dummy_dst), cv::Exception);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, MergeSplitEdgeCases, ALL_DEVICES);

////////////////////////////////////////////////////////////////////////////////
// Transpose

PARAM_TEST_CASE(Transpose, cv::cuda::DeviceInfo, cv::Size, MatType, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int type;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Transpose, Accuracy)
{
    cv::Mat src = randomMat(size, type);

    if (CV_MAT_DEPTH(type) == CV_64F && !supportFeature(devInfo, cv::cuda::NATIVE_DOUBLE))
    {
        try
        {
            cv::cuda::GpuMat dst;
            cv::cuda::transpose(loadMat(src), dst);
        }
        catch (const cv::Exception& e)
        {
            ASSERT_EQ(cv::Error::StsUnsupportedFormat, e.code);
        }
    }
    else
    {
        cv::cuda::GpuMat dst = createMat(cv::Size(size.height, size.width), type, useRoi);
        cv::cuda::transpose(loadMat(src, useRoi), dst);

        cv::Mat dst_gold;
        cv::transpose(src, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Transpose, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1),
                    MatType(CV_8UC4),
                    MatType(CV_16UC2),
                    MatType(CV_16SC2),
                    MatType(CV_32SC1),
                    MatType(CV_32SC2),
                    MatType(CV_64FC1)),
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// Flip

enum {FLIP_BOTH = 0, FLIP_X = 1, FLIP_Y = -1};
CV_ENUM(FlipCode, FLIP_BOTH, FLIP_X, FLIP_Y)
#define ALL_FLIP_CODES testing::Values(FlipCode(FLIP_BOTH), FlipCode(FLIP_X), FlipCode(FLIP_Y))

PARAM_TEST_CASE(Flip, cv::cuda::DeviceInfo, cv::Size, MatType, FlipCode, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int type;
    int flip_code;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        flip_code = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Flip, Accuracy)
{
    cv::Mat src = randomMat(size, type);

    cv::cuda::GpuMat dst = createMat(size, type, useRoi);
    cv::cuda::flip(loadMat(src, useRoi), dst, flip_code);

    cv::Mat dst_gold;
    cv::flip(src, dst_gold, flip_code);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

CUDA_TEST_P(Flip, AccuracyInplace)
{
    cv::Mat src = randomMat(size, type);
    bool isSizeOdd = ((size.width & 1) == 1) || ((size.height & 1) == 1);
    cv::cuda::GpuMat srcDst = loadMat(src, useRoi);
    if(isSizeOdd)
    {
        EXPECT_THROW(cv::cuda::flip(srcDst, srcDst, flip_code), cv::Exception);
        return;
    }
    cv::cuda::flip(srcDst, srcDst, flip_code);

    cv::Mat dst_gold;
    cv::flip(src, dst_gold, flip_code);

    EXPECT_MAT_NEAR(dst_gold, srcDst, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, Flip, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1),
                    MatType(CV_8UC3),
                    MatType(CV_8UC4),
                    MatType(CV_16UC1),
                    MatType(CV_16UC3),
                    MatType(CV_16UC4),
                    MatType(CV_32SC1),
                    MatType(CV_32SC3),
                    MatType(CV_32SC4),
                    MatType(CV_32FC1),
                    MatType(CV_32FC3),
                    MatType(CV_32FC4)),
    ALL_FLIP_CODES,
    WHOLE_SUBMAT));

////////////////////////////////////////////////////////////////////////////////
// LUT

PARAM_TEST_CASE(LUT, cv::cuda::DeviceInfo, cv::Size, MatType, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int type;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        useRoi = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(LUT, OneChannel)
{
    cv::Mat src = randomMat(size, type);
    cv::Mat lut = randomMat(cv::Size(256, 1), CV_8UC1);

    cv::Ptr<cv::cuda::LookUpTable> lutAlg = cv::cuda::createLookUpTable(lut);

    cv::cuda::GpuMat dst = createMat(size, CV_MAKE_TYPE(lut.depth(), src.channels()));
    lutAlg->transform(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::LUT(src, lut, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

CUDA_TEST_P(LUT, MultiChannel)
{
    cv::Mat src = randomMat(size, type);
    cv::Mat lut = randomMat(cv::Size(256, 1), CV_MAKE_TYPE(CV_8U, src.channels()));

    cv::Ptr<cv::cuda::LookUpTable> lutAlg = cv::cuda::createLookUpTable(lut);

    cv::cuda::GpuMat dst = createMat(size, CV_MAKE_TYPE(lut.depth(), src.channels()), useRoi);
    lutAlg->transform(loadMat(src, useRoi), dst);

    cv::Mat dst_gold;
    cv::LUT(src, lut, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, LUT, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3)),
    WHOLE_SUBMAT));

//////////////////////////////////////////////////////////////////////////////
// CopyMakeBorder

namespace
{
    IMPLEMENT_PARAM_CLASS(Border, int)
}

PARAM_TEST_CASE(CopyMakeBorder, cv::cuda::DeviceInfo, cv::Size, MatType, Border, BorderType, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int type;
    int border;
    int borderType;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        border = GET_PARAM(3);
        borderType = GET_PARAM(4);
        useRoi = GET_PARAM(5);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(CopyMakeBorder, Accuracy)
{
    cv::Mat src = randomMat(size, type);
    cv::Scalar val = randomScalar(0, 255);

    cv::cuda::GpuMat dst = createMat(cv::Size(size.width + 2 * border, size.height + 2 * border), type, useRoi);
    cv::cuda::copyMakeBorder(loadMat(src, useRoi), dst, border, border, border, border, borderType, val);

    cv::Mat dst_gold;
    cv::copyMakeBorder(src, dst_gold, border, border, border, border, borderType, val);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Arithm, CopyMakeBorder, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1),
                    MatType(CV_8UC3),
                    MatType(CV_8UC4),
                    MatType(CV_16UC1),
                    MatType(CV_16UC3),
                    MatType(CV_16UC4),
                    MatType(CV_32FC1),
                    MatType(CV_32FC3),
                    MatType(CV_32FC4)),
    testing::Values(Border(1), Border(10), Border(50)),
    ALL_BORDER_TYPES,
    WHOLE_SUBMAT));


}} // namespace
#endif // HAVE_CUDA
