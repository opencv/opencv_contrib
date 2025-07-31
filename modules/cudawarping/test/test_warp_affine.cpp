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

namespace
{
    cv::Mat createTransformMatrix(cv::Size srcSize, double angle)
    {
        cv::Mat M(2, 3, CV_64FC1);

        M.at<double>(0, 0) = std::cos(angle); M.at<double>(0, 1) = -std::sin(angle); M.at<double>(0, 2) = srcSize.width / 2;
        M.at<double>(1, 0) = std::sin(angle); M.at<double>(1, 1) =  std::cos(angle); M.at<double>(1, 2) = 0.0;

        return M;
    }
}

///////////////////////////////////////////////////////////////////
// Test buildWarpAffineMaps

PARAM_TEST_CASE(BuildWarpAffineMaps, cv::cuda::DeviceInfo, cv::Size, Inverse)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    bool inverse;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        inverse = GET_PARAM(2);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(BuildWarpAffineMaps, Accuracy)
{
    cv::Mat M = createTransformMatrix(size, CV_PI / 4);
    cv::Mat src = randomMat(randomSize(200, 400), CV_8UC1);

    cv::cuda::GpuMat xmap, ymap;
    cv::cuda::buildWarpAffineMaps(M, inverse, size, xmap, ymap);

    int interpolation = cv::INTER_NEAREST;
    int borderMode = cv::BORDER_CONSTANT;
    int flags = interpolation;
    if (inverse)
        flags |= cv::WARP_INVERSE_MAP;

    cv::Mat dst;
    cv::remap(src, dst, cv::Mat(xmap), cv::Mat(ymap), interpolation, borderMode);

    cv::Mat dst_gold;
    cv::warpAffine(src, dst_gold, M, size, flags, borderMode);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Warping, BuildWarpAffineMaps, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    DIRECT_INVERSE));

///////////////////////////////////////////////////////////////////
// Gold implementation

namespace
{
    template <typename T, template <typename> class Interpolator> void warpAffineImpl(const cv::Mat& src, const cv::Mat& M, cv::Size dsize, cv::Mat& dst, int borderType, cv::Scalar borderVal)
    {
        const int cn = src.channels();

        dst.create(dsize, src.type());

        for (int y = 0; y < dsize.height; ++y)
        {
            for (int x = 0; x < dsize.width; ++x)
            {
                float xcoo = static_cast<float>(M.at<double>(0, 0) * x + M.at<double>(0, 1) * y + M.at<double>(0, 2));
                float ycoo = static_cast<float>(M.at<double>(1, 0) * x + M.at<double>(1, 1) * y + M.at<double>(1, 2));

                for (int c = 0; c < cn; ++c)
                    dst.at<T>(y, x * cn + c) = Interpolator<T>::getValue(src, ycoo, xcoo, c, borderType, borderVal);
            }
        }
    }

    void warpAffineGold(const cv::Mat& src, const cv::Mat& M, bool inverse, cv::Size dsize, cv::Mat& dst, int interpolation, int borderType, cv::Scalar borderVal)
    {
        typedef void (*func_t)(const cv::Mat& src, const cv::Mat& M, cv::Size dsize, cv::Mat& dst, int borderType, cv::Scalar borderVal);

        static const func_t nearest_funcs[] =
        {
            warpAffineImpl<unsigned char, NearestInterpolator>,
            warpAffineImpl<signed char, NearestInterpolator>,
            warpAffineImpl<unsigned short, NearestInterpolator>,
            warpAffineImpl<short, NearestInterpolator>,
            warpAffineImpl<int, NearestInterpolator>,
            warpAffineImpl<float, NearestInterpolator>
        };

        static const func_t linear_funcs[] =
        {
            warpAffineImpl<unsigned char, LinearInterpolator>,
            warpAffineImpl<signed char, LinearInterpolator>,
            warpAffineImpl<unsigned short, LinearInterpolator>,
            warpAffineImpl<short, LinearInterpolator>,
            warpAffineImpl<int, LinearInterpolator>,
            warpAffineImpl<float, LinearInterpolator>
        };

        static const func_t cubic_funcs[] =
        {
            warpAffineImpl<unsigned char, CubicInterpolator>,
            warpAffineImpl<signed char, CubicInterpolator>,
            warpAffineImpl<unsigned short, CubicInterpolator>,
            warpAffineImpl<short, CubicInterpolator>,
            warpAffineImpl<int, CubicInterpolator>,
            warpAffineImpl<float, CubicInterpolator>
        };

        static const func_t* funcs[] = {nearest_funcs, linear_funcs, cubic_funcs};

        if (inverse)
            funcs[interpolation][src.depth()](src, M, dsize, dst, borderType, borderVal);
        else
        {
            cv::Mat iM;
            cv::invertAffineTransform(M, iM);
            funcs[interpolation][src.depth()](src, iM, dsize, dst, borderType, borderVal);
        }
    }
}

///////////////////////////////////////////////////////////////////
// Test

PARAM_TEST_CASE(WarpAffine, cv::cuda::DeviceInfo, cv::Size, MatType, Inverse, Interpolation, BorderType, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int type;
    bool inverse;
    int interpolation;
    int borderType;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        inverse = GET_PARAM(3);
        interpolation = GET_PARAM(4);
        borderType = GET_PARAM(5);
        useRoi = GET_PARAM(6);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(WarpAffine, Accuracy)
{
    cv::Mat src = randomMat(size, type);
    cv::Mat M = createTransformMatrix(size, CV_PI / 3);
    int flags = interpolation;
    if (inverse)
        flags |= cv::WARP_INVERSE_MAP;
    cv::Scalar val = randomScalar(0.0, 255.0);

    cv::cuda::GpuMat dst = createMat(size, type, useRoi);
    cv::cuda::warpAffine(loadMat(src, useRoi), dst, M, size, flags, borderType, val);

    cv::Mat dst_gold;
    warpAffineGold(src, M, inverse, size, dst_gold, interpolation, borderType, val);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() == CV_32F ? 1e-1 : 1.0);
}

CUDA_TEST_P(WarpAffine, OverlapDetection)
{
    cv::Mat src = randomMat(size, type);
    ASSERT_FALSE(src.empty());
    cv::cuda::GpuMat gpuSrc;
    gpuSrc.upload(src);

    cv::Mat M = cv::Mat::eye(2, 3, CV_64FC1);
    int flags = interpolation;
    if (inverse)
        flags |= cv::WARP_INVERSE_MAP;

    {
        cv::cuda::GpuMat gpuDst(gpuSrc, cv::Rect(0, 0, size.width, size.height));

        EXPECT_THROW(
            cv::cuda::warpAffine(gpuSrc, gpuDst, M, size, flags, borderType, cv::Scalar::all(0)),
            cv::Exception);
    }

    {
        cv::cuda::GpuMat gpuDst(size, gpuSrc.type());
        ASSERT_NE(gpuSrc.data, gpuDst.data); // Confirm they are distinct

        EXPECT_NO_THROW({
            cv::cuda::warpAffine(gpuSrc, gpuDst, M, size, flags, borderType, cv::Scalar::all(0));
        });
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Warping, WarpAffine, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4), MatType(CV_16UC1), MatType(CV_16UC3), MatType(CV_16UC4), MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4)),
    DIRECT_INVERSE,
    testing::Values(Interpolation(cv::INTER_NEAREST), Interpolation(cv::INTER_LINEAR), Interpolation(cv::INTER_CUBIC)),
    testing::Values(BorderType(cv::BORDER_REFLECT101), BorderType(cv::BORDER_REPLICATE), BorderType(cv::BORDER_REFLECT), BorderType(cv::BORDER_WRAP)),
    WHOLE_SUBMAT));

///////////////////////////////////////////////////////////////////
// Test NPP

PARAM_TEST_CASE(WarpAffineNPP, cv::cuda::DeviceInfo, MatType, Inverse, Interpolation)
{
    cv::cuda::DeviceInfo devInfo;
    int type;
    bool inverse;
    int interpolation;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        type = GET_PARAM(1);
        inverse = GET_PARAM(2);
        interpolation = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(WarpAffineNPP, Accuracy)
{
    cv::Mat src = readImageType("stereobp/aloe-L.png", type);
    ASSERT_FALSE(src.empty());

    cv::Mat M = createTransformMatrix(src.size(), CV_PI / 4);
    int flags = interpolation;
    if (inverse)
        flags |= cv::WARP_INVERSE_MAP;

    cv::cuda::GpuMat dst;
    cv::cuda::warpAffine(loadMat(src), dst, M, src.size(), flags);

    cv::Mat dst_gold;
    warpAffineGold(src, M, inverse, src.size(), dst_gold, interpolation, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    EXPECT_MAT_SIMILAR(dst_gold, dst, 2e-2);
}

INSTANTIATE_TEST_CASE_P(CUDA_Warping, WarpAffineNPP, testing::Combine(
    ALL_DEVICES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4), MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4)),
    DIRECT_INVERSE,
    testing::Values(Interpolation(cv::INTER_NEAREST), Interpolation(cv::INTER_LINEAR), Interpolation(cv::INTER_CUBIC))));


}} // namespace
#endif // HAVE_CUDA
