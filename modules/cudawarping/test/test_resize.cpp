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

///////////////////////////////////////////////////////////////////
// Gold implementation

namespace
{
    template <typename T, template <typename> class Interpolator>
    void resizeImpl(const cv::Mat& src, cv::Mat& dst, double fx, double fy)
    {
        const int cn = src.channels();

        cv::Size dsize(cv::saturate_cast<int>(src.cols * fx), cv::saturate_cast<int>(src.rows * fy));

        dst.create(dsize, src.type());

        float ifx = static_cast<float>(1.0 / fx);
        float ify = static_cast<float>(1.0 / fy);

        for (int y = 0; y < dsize.height; ++y)
        {
            for (int x = 0; x < dsize.width; ++x)
            {
                for (int c = 0; c < cn; ++c)
                    dst.at<T>(y, x * cn + c) = Interpolator<T>::getValue(src, y * ify, x * ifx, c, cv::BORDER_REPLICATE);
            }
        }
    }

    void resizeGold(const cv::Mat& src, cv::Mat& dst, double fx, double fy, int interpolation)
    {
        typedef void (*func_t)(const cv::Mat& src, cv::Mat& dst, double fx, double fy);

        static const func_t nearest_funcs[] =
        {
            resizeImpl<unsigned char, NearestInterpolator>,
            resizeImpl<signed char, NearestInterpolator>,
            resizeImpl<unsigned short, NearestInterpolator>,
            resizeImpl<short, NearestInterpolator>,
            resizeImpl<int, NearestInterpolator>,
            resizeImpl<float, NearestInterpolator>
        };


        static const func_t linear_funcs[] =
        {
            resizeImpl<unsigned char, LinearInterpolator>,
            resizeImpl<signed char, LinearInterpolator>,
            resizeImpl<unsigned short, LinearInterpolator>,
            resizeImpl<short, LinearInterpolator>,
            resizeImpl<int, LinearInterpolator>,
            resizeImpl<float, LinearInterpolator>
        };

        static const func_t cubic_funcs[] =
        {
            resizeImpl<unsigned char, CubicInterpolator>,
            resizeImpl<signed char, CubicInterpolator>,
            resizeImpl<unsigned short, CubicInterpolator>,
            resizeImpl<short, CubicInterpolator>,
            resizeImpl<int, CubicInterpolator>,
            resizeImpl<float, CubicInterpolator>
        };

        static const func_t* funcs[] = {nearest_funcs, linear_funcs, cubic_funcs};

        funcs[interpolation][src.depth()](src, dst, fx, fy);
    }
}

///////////////////////////////////////////////////////////////////
// Test

PARAM_TEST_CASE(Resize, cv::cuda::DeviceInfo, cv::Size, MatType, double, Interpolation, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    double coeff;
    int interpolation;
    int type;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        coeff = GET_PARAM(3);
        interpolation = GET_PARAM(4);
        useRoi = GET_PARAM(5);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(Resize, Accuracy)
{
    cv::Mat src = randomMat(size, type);

    cv::cuda::GpuMat dst = createMat(cv::Size(cv::saturate_cast<int>(src.cols * coeff), cv::saturate_cast<int>(src.rows * coeff)), type, useRoi);
    cv::cuda::resize(loadMat(src, useRoi), dst, cv::Size(), coeff, coeff, interpolation);

    cv::Mat dst_gold;
    resizeGold(src, dst_gold, coeff, coeff, interpolation);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() == CV_32F ? 1e-2 : 1.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Warping, Resize, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4), MatType(CV_16UC1), MatType(CV_16UC3), MatType(CV_16UC4), MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4)),
    testing::Values(0.3, 0.5, 1.5, 2.0),
    testing::Values(Interpolation(cv::INTER_NEAREST), Interpolation(cv::INTER_LINEAR), Interpolation(cv::INTER_CUBIC)),
    WHOLE_SUBMAT));

/////////////////

PARAM_TEST_CASE(ResizeSameAsHost, cv::cuda::DeviceInfo, cv::Size, MatType, double, Interpolation, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    double coeff;
    int interpolation;
    int type;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        coeff = GET_PARAM(3);
        interpolation = GET_PARAM(4);
        useRoi = GET_PARAM(5);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

// downscaling only: used for classifiers
CUDA_TEST_P(ResizeSameAsHost, Accuracy)
{
    cv::Mat src = randomMat(size, type);

    cv::cuda::GpuMat dst = createMat(cv::Size(cv::saturate_cast<int>(src.cols * coeff), cv::saturate_cast<int>(src.rows * coeff)), type, useRoi);
    cv::cuda::resize(loadMat(src, useRoi), dst, cv::Size(), coeff, coeff, interpolation);

    cv::Mat dst_gold;
    cv::resize(src, dst_gold, cv::Size(), coeff, coeff, interpolation);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() == CV_32F ? 1e-2 : 1.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Warping, ResizeSameAsHost, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4), MatType(CV_16UC1), MatType(CV_16UC3), MatType(CV_16UC4), MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4)),
    testing::Values(0.3, 0.5),
    testing::Values(Interpolation(cv::INTER_NEAREST), Interpolation(cv::INTER_AREA)),
    WHOLE_SUBMAT));

PARAM_TEST_CASE(ResizeTextures, cv::cuda::DeviceInfo, Interpolation)
{
    cv::cuda::DeviceInfo devInfo;
    Interpolation interpolation;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        interpolation = GET_PARAM(1);
        cv::cuda::setDevice(devInfo.deviceID());
    }
};

void ResizeThread(const Interpolation interp, const GpuMat& imgIn, const std::vector<GpuMat>& imgsOut, Stream& stream) {
    for (auto& imgOut : imgsOut)
        cv::cuda::resize(imgIn, imgOut, imgOut.size(), 0, 0, interp, stream);
}

CUDA_TEST_P(ResizeTextures, Accuracy)
{
    constexpr int nThreads = 5;
    constexpr int nIters = 5;
    const Size szIn(100, 100);
    const Size szOut(200, 200);
    vector<Stream> streams(nThreads, cv::cuda::Stream::Null());
    vector<GpuMat> imgsIn;
    vector<vector<GpuMat>> imgsOut;
    for (int i = 0; i < nThreads; i++) {
        imgsIn.push_back(GpuMat(szIn, CV_8UC1, i));
        vector<GpuMat> imgsOutPerThread;
        for (int j = 0; j < nIters; j++)
            imgsOutPerThread.push_back(GpuMat(szOut, CV_8UC1));
        imgsOut.push_back(imgsOutPerThread);
    }

    vector<std::thread> thread(nThreads);
    for (int i = 0; i < nThreads; i++) thread.at(i) = std::thread(ResizeThread, interpolation, std::ref(imgsIn.at(i)), std::ref(imgsOut.at(i)), std::ref(streams.at(i)));
    for (int i = 0; i < nThreads; i++) thread.at(i).join();

    for (int i = 0; i < nThreads; i++) {
        GpuMat imgOutGs;
        cv::cuda::resize(imgsIn.at(i), imgOutGs, szOut, 0, 0, interpolation, streams.at(i));
        Mat imgOutGsHost; imgOutGs.download(imgOutGsHost);
        for (const auto& imgOut : imgsOut.at(i)) {
            Mat imgOutHost; imgOut.download(imgOutHost);
            ASSERT_TRUE(cv::norm(imgOutHost, imgOutGsHost, NORM_INF) == 0);
        }
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Warping, ResizeTextures, testing::Combine(
    ALL_DEVICES,
    testing::Values(Interpolation(cv::INTER_NEAREST), Interpolation(cv::INTER_LINEAR), Interpolation(cv::INTER_CUBIC))));


}} // namespace
#endif // HAVE_CUDA
