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
namespace opencv_test {
    namespace {

#if defined(HAVE_NVCUVID) || defined(HAVE_NVCUVENC)
CV_ENUM(ColorFormats, cudacodec::ColorFormat::BGR, cudacodec::ColorFormat::BGRA, cudacodec::ColorFormat::RGB, cudacodec::ColorFormat::RGBA, cudacodec::ColorFormat::GRAY)
CV_ENUM(SurfaceFormats, cudacodec::SurfaceFormat::SF_NV12, cudacodec::SurfaceFormat::SF_P016, cudacodec::SurfaceFormat::SF_YUV444, cudacodec::SurfaceFormat::SF_YUV444_16Bit)
CV_ENUM(BitDepths, cudacodec::BitDepth::UNCHANGED, cudacodec::BitDepth::EIGHT, cudacodec::BitDepth::SIXTEEN)

struct SetDevice : testing::TestWithParam<cv::cuda::DeviceInfo>
{
    cv::cuda::DeviceInfo devInfo;
    virtual void SetUp(){
        devInfo = GetParam();
        cv::cuda::setDevice(devInfo.deviceID());
    }
};

PARAM_TEST_CASE(CheckSet, cv::cuda::DeviceInfo, std::string)
{
};

typedef tuple<std::string, int> check_extra_data_params_t;
PARAM_TEST_CASE(CheckExtraData, cv::cuda::DeviceInfo, check_extra_data_params_t)
{
};

PARAM_TEST_CASE(Scaling, cv::cuda::DeviceInfo, std::string, Size2f, Rect2f, Rect2f)
{
};

struct DisplayResolution : testing::TestWithParam<cv::cuda::DeviceInfo>
{
};

PARAM_TEST_CASE(Video, cv::cuda::DeviceInfo, std::string)
{
};

typedef tuple<std::string, bool> color_conversion_params_t;
PARAM_TEST_CASE(ColorConversionLumaChromaRange, cv::cuda::DeviceInfo, color_conversion_params_t)
{
};

PARAM_TEST_CASE(ColorConversionFormat, cv::cuda::DeviceInfo, ColorFormats)
{
};

struct ColorConversionPlanar : SetDevice
{
};

PARAM_TEST_CASE(ColorConversionBitdepth, cv::cuda::DeviceInfo, BitDepths)
{
};

struct ReconfigureDecoderWithScaling : SetDevice
{
};

PARAM_TEST_CASE(ReconfigureDecoder, cv::cuda::DeviceInfo, int)
{
};

PARAM_TEST_CASE(VideoReadRaw, cv::cuda::DeviceInfo, std::string)
{
};

typedef tuple<std::string, bool> histogram_params_t;
PARAM_TEST_CASE(Histogram, cv::cuda::DeviceInfo, histogram_params_t)
{
};

PARAM_TEST_CASE(CheckKeyFrame, cv::cuda::DeviceInfo, std::string)
{
};

PARAM_TEST_CASE(CheckDecodeSurfaces, cv::cuda::DeviceInfo, std::string)
{
};

PARAM_TEST_CASE(CheckInitParams, cv::cuda::DeviceInfo, std::string, bool, bool, bool)
{
};

struct CheckParams : SetDevice
{
};

struct Seek : SetDevice
{
};

PARAM_TEST_CASE(YuvConverter, cv::cuda::DeviceInfo, SurfaceFormats, ColorFormats, BitDepths, bool, bool)
{
};

#if defined(HAVE_NVCUVID)
//////////////////////////////////////////////////////
// VideoReader

//==========================================================================

CUDA_TEST_P(CheckSet, Reader)
{
    cv::cuda::setDevice(GET_PARAM(0).deviceID());

    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend was not found");

    std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + +"../" + GET_PARAM(1);
    cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile);
    double unsupportedVal = -1;
    ASSERT_FALSE(reader->get(cv::cudacodec::VideoReaderProps::PROP_NOT_SUPPORTED, unsupportedVal));
    double rawModeVal = -1;
    ASSERT_TRUE(reader->get(cv::cudacodec::VideoReaderProps::PROP_RAW_MODE, rawModeVal));
    ASSERT_FALSE(rawModeVal);
    ASSERT_TRUE(reader->set(cv::cudacodec::VideoReaderProps::PROP_RAW_MODE,true));
    ASSERT_TRUE(reader->get(cv::cudacodec::VideoReaderProps::PROP_RAW_MODE, rawModeVal));
    ASSERT_TRUE(rawModeVal);
    bool rawPacketsAvailable = false;
    while (reader->grab()) {
        double nRawPackages = -1;
        ASSERT_TRUE(reader->get(cv::cudacodec::VideoReaderProps::PROP_NUMBER_OF_RAW_PACKAGES_SINCE_LAST_GRAB, nRawPackages));
        if (nRawPackages > 0) {
            rawPacketsAvailable = true;
            break;
        }
    }
    ASSERT_TRUE(rawPacketsAvailable);
}

CUDA_TEST_P(CheckExtraData, Reader)
{
    // RTSP streaming is only supported by the FFmpeg back end
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend not found");

    cv::cuda::setDevice(GET_PARAM(0).deviceID());
    const string path = get<0>(GET_PARAM(1));
    const int sz = get<1>(GET_PARAM(1));
    std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../" + path;
    cv::cudacodec::VideoReaderInitParams params;
    params.rawMode = true;
    cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile, {}, params);
    double extraDataIdx = -1;
    ASSERT_TRUE(reader->get(cv::cudacodec::VideoReaderProps::PROP_EXTRA_DATA_INDEX, extraDataIdx));
    ASSERT_EQ(extraDataIdx, 1 );
    ASSERT_TRUE(reader->grab());
    cv::Mat extraData;
    const bool newData = reader->retrieve(extraData, static_cast<size_t>(extraDataIdx));
    ASSERT_TRUE((newData && sz) || (!newData && !sz));
    ASSERT_EQ(extraData.total(), sz);
}

CUDA_TEST_P(CheckKeyFrame, Reader)
{
    cv::cuda::setDevice(GET_PARAM(0).deviceID());

    // RTSP streaming is only supported by the FFmpeg back end
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend not found");

    const string path = GET_PARAM(1);
    std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../" + path;
    cv::cudacodec::VideoReaderInitParams params;
    params.rawMode = true;
    cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile, {}, params);
    double rawIdxBase = -1;
    ASSERT_TRUE(reader->get(cv::cudacodec::VideoReaderProps::PROP_RAW_PACKAGES_BASE_INDEX, rawIdxBase));
    ASSERT_EQ(rawIdxBase, 2);
    constexpr int maxNPackagesToCheck = 2;
    int nPackages = 0;
    while (nPackages < maxNPackagesToCheck) {
        ASSERT_TRUE(reader->grab());
        double N = -1;
        ASSERT_TRUE(reader->get(cv::cudacodec::VideoReaderProps::PROP_NUMBER_OF_RAW_PACKAGES_SINCE_LAST_GRAB,N));
        for (int i = static_cast<int>(rawIdxBase); i < static_cast<int>(N + rawIdxBase); i++) {
            nPackages++;
            double containsKeyFrame = i;
            ASSERT_TRUE(reader->get(cv::cudacodec::VideoReaderProps::PROP_LRF_HAS_KEY_FRAME, containsKeyFrame));
            ASSERT_TRUE((nPackages == 1 && containsKeyFrame) || (nPackages == 2 && !containsKeyFrame)) << "nPackage: " << i;
            if (nPackages >= maxNPackagesToCheck)
                break;
        }
    }
}

void ForceAlignment(Rect& srcRoi, Rect& targetRoi, Size& targetSz) {
    targetSz.width = targetSz.width - targetSz.width % 2; targetSz.height = targetSz.height - targetSz.height % 2;
    srcRoi.x = srcRoi.x - srcRoi.x % 4; srcRoi.width = srcRoi.width - srcRoi.width % 4;
    srcRoi.y = srcRoi.y - srcRoi.y % 2; srcRoi.height = srcRoi.height - srcRoi.height % 2;
    targetRoi.x = targetRoi.x - targetRoi.x % 4; targetRoi.width = targetRoi.width - targetRoi.width % 4;
    targetRoi.y = targetRoi.y - targetRoi.y % 2; targetRoi.height = targetRoi.height - targetRoi.height % 2;
}

CUDA_TEST_P(Scaling, Reader)
{
    cv::cuda::setDevice(GET_PARAM(0).deviceID());
    std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../" + GET_PARAM(1);
    const Size2f targetSzIn = GET_PARAM(2);
    const Rect2f srcRoiIn = GET_PARAM(3);
    const Rect2f targetRoiIn = GET_PARAM(4);

    GpuMat frameOr;
    {
        cv::Ptr<cv::cudacodec::VideoReader> readerGs = cv::cudacodec::createVideoReader(inputFile);
        ASSERT_TRUE(readerGs->set(cudacodec::ColorFormat::GRAY));
        ASSERT_TRUE(readerGs->nextFrame(frameOr));
    }

    cudacodec::VideoReaderInitParams params;
    params.targetSz = Size(static_cast<int>(frameOr.cols * targetSzIn.width), static_cast<int>(frameOr.rows * targetSzIn.height));
    params.srcRoi = Rect(static_cast<int>(frameOr.cols * srcRoiIn.x), static_cast<int>(frameOr.rows * srcRoiIn.y), static_cast<int>(frameOr.cols * srcRoiIn.width),
        static_cast<int>(frameOr.rows * srcRoiIn.height));
    params.targetRoi = Rect(static_cast<int>(params.targetSz.width * targetRoiIn.x), static_cast<int>(params.targetSz.height * targetRoiIn.y),
        static_cast<int>(params.targetSz.width * targetRoiIn.width), static_cast<int>(params.targetSz.height * targetRoiIn.height));

    cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile, {}, params);
    const cudacodec::FormatInfo format = reader->format();
    ASSERT_TRUE(format.valid);
    ASSERT_TRUE(reader->set(cudacodec::ColorFormat::GRAY));
    GpuMat frame;
    ASSERT_TRUE(reader->nextFrame(frame));
    Size targetSzOut = params.targetSz;
    Rect srcRoiOut = params.srcRoi, targetRoiOut = params.targetRoi;
    ForceAlignment(srcRoiOut, targetRoiOut, targetSzOut);
    ASSERT_TRUE(format.targetSz == targetSzOut && format.srcRoi == srcRoiOut && format.targetRoi == targetRoiOut);
    ASSERT_TRUE(frame.size() == targetSzOut);
    GpuMat frameGs;
    cv::cuda::resize(frameOr(srcRoiOut), frameGs, targetRoiOut.size(), 0, 0, INTER_AREA);
    // assert on mean absolute error due to different resize algorithms
    const double mae = cv::cuda::norm(frameGs, frame(targetRoiOut), NORM_L1)/frameGs.size().area();
    ASSERT_LT(mae, 2.75);
}

CUDA_TEST_P(DisplayResolution, Reader)
{
    cv::cuda::setDevice(GetParam().deviceID());
    std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../cv/video/1920x1080.avi";
    const Rect displayArea(0, 0, 1920, 1080);

    GpuMat frame;
    {
        // verify the output frame is the diplay size (1920x1080) and not the coded size (1920x1088)
        cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile);
        reader->set(cudacodec::ColorFormat::GRAY);
        ASSERT_TRUE(reader->nextFrame(frame));
        const cudacodec::FormatInfo format = reader->format();
        ASSERT_TRUE(format.displayArea == displayArea);
        ASSERT_TRUE(frame.size() == displayArea.size() && frame.size() == format.targetSz);
    }

    {
        // extra check to verify display frame has not been post-processed and is just a cropped version of the coded sized frame
        cudacodec::VideoReaderInitParams params;
        params.srcRoi = Rect(0, 0, 1920, 1088);
        cv::Ptr<cv::cudacodec::VideoReader> readerCodedSz = cv::cudacodec::createVideoReader(inputFile, {}, params);
        readerCodedSz->set(cudacodec::ColorFormat::GRAY);
        GpuMat frameCodedSz;
        ASSERT_TRUE(readerCodedSz->nextFrame(frameCodedSz));
        const cudacodec::FormatInfo formatCodedSz = readerCodedSz->format();
        const double err = cv::cuda::norm(frame, frameCodedSz(displayArea), NORM_INF);
        ASSERT_TRUE(err == 0);
    }
}

CUDA_TEST_P(Video, Reader)
{
    cv::cuda::setDevice(GET_PARAM(0).deviceID());
    const std::string relativeFilePath = GET_PARAM(1);

    // CUDA demuxer has to fall back to ffmpeg to process "cv/video/768x576.avi"
    if (relativeFilePath == "cv/video/768x576.avi" && !videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend not found  - SKIP");

    const std::vector<std::pair< cudacodec::ColorFormat, int>> formatsToChannels = {
        {cudacodec::ColorFormat::GRAY,1},
        {cudacodec::ColorFormat::BGR,3},
        {cudacodec::ColorFormat::BGRA,4},
        {cudacodec::ColorFormat::NV_YUV_SURFACE_FORMAT,1}
    };

    std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../" + relativeFilePath;
    cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile);
    cv::cudacodec::FormatInfo fmt = reader->format();
    cv::cuda::GpuMat frame;
    for (int i = 0; i < 10; i++)
    {
        const std::pair< cudacodec::ColorFormat, int>& formatToChannels = formatsToChannels[i % formatsToChannels.size()];
        ASSERT_TRUE(reader->set(formatToChannels.first));
        double colorFormat;
        ASSERT_TRUE(reader->get(cudacodec::VideoReaderProps::PROP_COLOR_FORMAT, colorFormat) && static_cast<cudacodec::ColorFormat>(colorFormat) == formatToChannels.first);
        ASSERT_TRUE(reader->nextFrame(frame));
        const int height = formatToChannels.first == cudacodec::ColorFormat::NV_YUV_SURFACE_FORMAT ? static_cast<int>(1.5 * fmt.height) : fmt.height;
        ASSERT_TRUE(frame.cols == fmt.width && frame.rows == height);
        ASSERT_FALSE(frame.empty());
        ASSERT_TRUE(frame.channels() == formatToChannels.second);
    }
}

CUDA_TEST_P(ColorConversionLumaChromaRange, Reader)
{
    cv::cuda::setDevice(GET_PARAM(0).deviceID());
    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../" + get<0>(GET_PARAM(1));
    const bool videoFullRangeFlag = get<1>(GET_PARAM(1));
    cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile);
    cv::cudacodec::FormatInfo fmt = reader->format();
    reader->set(cudacodec::ColorFormat::BGR);
    cv::VideoCapture cap(inputFile);

    cv::cuda::GpuMat frame;
    Mat frameHost, frameHostGs, frameFromDevice;
    for (int i = 0; i < 10; i++)
    {
        reader->nextFrame(frame);
        frame.download(frameFromDevice);
        cap.read(frameHost);
        fmt = reader->format();
        ASSERT_TRUE(fmt.videoFullRangeFlag == videoFullRangeFlag);
        frameHostGs = frameHost;
        EXPECT_MAT_NEAR(frameHostGs, frameFromDevice, 2);
    }
}

CUDA_TEST_P(ColorConversionFormat, Reader)
{
    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../highgui/video/big_buck_bunny.h264";
    cv::cuda::setDevice(GET_PARAM(0).deviceID());
    const cudacodec::ColorFormat colorFormat = static_cast<cudacodec::ColorFormat>(static_cast<int>(GET_PARAM(1)));
    cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile);
    double colorFormatGetVal;
    ASSERT_TRUE(reader->get(cudacodec::VideoReaderProps::PROP_COLOR_FORMAT, colorFormatGetVal));
    ASSERT_EQ(cudacodec::ColorFormat::BGRA, static_cast<cudacodec::ColorFormat>(colorFormatGetVal));
    reader->set(colorFormat);
    ASSERT_TRUE(reader->get(cudacodec::VideoReaderProps::PROP_COLOR_FORMAT, colorFormatGetVal));
    ASSERT_EQ(colorFormat, static_cast<cudacodec::ColorFormat>(colorFormatGetVal));
    cv::VideoCapture cap(inputFile);

    int maxDiff = 2;
    cv::cuda::GpuMat frame;
    Mat frameHost, frameHostGs, frameFromDevice, unused;
    for (int i = 0; i < 10; i++)
    {
        reader->nextFrame(frame);
        frame.download(frameFromDevice);
        cap.read(frameHost);
        switch (colorFormat)
        {
        case cudacodec::ColorFormat::BGRA:
            cv::cvtColor(frameHost, frameHostGs, cv::COLOR_BGR2BGRA);
            break;
        case cudacodec::ColorFormat::RGB:
            cv::cvtColor(frameHost, frameHostGs, cv::COLOR_BGR2RGB);
            break;
        case cudacodec::ColorFormat::RGBA:
            cv::cvtColor(frameHost, frameHostGs, cv::COLOR_BGR2RGBA);
            break;
        case cudacodec::ColorFormat::GRAY:
            cv::cvtColor(frameHost, frameHostGs, cv::COLOR_BGR2GRAY);
            // Increased error because of different conversion pipelines. i.e. frameFromDevice (NV12 -> GRAY) and frameHostGs (NV12 -> BGR -> GRAY).  Due to 420 subsampling NV12 -> BGR can increase the luminance of neighbouring pixels if they are significantly different to each other meaning the subsequent conversion BGR -> GRAY will be different to the direct NV12 -> GRAY conversion.
            maxDiff = 15;
            break;
        default:
            frameHostGs = frameHost;
        }
        EXPECT_MAT_NEAR(frameHostGs, frameFromDevice, maxDiff);
    }
}

CUDA_TEST_P(ColorConversionPlanar, Reader)
{
    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../highgui/video/big_buck_bunny.h264";
    cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile);
    double planarGetVal;
    ASSERT_TRUE(reader->get(cudacodec::VideoReaderProps::PROP_PLANAR, planarGetVal));
    ASSERT_FALSE(static_cast<bool>(planarGetVal));
    reader->set(cudacodec::ColorFormat::BGR, cudacodec::BitDepth::UNCHANGED, true);
    ASSERT_TRUE(reader->get(cudacodec::VideoReaderProps::PROP_PLANAR, planarGetVal));
    ASSERT_TRUE(static_cast<bool>(planarGetVal));
    cv::VideoCapture cap(inputFile);

    cv::cuda::GpuMat frame;
    Mat frameHost, frameHostGs, frameFromDevice;
    for (int i = 0; i < 10; i++)
    {
        reader->nextFrame(frame);
        frame.download(frameFromDevice);
        cap.read(frameHost);
        Mat bgrSplit[3];
        cv::split(frameHost, bgrSplit);
        if(i == 0)
            frameHostGs = Mat(frameHost.rows * 3, frameHost.cols, CV_8U);
        bgrSplit[0].copyTo(frameHostGs(Rect(0, 0, frameHost.cols, frameHost.rows)));
        bgrSplit[1].copyTo(frameHostGs(Rect(0, frameHost.rows, frameHost.cols, frameHost.rows)));
        bgrSplit[2].copyTo(frameHostGs(Rect(0, 2 * frameHost.rows, frameHost.cols, frameHost.rows)));
        EXPECT_MAT_NEAR(frameHostGs, frameFromDevice, 2);
    }
}

CUDA_TEST_P(ColorConversionBitdepth, Reader)
{
    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../highgui/video/big_buck_bunny.h264";
    cv::cuda::setDevice(GET_PARAM(0).deviceID());
    const cudacodec::BitDepth bitDepth = static_cast<cudacodec::BitDepth>(static_cast<int>(GET_PARAM(1)));
    cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile);
    double bitDepthGetVal;
    ASSERT_TRUE(reader->get(cudacodec::VideoReaderProps::PROP_BIT_DEPTH, bitDepthGetVal));
    ASSERT_EQ(cudacodec::BitDepth::UNCHANGED, static_cast<cudacodec::BitDepth>(bitDepthGetVal));
    reader->set(cudacodec::ColorFormat::BGR, bitDepth);
    ASSERT_TRUE(reader->get(cudacodec::VideoReaderProps::PROP_BIT_DEPTH, bitDepthGetVal));
    ASSERT_EQ(bitDepth, static_cast<cudacodec::BitDepth>(bitDepthGetVal));
    cv::VideoCapture cap(inputFile);

    int maxDiff = 2;
    cv::cuda::GpuMat frame;
    Mat frameHost, frameHostGs, frameFromDevice;
    for (int i = 0; i < 10; i++)
    {
        reader->nextFrame(frame);
        frame.download(frameFromDevice);
        cap.read(frameHost);
        switch (bitDepth)
        {
        case cudacodec::BitDepth::EIGHT:
        default:
            frameHostGs = frameHost;
            break;
        case cudacodec::BitDepth::SIXTEEN:
            frameHost.convertTo(frameHostGs, CV_16U);
            frameHostGs *= pow(2, 8);
            maxDiff = 512;
        }
        EXPECT_MAT_NEAR(frameHostGs, frameFromDevice, maxDiff);
    }
}

CUDA_TEST_P(ReconfigureDecoderWithScaling, Reader)
{
    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../highgui/video/big_buck_bunny_multi_res.h264";

    GpuMat frameOr;
    {
        cv::Ptr<cv::cudacodec::VideoReader> readerGs = cv::cudacodec::createVideoReader(inputFile);
        ASSERT_TRUE(readerGs->nextFrame(frameOr));
    }

    cv::cudacodec::VideoReaderInitParams params;
    const Size2f targetSzNew(0.8f, 0.9f);
    const Rect2f srcRoiNew(0.25f, 0.25f, 0.5f, 0.5f);
    const Rect2f targetRoiNew(0.2f, 0.3f, 0.6f, 0.7f);
    params.targetSz = Size(static_cast<int>(frameOr.cols * targetSzNew.width), static_cast<int>(frameOr.rows * targetSzNew.height));
    params.srcRoi = Rect(static_cast<int>(frameOr.cols * srcRoiNew.x), static_cast<int>(frameOr.rows * srcRoiNew.y), static_cast<int>(frameOr.cols * srcRoiNew.width),
        static_cast<int>(frameOr.rows * srcRoiNew.height));
    params.targetRoi = Rect(static_cast<int>(params.targetSz.width * targetRoiNew.x), static_cast<int>(params.targetSz.height * targetRoiNew.y),
        static_cast<int>(params.targetSz.width * targetRoiNew.width), static_cast<int>(params.targetSz.height * targetRoiNew.height));

    Size targetSzOut = params.targetSz;
    Rect srcRoiOut = params.srcRoi, targetRoiOut = params.targetRoi;
    ForceAlignment(srcRoiOut, targetRoiOut, targetSzOut);
    GpuMat mask(targetSzOut, CV_8U, Scalar(255));
    mask(targetRoiOut).setTo(0);

    cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile, {}, params);
    reader->set(cudacodec::ColorFormat::GRAY);
    cv::cudacodec::FormatInfo fmt;
    cv::cuda::GpuMat frame;
    int nFrames = 0;
    Size initialSize;
    while (reader->nextFrame(frame))
    {
        ASSERT_TRUE(!frame.empty());
        if (nFrames++ == 0)
            initialSize = frame.size();
        fmt = reader->format();
        ASSERT_TRUE(frame.size() == initialSize);
        ASSERT_TRUE((frame.size() == targetSzOut) && (fmt.targetSz == targetSzOut) && (fmt.srcRoi == srcRoiOut) && (fmt.targetRoi == targetRoiOut));
        // simple check - zero borders, non zero contents
        ASSERT_TRUE(!cuda::absSum(frame, mask)[0] && cuda::sum(frame)[0]);
    }
    ASSERT_TRUE(nFrames == 40);
}

CUDA_TEST_P(ReconfigureDecoder, Reader)
{
    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../highgui/video/big_buck_bunny_multi_res.h264";
    cv::cuda::setDevice(GET_PARAM(0).deviceID());
    const int minNumDecodeSurfaces = GET_PARAM(1);
    cv::cudacodec::VideoReaderInitParams params;
    params.minNumDecodeSurfaces = minNumDecodeSurfaces;
    cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile, {}, params);
    reader->set(cudacodec::ColorFormat::GRAY);
    cv::cudacodec::FormatInfo fmt;
    cv::cuda::GpuMat frame, mask;
    int nFrames = 0;
    Size initialSize, initialCodedSize;
    while(reader->nextFrame(frame))
    {
        ASSERT_TRUE(!frame.empty());
        fmt = reader->format();
        if (nFrames++ == 0) {
            initialSize = frame.size();
            initialCodedSize = Size(fmt.ulWidth, fmt.ulHeight);
        }
        ASSERT_TRUE(frame.size() == initialSize);
        ASSERT_TRUE(fmt.srcRoi.empty());
        const bool resChanged = (initialCodedSize.width != fmt.ulWidth) || (initialCodedSize.height != fmt.ulHeight);
        if (resChanged)
            ASSERT_TRUE(fmt.targetRoi.empty());
    }
    ASSERT_TRUE(nFrames == 40);
}

CUDA_TEST_P(VideoReadRaw, Reader)
{
    cv::cuda::setDevice(GET_PARAM(0).deviceID());

    // RTSP streaming is only supported by the FFmpeg back end
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend not found");

    std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../" + GET_PARAM(1);
    const string fileNameOut = tempfile("test_container_stream");
    {
        std::ofstream file(fileNameOut, std::ios::binary);
        ASSERT_TRUE(file.is_open());
        cv::cudacodec::VideoReaderInitParams params;
        params.rawMode = true;
        cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile, {}, params);
        double rawIdxBase = -1;
        ASSERT_TRUE(reader->get(cv::cudacodec::VideoReaderProps::PROP_RAW_PACKAGES_BASE_INDEX, rawIdxBase));
        ASSERT_EQ(rawIdxBase, 2);
        cv::cuda::GpuMat frame;
        for (int i = 0; i < 100; i++)
        {
            ASSERT_TRUE(reader->grab());
            ASSERT_TRUE(reader->retrieve(frame));
            ASSERT_FALSE(frame.empty());
            double N = -1;
            ASSERT_TRUE(reader->get(cv::cudacodec::VideoReaderProps::PROP_NUMBER_OF_RAW_PACKAGES_SINCE_LAST_GRAB,N));
            ASSERT_TRUE(N >= 0) << N << " < 0";
            for (int j = static_cast<int>(rawIdxBase); j <= static_cast<int>(N + rawIdxBase); j++) {
                Mat rawPackets;
                reader->retrieve(rawPackets, j);
                file.write((char*)rawPackets.data, rawPackets.total());
            }
        }
    }

    std::cout << "Checking written video stream: " << fileNameOut << std::endl;

    {
        cv::Ptr<cv::cudacodec::VideoReader> readerReference = cv::cudacodec::createVideoReader(inputFile);
        cv::cudacodec::VideoReaderInitParams params;
        params.rawMode = true;
        cv::Ptr<cv::cudacodec::VideoReader> readerActual = cv::cudacodec::createVideoReader(fileNameOut, {}, params);
        double decodedFrameIdx = -1;
        ASSERT_TRUE(readerActual->get(cv::cudacodec::VideoReaderProps::PROP_DECODED_FRAME_IDX, decodedFrameIdx));
        ASSERT_EQ(decodedFrameIdx, 0);
        cv::cuda::GpuMat reference, actual;
        cv::Mat referenceHost, actualHost;
        for (int i = 0; i < 100; i++)
        {
            ASSERT_TRUE(readerReference->nextFrame(reference));
            ASSERT_TRUE(readerActual->grab());
            ASSERT_TRUE(readerActual->retrieve(actual, static_cast<size_t>(decodedFrameIdx)));
            actual.download(actualHost);
            reference.download(referenceHost);
            ASSERT_TRUE(cvtest::norm(actualHost, referenceHost, NORM_INF) == 0);
        }
    }

    ASSERT_EQ(0, remove(fileNameOut.c_str()));
}

CUDA_TEST_P(Histogram, Reader)
{
    cuda::setDevice(GET_PARAM(0).deviceID());
    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../" + get<0>(GET_PARAM(1));
    const bool histAvailable = get<1>(GET_PARAM(1));
    cudacodec::VideoReaderInitParams params;
    params.enableHistogram = histAvailable;
    Ptr<cudacodec::VideoReader> reader;
    try {
        reader = cudacodec::createVideoReader(inputFile, {}, params);
    }
    catch (const cv::Exception& e) {
        throw SkipTestException(e.msg);
    }
    const cudacodec::FormatInfo fmt = reader->format();
    ASSERT_EQ(histAvailable, fmt.enableHistogram);
    reader->set(cudacodec::ColorFormat::GRAY);
    GpuMat frame, hist;
    reader->nextFrame(frame, hist);
    if (histAvailable) {
        ASSERT_TRUE(!hist.empty());
        Mat frameHost, histGsHostFloat, histGs, histHost;
        frame.download(frameHost);
        const int histSize = 256;
        const float range[] = { 0, 256 };
        const float* histRange[] = { range };
        cv::calcHist(&frameHost, 1, 0, Mat(), histGsHostFloat, 1, &histSize, histRange);
        histGsHostFloat.convertTo(histGs, CV_32S);
        if (fmt.videoFullRangeFlag)
            hist.download(histHost);
        else
            cudacodec::MapHist(hist, histHost);
        const double err = cv::norm(histGs.t(), histHost, NORM_INF);
        ASSERT_EQ(err, 0);
    }
    else {
        ASSERT_TRUE(hist.empty());
    }
}

CUDA_TEST_P(CheckParams, Reader)
{
    std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../highgui/video/big_buck_bunny.mp4";
    {
        cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile);
        double msActual = -1;
        ASSERT_FALSE(reader->get(cv::VideoCaptureProperties::CAP_PROP_OPEN_TIMEOUT_MSEC, msActual));
    }

    {
        constexpr int msReference = 3333;
        cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile, {
            cv::VideoCaptureProperties::CAP_PROP_OPEN_TIMEOUT_MSEC, msReference });
        double msActual = -1;
        ASSERT_TRUE(reader->get(cv::VideoCaptureProperties::CAP_PROP_OPEN_TIMEOUT_MSEC, msActual));
        ASSERT_EQ(msActual, msReference);
    }
}

CUDA_TEST_P(CheckParams, CaptureProps)
{
    std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../highgui/video/big_buck_bunny.mp4";
    cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile);
    double width, height, fps, iFrame;
    ASSERT_TRUE(reader->get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH, width));
    ASSERT_EQ(672, width);
    ASSERT_TRUE(reader->get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT, height));
    ASSERT_EQ(384, height);
    ASSERT_TRUE(reader->get(cv::VideoCaptureProperties::CAP_PROP_FPS, fps));
    ASSERT_EQ(24, fps);
    ASSERT_TRUE(reader->grab());
    ASSERT_TRUE(reader->get(cv::VideoCaptureProperties::CAP_PROP_POS_FRAMES, iFrame));
    ASSERT_EQ(iFrame, 1.);
}

CUDA_TEST_P(CheckDecodeSurfaces, Reader)
{
    cv::cuda::setDevice(GET_PARAM(0).deviceID());
    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../" + GET_PARAM(1);
    int ulNumDecodeSurfaces = 0;
    {
        cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile);
        cv::cudacodec::FormatInfo fmt = reader->format();
        ulNumDecodeSurfaces = fmt.ulNumDecodeSurfaces;
    }

    {
        cv::cudacodec::VideoReaderInitParams params;
        params.minNumDecodeSurfaces = ulNumDecodeSurfaces - 1;
        cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile, {}, params);
        cv::cudacodec::FormatInfo fmt = reader->format();
        ASSERT_TRUE(fmt.ulNumDecodeSurfaces == ulNumDecodeSurfaces);
        for (int i = 0; i < 100; i++) ASSERT_TRUE(reader->grab());
    }

    {
        cv::cudacodec::VideoReaderInitParams params;
        params.minNumDecodeSurfaces = ulNumDecodeSurfaces + 1;
        cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile, {}, params);
        cv::cudacodec::FormatInfo fmt = reader->format();
        ASSERT_TRUE(fmt.ulNumDecodeSurfaces == ulNumDecodeSurfaces + 1);
        for (int i = 0; i < 100; i++) ASSERT_TRUE(reader->grab());
    }
}

CUDA_TEST_P(CheckInitParams, Reader)
{
    cv::cuda::setDevice(GET_PARAM(0).deviceID());
    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../" + GET_PARAM(1);
    cv::cudacodec::VideoReaderInitParams params;
    params.udpSource = GET_PARAM(2);
    params.allowFrameDrop = GET_PARAM(3);
    params.rawMode = GET_PARAM(4);
    double udpSource = 0, allowFrameDrop = 0, rawMode = 0;
    cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile, {}, params);
    ASSERT_TRUE(reader->get(cv::cudacodec::VideoReaderProps::PROP_UDP_SOURCE, udpSource) && static_cast<bool>(udpSource) == params.udpSource);
    ASSERT_TRUE(reader->get(cv::cudacodec::VideoReaderProps::PROP_ALLOW_FRAME_DROP, allowFrameDrop) && static_cast<bool>(allowFrameDrop) == params.allowFrameDrop);
    ASSERT_TRUE(reader->get(cv::cudacodec::VideoReaderProps::PROP_RAW_MODE, rawMode) && static_cast<bool>(rawMode) == params.rawMode);
}

CUDA_TEST_P(Seek, Reader)
{
    std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../highgui/video/big_buck_bunny.mp4";
    // seek to a non key frame
    const int firstFrameIdx = 18;

    GpuMat frameGs;
    {
        cv::Ptr<cv::cudacodec::VideoReader> readerGs = cv::cudacodec::createVideoReader(inputFile);
        ASSERT_TRUE(readerGs->set(cudacodec::ColorFormat::GRAY));
        for (int i = 0; i <= firstFrameIdx; i++)
            ASSERT_TRUE(readerGs->nextFrame(frameGs));
    }

    cudacodec::VideoReaderInitParams params;
    params.firstFrameIdx = firstFrameIdx;
    cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile, {}, params);
    double iFrame = 0.;
    ASSERT_TRUE(reader->get(cv::VideoCaptureProperties::CAP_PROP_POS_FRAMES, iFrame));
    ASSERT_EQ(iFrame, static_cast<double>(firstFrameIdx));
    ASSERT_TRUE(reader->set(cudacodec::ColorFormat::GRAY));
    GpuMat frame;
    ASSERT_TRUE(reader->nextFrame(frame));
    ASSERT_EQ(cuda::norm(frameGs, frame, NORM_INF), 0.0);
    ASSERT_TRUE(reader->get(cv::VideoCaptureProperties::CAP_PROP_POS_FRAMES, iFrame));
    ASSERT_EQ(iFrame, static_cast<double>(firstFrameIdx+1));
}


void inline GetConstants(float& wr, float& wb, int& black, int& white, int& uvWhite, int& max, bool fullRange = false) {
    if (fullRange) {
        black = 0; white = 255; uvWhite = 255;
    }
    else {
        black = 16; white = 235; uvWhite = 240;
    }
    max = 255;
    wr = 0.2990f; wb = 0.1140f;
}

std::array<std::array<float, 3>, 3> getYuv2RgbMatrix(const bool fullRange = false) {
    float wr, wb;
    int black, white, uvWhite, max;
    GetConstants(wr, wb, black, white, uvWhite, max, fullRange);
    std::array<std::array<float, 3>, 3> mat = { {
        {1.0f, 0.0f, (1.0f - wr) / 0.5f},
        {1.0f, -wb * (1.0f - wb) / 0.5f / (1 - wb - wr), -wr * (1 - wr) / 0.5f / (1 - wb - wr)},
        {1.0f, (1.0f - wb) / 0.5f, 0.0f},
    } };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (j == 0)
                mat[i][j] = (float)(1.0 * max / (white - black) * mat[i][j]);
            else
                mat[i][j] = (float)(1.0 * max / (uvWhite - black) * mat[i][j]);
        }
    }
    return mat;
}

std::array<std::array<float, 3>, 3> getRgb2YuvMatrix(const bool fullRange = false) {
    float wr, wb;
    int black, white, max, uvWhite;
    GetConstants(wr, wb, black, white, uvWhite, max, fullRange);
    std::array<std::array<float, 3>, 3> mat = { {
        {wr, 1.0f - wb - wr, wb},
        {-0.5f * wr / (1.0f - wb), -0.5f * (1 - wb - wr) / (1.0f - wb), 0.5f},
        {0.5f, -0.5f * (1.0f - wb - wr) / (1.0f - wr), -0.5f * wb / (1.0f - wr)},
    } };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (j == 0)
                mat[i][j] = (float)(1.0 * (white - black) / max * mat[i][j]);
            else
                mat[i][j] = (float)(1.0 * (uvWhite - black) / max * mat[i][j]);
        }
    }
    return mat;
}

void generateGray(Mat bgr, Mat& y, Mat& grayFromY, const bool fullRange) {
    Mat yuvI420;
    cv::cvtColor(bgr, yuvI420, COLOR_BGR2YUV_I420);
    yuvI420(Rect(0, 0, bgr.cols, bgr.rows)).copyTo(y);
    if (fullRange) {
        y -= 16;
        y *= 255.0 / 219.0;
    }
    y.copyTo(grayFromY);
    if (!fullRange) {
        grayFromY -= 16;
        grayFromY *= 255.0 / 219.0;
    }
}

void generateNv12(Mat bgr, Mat& nv12Interleaved, Mat& bgrFromYuv, const bool fullRange) {
    Mat yuvI420;
    cv::cvtColor(bgr, yuvI420, COLOR_BGR2YUV_I420);
    cv::cvtColor(yuvI420, bgrFromYuv, COLOR_YUV2BGR_I420);

    Mat uv = yuvI420(Rect(0, bgr.rows, bgr.cols, bgr.rows / 2));
    Mat u0 = uv(Rect(0, 0, uv.cols / 2, uv.rows / 2));
    Mat u1 = uv(Rect(uv.cols / 2, 0, uv.cols / 2, uv.rows / 2));
    Mat v0 = uv(Rect(0, uv.rows / 2, uv.cols / 2, uv.rows / 2));
    Mat v1 = uv(Rect(uv.cols / 2, uv.rows / 2, uv.cols / 2, uv.rows / 2));

    Mat u(uv.rows, uv.cols / 2, CV_8U);
    Mat ur0(u0.rows, u0.cols, CV_8U, u.data, u0.cols * 2);
    Mat ur1(u0.rows, u0.cols, CV_8U, u.data + u0.cols, u0.cols * 2);
    u0.copyTo(ur0);
    u1.copyTo(ur1);

    Mat v(uv.rows, uv.cols / 2, CV_8U);
    Mat vr0(v0.rows, v0.cols, CV_8U, v.data, v0.cols * 2);
    Mat vr1(v0.rows, v0.cols, CV_8U, v.data + v0.cols, v0.cols * 2);
    v0.copyTo(vr0);
    v1.copyTo(vr1);

    Mat uv2Channel;
    Mat uvArray[2] = { u,v };
    cv::merge(uvArray, 2, uv2Channel);

    Mat y = yuvI420(Rect(0, 0, bgr.cols, bgr.rows));
    Mat uvInterleaved(uv2Channel.rows, uv2Channel.cols * 2, CV_8U, uv2Channel.data, uv2Channel.step[0]);

    if (fullRange) {
        Mat y32F;
        y = (y - 16) * 255.0 / 219.0;
        uvInterleaved = (uvInterleaved - 128) * 255.0 / 224.0 + 128;
    }

    nv12Interleaved = Mat(yuvI420.size(), CV_8UC1);
    y.copyTo(nv12Interleaved(Rect(0, 0, bgr.cols, bgr.rows)));
    uvInterleaved.copyTo(nv12Interleaved(Rect(0, bgr.rows, uvInterleaved.cols, uvInterleaved.rows)));
}

void generateYuv444(Mat bgr, Mat& yuv444, Mat& bgrFromYuv, const bool fullRange) {
    std::array<std::array<float, 3>, 3> matrix = getRgb2YuvMatrix(fullRange);
    const int yAdj = fullRange ? 0 : 16, uvAdj = 128;
    Mat bgr32F;
    bgr.convertTo(bgr32F, CV_32F);
    Mat bgrSplit32F[3];
    cv::split(bgr32F, bgrSplit32F);
    Mat yuv32 = Mat(bgr.rows * 3, bgr.cols, CV_32F);
    Mat Y = matrix[0][0] * bgrSplit32F[2] + matrix[0][1] * bgrSplit32F[1] + matrix[0][2] * bgrSplit32F[0] + yAdj;
    Y.copyTo(yuv32(Rect(0, 0, bgr.cols, bgr.rows)));
    Mat U = matrix[1][0] * bgrSplit32F[2] + matrix[1][1] * bgrSplit32F[1] + matrix[1][2] * bgrSplit32F[0] + uvAdj;
    U.copyTo(yuv32(Rect(0, bgr.rows, bgr.cols, bgr.rows)));
    Mat V = matrix[2][0] * bgrSplit32F[2] + matrix[2][1] * bgrSplit32F[1] + matrix[2][2] * bgrSplit32F[0] + uvAdj;
    V.copyTo(yuv32(Rect(0, 2 * bgr.rows, bgr.cols, bgr.rows)));
    yuv32.convertTo(yuv444, CV_8UC1);

    Mat y8 = yuv444(Rect(0, 0, bgr.cols, bgr.rows));
    Mat u8 = yuv444(Rect(0, bgr.rows, bgr.cols, bgr.rows));
    Mat v8 = yuv444(Rect(0, 2 * bgr.rows, bgr.cols, bgr.rows));
    y8.convertTo(Y, CV_32F);
    u8.convertTo(U, CV_32F);
    v8.convertTo(V, CV_32F);

    if (!fullRange) Y -= 16;
    U -= 128;
    V -= 128;
    matrix = getYuv2RgbMatrix(fullRange);
    Mat bgrFromYuvSplit32F[3];
    bgrFromYuvSplit32F[0] = matrix[2][0] * Y + matrix[2][1] * U;
    bgrFromYuvSplit32F[1] = matrix[1][0] * Y + matrix[1][1] * U + matrix[1][2] * V;
    bgrFromYuvSplit32F[2] = matrix[0][0] * Y + matrix[0][2] * V;
    Mat bgrFromYuv32F;
    cv::merge(bgrFromYuvSplit32F, 3, bgrFromYuv32F);
    bgrFromYuv32F.convertTo(bgrFromYuv, CV_8UC3);
}

void generateTestImages(Mat bgrIn, Mat& testImg, Mat& out, const cudacodec::SurfaceFormat inputFormat, const cudacodec::ColorFormat outputFormat, const cudacodec::BitDepth outputBitDepth = cudacodec::BitDepth::EIGHT, bool planar = false, const bool fullRange = false) {
    Mat imgOutFromYuv, imgOut8;
    Mat yuv8;

    switch (inputFormat) {
    case cudacodec::SurfaceFormat::SF_NV12:
    case cudacodec::SurfaceFormat::SF_P016:
        if (outputFormat == cudacodec::ColorFormat::GRAY) {
            yuv8 = Mat(static_cast<int>(bgrIn.rows * 1.5), bgrIn.cols, CV_8U);
            Mat y = yuv8(Rect(0, 0, bgrIn.cols, bgrIn.rows));
            generateGray(bgrIn, y, imgOutFromYuv, fullRange);
        }
        else
            generateNv12(bgrIn, yuv8, imgOutFromYuv, fullRange);
        break;
    case cudacodec::SurfaceFormat::SF_YUV444:
    case cudacodec::SurfaceFormat::SF_YUV444_16Bit:
        if (outputFormat == cudacodec::ColorFormat::GRAY) {
            yuv8 = Mat(bgrIn.rows * 3, bgrIn.cols, CV_8U);
            Mat y = yuv8(Rect(0, 0, bgrIn.cols, bgrIn.rows));
            generateGray(bgrIn, y, imgOutFromYuv, fullRange);
        }
        else
            generateYuv444(bgrIn, yuv8, imgOutFromYuv, fullRange);
        break;
    }

    if (inputFormat == cudacodec::SurfaceFormat::SF_P016 || inputFormat == cudacodec::SurfaceFormat::SF_YUV444_16Bit) {
        yuv8.convertTo(testImg, CV_16U);
        testImg *= pow(2, 8);
    }
    else
        yuv8.copyTo(testImg);

    switch (outputFormat) {
    case cudacodec::ColorFormat::BGR:
        imgOut8 = imgOutFromYuv;
        break;
    case cudacodec::ColorFormat::BGRA: {
        cv::cvtColor(imgOutFromYuv, imgOut8, COLOR_BGR2BGRA);
        break;
    }
    case cudacodec::ColorFormat::RGB: {
        cv::cvtColor(imgOutFromYuv, imgOut8, COLOR_BGR2RGB);
        break;
    }
    case cudacodec::ColorFormat::RGBA: {
        cv::cvtColor(imgOutFromYuv, imgOut8, COLOR_BGR2RGBA);
        break;
    }
    case cudacodec::ColorFormat::GRAY: {
        imgOut8 = imgOutFromYuv;
        break;
    }
    }

    Mat imgOutBitDepthOut;
    if (outputBitDepth == cudacodec::BitDepth::SIXTEEN) {
        imgOut8.convertTo(imgOutBitDepthOut, CV_16U);
        imgOutBitDepthOut *= pow(2, 8);
    }
    else
        imgOutBitDepthOut = imgOut8;

    if (planar && outputFormat != cudacodec::ColorFormat::GRAY) {
        Mat* bgrSplit = new Mat[imgOutBitDepthOut.channels()];
        cv::split(imgOutBitDepthOut, bgrSplit);
        const int type = CV_MAKE_TYPE(CV_MAT_DEPTH(imgOutBitDepthOut.flags), 1);
        out = Mat(imgOutBitDepthOut.rows * imgOutBitDepthOut.channels(), imgOutBitDepthOut.cols, type);
        for (int i = 0; i < imgOut8.channels(); i++)
            bgrSplit[i].copyTo(out(Rect(0, i * imgOut8.rows, imgOut8.cols, imgOut8.rows)));
        delete[] bgrSplit;
    }
    else
        imgOutBitDepthOut.copyTo(out);
}

CUDA_TEST_P(YuvConverter, Reader)
{
    cv::cuda::setDevice(GET_PARAM(0).deviceID());
    const cudacodec::SurfaceFormat surfaceFormat = static_cast<cudacodec::SurfaceFormat>(static_cast<int>(GET_PARAM(1)));
    const cudacodec::ColorFormat outputFormat = static_cast<cudacodec::ColorFormat>(static_cast<int>(GET_PARAM(2)));
    const cudacodec::BitDepth bitDepth = static_cast<cudacodec::BitDepth>(static_cast<int>(GET_PARAM(3)));
    const bool planar = GET_PARAM(4);
    const bool fullRange = GET_PARAM(5);
    std::string imgPath = std::string(cvtest::TS::ptr()->get_data_path()) + "../python/images/baboon.jpg";
    Ptr<cv::cudacodec::NVSurfaceToColorConverter> yuvConverter = cudacodec::createNVSurfaceToColorConverter(cv::cudacodec::ColorSpaceStandard::BT601, fullRange);
    Mat bgr = imread(imgPath), bgrHost;
    Mat nv12Interleaved, bgrFromYuv;
    generateTestImages(bgr, nv12Interleaved, bgrFromYuv, surfaceFormat, outputFormat, bitDepth, planar, fullRange);
    GpuMat nv12Device(nv12Interleaved), bgrDevice(bgrFromYuv.size(), bgrFromYuv.type());
    yuvConverter->convert(nv12Device, bgrDevice, surfaceFormat, outputFormat, bitDepth, planar, fullRange);
    bgrDevice.download(bgrHost);
    EXPECT_MAT_NEAR(bgrFromYuv, bgrHost, bitDepth == cudacodec::BitDepth::EIGHT ? 2 :512);
}

#endif // HAVE_NVCUVID

#if defined(HAVE_NVCUVID) && defined(HAVE_NVCUVENC)

struct H264ToH265 : SetDevice
{
};

CUDA_TEST_P(H264ToH265, Transcode)
{
    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../highgui/video/big_buck_bunny.h264";
    constexpr cv::cudacodec::ColorFormat colorFormat = cv::cudacodec::ColorFormat::NV_NV12;
    constexpr double fps = 25;
    const cudacodec::Codec codec = cudacodec::Codec::HEVC;
    const std::string ext = ".mp4";
    const std::string outputFile = cv::tempfile(ext.c_str());
    constexpr int nFrames = 5;
    Size frameSz;
    {
        cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile);
        cv::cudacodec::FormatInfo fmt = reader->format();
        reader->set(cudacodec::ColorFormat::NV_YUV_SURFACE_FORMAT);
        cv::Ptr<cv::cudacodec::VideoWriter> writer;
        cv::cuda::GpuMat frame;
        cv::cuda::Stream stream;
        for (int i = 0; i < nFrames; ++i) {
            ASSERT_TRUE(reader->nextFrame(frame, stream));
            ASSERT_FALSE(frame.empty());
            if (writer.empty()) {
                frameSz = Size(fmt.width, fmt.height);
                writer = cv::cudacodec::createVideoWriter(outputFile, frameSz, codec, fps, colorFormat, 0, stream);
            }
            writer->write(frame);
        }
    }

    {
        cv::VideoCapture cap(outputFile);
        ASSERT_TRUE(cap.isOpened());
        const int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
        const int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
        ASSERT_EQ(frameSz, Size(width, height));
        ASSERT_EQ(fps, cap.get(CAP_PROP_FPS));
        Mat frame;
        for (int i = 0; i < nFrames; ++i) {
            cap >> frame;
            ASSERT_FALSE(frame.empty());
            const int pts = static_cast<int>(cap.get(CAP_PROP_PTS));
            ASSERT_EQ(i, pts > 0 ? pts : 0); // FFmpeg back end returns dts if pts is zero.
        }
    }
    ASSERT_EQ(0, remove(outputFile.c_str()));
}

INSTANTIATE_TEST_CASE_P(CUDA_Codec, H264ToH265, ALL_DEVICES);

CV_ENUM(YuvColorFormats, cudacodec::ColorFormat::NV_YUV444, cudacodec::ColorFormat::NV_YUV420_10BIT, cudacodec::ColorFormat::NV_YUV444_10BIT)
PARAM_TEST_CASE(YUVFormats, cv::cuda::DeviceInfo, YuvColorFormats, bool)
{
};

CUDA_TEST_P(YUVFormats, Transcode)
{
    cv::cuda::setDevice(GET_PARAM(0).deviceID());
    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../highgui/video/big_buck_bunny.h265";
    const cv::cudacodec::ColorFormat writerColorFormat = static_cast<cudacodec::ColorFormat>(static_cast<int>(GET_PARAM(1)));
    const bool fullRange = GET_PARAM(2);
    constexpr double fps = 25;
    const cudacodec::Codec codec = cudacodec::Codec::HEVC;
    const std::string ext = ".mp4";
    const std::string outputFile = cv::tempfile(ext.c_str());
    constexpr int nFrames = 5;
    vector<Mat> bgrGs;
    {
        VideoCapture cap(inputFile);
        cv::Ptr<cv::cudacodec::VideoWriter> writer;
        Mat frame, yuv, bgr;
        cv::cudacodec::EncoderParams params;
        params.tuningInfo = cv::cudacodec::EncodeTuningInfo::ENC_TUNING_INFO_LOSSLESS;
        params.rateControlMode = cv::cudacodec::EncodeParamsRcMode::ENC_PARAMS_RC_CONSTQP;
        params.videoFullRangeFlag = fullRange;
        for (int i = 0; i < nFrames; ++i) {
            ASSERT_TRUE(cap.read(frame));
            ASSERT_FALSE(frame.empty());
            cudacodec::SurfaceFormat yuvFormat = cudacodec::SurfaceFormat::SF_YUV444;
            cudacodec::BitDepth bitDepth = cudacodec::BitDepth::EIGHT;
            if (writerColorFormat == cudacodec::ColorFormat::NV_YUV444_10BIT) {
                yuvFormat = cudacodec::SurfaceFormat::SF_YUV444_16Bit;
                bitDepth = cudacodec::BitDepth::SIXTEEN;
            }
            else if (writerColorFormat == cudacodec::ColorFormat::NV_YUV420_10BIT){
                yuvFormat = cudacodec::SurfaceFormat::SF_P016;
                bitDepth = cudacodec::BitDepth::SIXTEEN;
            }
            generateTestImages(frame, yuv, bgr, yuvFormat, cudacodec::ColorFormat::BGR, bitDepth, false, fullRange);
            bgrGs.push_back(bgr.clone());
            if (writer.empty())
                writer = cv::cudacodec::createVideoWriter(outputFile, frame.size(), codec, fps, writerColorFormat, params);
            writer->write(yuv);
        }
    }

    {
        cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(outputFile);
        reader->set(cudacodec::ColorFormat::BGR);
        cv::cuda::GpuMat frame, frameGs;
        Mat frameHost, frameGsHost;
        for (int i = 0; i < nFrames; ++i) {
            ASSERT_TRUE(reader->nextFrame(frame));
            frame.download(frameHost);
            frameGsHost = bgrGs[i];
            const int diff = writerColorFormat == cudacodec::ColorFormat::NV_YUV420_10BIT || writerColorFormat == cudacodec::ColorFormat::NV_YUV444_10BIT ? 512 : 1;
            EXPECT_MAT_NEAR(frameHost, frameGsHost, diff);
        }
    }
    ASSERT_EQ(0, remove(outputFile.c_str()));
}

INSTANTIATE_TEST_CASE_P(CUDA_Codec, YUVFormats, testing::Combine(ALL_DEVICES, YuvColorFormats::all(), testing::Bool()));
#endif

#if defined(HAVE_NVCUVENC)

//////////////////////////////////////////////////////
// VideoWriter

//==========================================================================

void CvtColor(const Mat& in, Mat& out, const cudacodec::ColorFormat surfaceFormatCv) {
    switch (surfaceFormatCv) {
    case(cudacodec::ColorFormat::RGB):
        return cv::cvtColor(in, out, COLOR_BGR2RGB);
    case(cudacodec::ColorFormat::BGRA):
        return cv::cvtColor(in, out, COLOR_BGR2BGRA);
    case(cudacodec::ColorFormat::RGBA):
        return cv::cvtColor(in, out, COLOR_BGR2RGBA);
    case(cudacodec::ColorFormat::GRAY):
        return cv::cvtColor(in, out, COLOR_BGR2GRAY);
    default:
        in.copyTo(out);
    }
}

PARAM_TEST_CASE(Write, cv::cuda::DeviceInfo, bool, cv::cudacodec::Codec, double, cv::cudacodec::ColorFormat)
{
};

CUDA_TEST_P(Write, Writer)
{
    cv::cuda::setDevice(GET_PARAM(0).deviceID());
    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../highgui/video/big_buck_bunny.mp4";
    const bool deviceSrc = GET_PARAM(1);
    const cudacodec::Codec codec = GET_PARAM(2);
    const double fps = GET_PARAM(3);
    const cv::cudacodec::ColorFormat colorFormat = GET_PARAM(4);
    const std::string ext = ".mp4";
    const std::string outputFile = cv::tempfile(ext.c_str());
    constexpr int nFrames = 5;
    Size frameSz;
    {
        cv::VideoCapture cap(inputFile);
        ASSERT_TRUE(cap.isOpened());
        cv::Ptr<cv::cudacodec::VideoWriter> writer;
        cv::Mat frame, frameNewSf;
        cv::cuda::GpuMat dFrame;
        cv::cuda::Stream stream;
        for (int i = 0; i < nFrames; ++i) {
            cap >> frame;
            ASSERT_FALSE(frame.empty());
            if (writer.empty()) {
                frameSz = frame.size();
                writer = cv::cudacodec::createVideoWriter(outputFile, frameSz, codec, fps, colorFormat, 0, stream);
            }
            CvtColor(frame, frameNewSf, colorFormat);
            if (deviceSrc) {
                dFrame.upload(frameNewSf);
                writer->write(dFrame);
            }
            else
                writer->write(frameNewSf);
        }
    }

    {
        cv::VideoCapture cap(outputFile);
        ASSERT_TRUE(cap.isOpened());
        const int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
        const int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
        ASSERT_EQ(frameSz, Size(width, height));
        ASSERT_EQ(fps, cap.get(CAP_PROP_FPS));
        Mat frame;
        for (int i = 0; i < nFrames; ++i) {
            cap >> frame;
            ASSERT_FALSE(frame.empty());
            const int pts = static_cast<int>(cap.get(CAP_PROP_PTS));
            ASSERT_EQ(i, pts > 0 ? pts : 0); // FFmpeg back end returns dts if pts is zero.
        }
    }
    ASSERT_EQ(0, remove(outputFile.c_str()));
}

#define DEVICE_SRC true, false
#define FPS 10, 29
#define CODEC cv::cudacodec::Codec::H264, cv::cudacodec::Codec::HEVC
#define COLOR_FORMAT cv::cudacodec::ColorFormat::BGR, cv::cudacodec::ColorFormat::RGB, cv::cudacodec::ColorFormat::BGRA, \
cv::cudacodec::ColorFormat::RGBA, cv::cudacodec::ColorFormat::GRAY
INSTANTIATE_TEST_CASE_P(CUDA_Codec, Write, testing::Combine(ALL_DEVICES, testing::Values(DEVICE_SRC), testing::Values(CODEC), testing::Values(FPS),
    testing::Values(COLOR_FORMAT)));

PARAM_TEST_CASE(EncoderParams, cv::cuda::DeviceInfo, int)
{
    cv::cuda::DeviceInfo devInfo;
    cv::cudacodec::EncoderParams params;
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        cv::cuda::setDevice(devInfo.deviceID());
        // Fixed params for CBR test
        params.tuningInfo = cv::cudacodec::EncodeTuningInfo::ENC_TUNING_INFO_HIGH_QUALITY;
        params.encodingProfile = cv::cudacodec::EncodeProfile::ENC_H264_PROFILE_MAIN;
        params.rateControlMode = cv::cudacodec::EncodeParamsRcMode::ENC_PARAMS_RC_CBR;
        params.multiPassEncoding = cv::cudacodec::EncodeMultiPass::ENC_TWO_PASS_FULL_RESOLUTION;
        params.averageBitRate = 1000000;
        params.maxBitRate = 0;
        params.targetQuality = 0;
        params.gopLength = 5;
        params.idrPeriod = GET_PARAM(1);
    }
};

CUDA_TEST_P(EncoderParams, Writer)
{
    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../highgui/video/big_buck_bunny.mp4";
    constexpr double fps = 25.0;
    constexpr cudacodec::Codec codec = cudacodec::Codec::H264;
    const std::string ext = ".mp4";
    const std::string outputFile = cv::tempfile(ext.c_str());
    Size frameSz;
    const int nFrames = max(params.gopLength, params.idrPeriod) + 1;
    {
        cv::VideoCapture reader(inputFile);
        ASSERT_TRUE(reader.isOpened());
        const cv::cudacodec::ColorFormat colorFormat = cv::cudacodec::ColorFormat::BGR;
        cv::Ptr<cv::cudacodec::VideoWriter> writer;
        cv::Mat frame;
        cv::cuda::GpuMat dFrame;
        cv::cuda::Stream stream;
        for (int i = 0; i < nFrames; ++i) {
            reader >> frame;
            ASSERT_FALSE(frame.empty());
            dFrame.upload(frame);
            if (writer.empty()) {
                frameSz = frame.size();
                writer = cv::cudacodec::createVideoWriter(outputFile, frameSz, codec, fps, colorFormat, params, 0, stream);
                cv::cudacodec::EncoderParams paramsOut = writer->getEncoderParams();
                ASSERT_EQ(params, paramsOut);
            }
            writer->write(dFrame);
        }
    }

    {
        cv::VideoCapture cap(outputFile);
        ASSERT_TRUE(cap.isOpened());
        const int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
        const int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
        ASSERT_EQ(frameSz, Size(width, height));
        ASSERT_EQ(fps, cap.get(CAP_PROP_FPS));
        const bool checkFrameType = videoio_registry::hasBackend(CAP_FFMPEG);
        VideoCapture capRaw;
        int idrPeriod = 0;
        if (checkFrameType) {
            capRaw.open(outputFile, CAP_FFMPEG, { CAP_PROP_FORMAT, -1 });
            ASSERT_TRUE(capRaw.isOpened());
            idrPeriod = params.idrPeriod == 0 ? params.gopLength : params.idrPeriod;
        }
        const double frameTypeIAsciiCode = 73.0; // see CAP_PROP_FRAME_TYPE
        Mat frame, frameRaw;
        for (int i = 0; i < nFrames; ++i) {
            cap >> frame;
            ASSERT_FALSE(frame.empty());
            if (checkFrameType) {
                capRaw >> frameRaw;
                ASSERT_FALSE(frameRaw.empty());
                const bool intraFrameReference = cap.get(CAP_PROP_FRAME_TYPE) == frameTypeIAsciiCode;
                const bool intraFrameActual = i % params.gopLength == 0;
                ASSERT_EQ(intraFrameActual, intraFrameReference);
                const bool keyFrameActual = capRaw.get(CAP_PROP_LRF_HAS_KEY_FRAME) == 1.0;
                const bool keyFrameReference = i % idrPeriod == 0;
                ASSERT_EQ(keyFrameActual, keyFrameReference);
                const int pts = static_cast<int>(cap.get(CAP_PROP_PTS));
                ASSERT_EQ(i, pts > 0 ? pts : 0); // FFmpeg back end returns dts if pts is zero.
            }
        }
    }
    ASSERT_EQ(0, remove(outputFile.c_str()));
}

#define IDR_PERIOD testing::Values(5,10)
INSTANTIATE_TEST_CASE_P(CUDA_Codec, EncoderParams, testing::Combine(ALL_DEVICES, IDR_PERIOD));

#endif // HAVE_NVCUVENC

INSTANTIATE_TEST_CASE_P(CUDA_Codec, CheckSet, testing::Combine(
    ALL_DEVICES,
    testing::Values("highgui/video/big_buck_bunny.mp4")));

#define VIDEO_SRC_SCALING "highgui/video/big_buck_bunny.mp4"
#define TARGET_SZ Size2f(1,1), Size2f(0.8f,0.9f), Size2f(2.3f,1.8f)
#define SRC_ROI Rect2f(0,0,1,1), Rect2f(0.25f,0.25f,0.5f,0.5f)
#define TARGET_ROI Rect2f(0,0,1,1), Rect2f(0.2f,0.3f,0.6f,0.7f)
INSTANTIATE_TEST_CASE_P(CUDA_Codec, Scaling, testing::Combine(
    ALL_DEVICES, testing::Values(VIDEO_SRC_SCALING), testing::Values(TARGET_SZ), testing::Values(SRC_ROI), testing::Values(TARGET_ROI)));

INSTANTIATE_TEST_CASE_P(CUDA_Codec, DisplayResolution, ALL_DEVICES);

#define VIDEO_SRC_R testing::Values("highgui/video/big_buck_bunny.mp4", "cv/video/768x576.avi", "cv/video/1920x1080.avi", "highgui/video/big_buck_bunny.avi", \
    "highgui/video/big_buck_bunny.h264", "highgui/video/big_buck_bunny.h265", "highgui/video/big_buck_bunny.mpg", \
    "highgui/video/sample_322x242_15frames.yuv420p.libvpx-vp9.mp4")
    //, "highgui/video/sample_322x242_15frames.yuv420p.libaom-av1.mp4", \
    "cv/tracking/faceocc2/data/faceocc2.webm", "highgui/video/sample_322x242_15frames.yuv420p.mpeg2video.mp4", "highgui/video/sample_322x242_15frames.yuv420p.mjpeg.mp4")

INSTANTIATE_TEST_CASE_P(CUDA_Codec, Video, testing::Combine(ALL_DEVICES,VIDEO_SRC_R));

const color_conversion_params_t color_conversion_params[] =
{
    color_conversion_params_t("highgui/video/big_buck_bunny.h264", false),
    color_conversion_params_t("highgui/video/big_buck_bunny_full_color_range.h264", true),
};

INSTANTIATE_TEST_CASE_P(CUDA_Codec, ColorConversionLumaChromaRange, testing::Combine(
    ALL_DEVICES,
    testing::ValuesIn(color_conversion_params)));

INSTANTIATE_TEST_CASE_P(CUDA_Codec, ColorConversionFormat, testing::Combine(ALL_DEVICES, ColorFormats::all()));

INSTANTIATE_TEST_CASE_P(CUDA_Codec, ColorConversionPlanar, ALL_DEVICES);

INSTANTIATE_TEST_CASE_P(CUDA_Codec, ColorConversionBitdepth, testing::Combine(ALL_DEVICES, BitDepths::all()));

INSTANTIATE_TEST_CASE_P(CUDA_Codec, ReconfigureDecoderWithScaling, ALL_DEVICES);

#define N_DECODE_SURFACES testing::Values(0, 10)
INSTANTIATE_TEST_CASE_P(CUDA_Codec, ReconfigureDecoder, testing::Combine(ALL_DEVICES, N_DECODE_SURFACES));

#define VIDEO_SRC_RW "highgui/video/big_buck_bunny.h264", "highgui/video/big_buck_bunny.h265"
INSTANTIATE_TEST_CASE_P(CUDA_Codec, VideoReadRaw, testing::Combine(
    ALL_DEVICES,
    testing::Values(VIDEO_SRC_RW)));

const histogram_params_t histogram_params[] =
{
    histogram_params_t("highgui/video/big_buck_bunny.mp4", false),
    histogram_params_t("highgui/video/big_buck_bunny.h264", false),
    histogram_params_t("highgui/video/big_buck_bunny_full_color_range.h264", true),
};

INSTANTIATE_TEST_CASE_P(CUDA_Codec, Histogram, testing::Combine(ALL_DEVICES,testing::ValuesIn(histogram_params)));

const check_extra_data_params_t check_extra_data_params[] =
{
    check_extra_data_params_t("highgui/video/big_buck_bunny.mp4", 45),
    check_extra_data_params_t("highgui/video/big_buck_bunny.mov", 45),
    check_extra_data_params_t("highgui/video/big_buck_bunny.mjpg.avi", 0)
};

INSTANTIATE_TEST_CASE_P(CUDA_Codec, CheckExtraData, testing::Combine(
    ALL_DEVICES,
    testing::ValuesIn(check_extra_data_params)));

#define VIDEO_SRC_KEY "highgui/video/big_buck_bunny.mp4", "cv/video/768x576.avi", "cv/video/1920x1080.avi", "highgui/video/big_buck_bunny.avi", \
    "highgui/video/big_buck_bunny.h264", "highgui/video/big_buck_bunny.h265", "highgui/video/big_buck_bunny.mpg"
INSTANTIATE_TEST_CASE_P(CUDA_Codec, CheckKeyFrame, testing::Combine(
    ALL_DEVICES,
    testing::Values(VIDEO_SRC_KEY)));

INSTANTIATE_TEST_CASE_P(CUDA_Codec, CheckParams, ALL_DEVICES);

INSTANTIATE_TEST_CASE_P(CUDA_Codec, CheckDecodeSurfaces, testing::Combine(
    ALL_DEVICES,
    testing::Values("highgui/video/big_buck_bunny.mp4")));

INSTANTIATE_TEST_CASE_P(CUDA_Codec, CheckInitParams, testing::Combine(
    ALL_DEVICES,
    testing::Values("highgui/video/big_buck_bunny.mp4"),
    testing::Values(true,false), testing::Values(true,false), testing::Values(true,false)));

INSTANTIATE_TEST_CASE_P(CUDA_Codec, Seek, ALL_DEVICES);

#define BIT_DEPTHS testing::Values(BitDepths(cudacodec::BitDepth::EIGHT), BitDepths(cudacodec::BitDepth::SIXTEEN))
INSTANTIATE_TEST_CASE_P(CUDA_Codec, YuvConverter, testing::Combine(
    ALL_DEVICES, SurfaceFormats::all(), ColorFormats::all(), BIT_DEPTHS, testing::Bool(), testing::Bool()));

#endif // HAVE_NVCUVID || HAVE_NVCUVENC
}} // namespace
