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

#include "perf_precomp.hpp"
#include "opencv2/videoio.hpp"

namespace opencv_test { namespace {

#if defined(HAVE_NVCUVID) || defined(HAVE_NVCUVENC)

#if defined(HAVE_FFMPEG_WRAPPER) // should this be set in preprocessor or in cvconfig.h
#define VIDEO_SRC Values("cv/video/768x576.avi", "cv/video/1920x1080.avi")
#else
// CUDA demuxer has to fall back to ffmpeg to process "cv/video/768x576.avi"
#define VIDEO_SRC Values( "cv/video/1920x1080.avi")
#endif

#if defined(HAVE_NVCUVID)

DEF_PARAM_TEST_1(FileName, string);

//////////////////////////////////////////////////////
// VideoReader

PERF_TEST_P(FileName, VideoReader, VIDEO_SRC)
{
    declare.time(20);

    const string inputFile = perf::TestBase::getDataPath(GetParam());

    if (PERF_RUN_CUDA())
    {
        cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(inputFile);

        cv::cuda::GpuMat frame;

        TEST_CYCLE_N(10) d_reader->nextFrame(frame);

        CUDA_SANITY_CHECK(frame);
    }
    else
    {
        cv::VideoCapture reader(inputFile);
        ASSERT_TRUE( reader.isOpened() );

        cv::Mat frame;

        TEST_CYCLE_N(10) reader >> frame;

        CPU_SANITY_CHECK(frame);
    }
}

#endif

//////////////////////////////////////////////////////
// VideoWriter

#if defined(HAVE_NVCUVENC)

DEF_PARAM_TEST(WriteToFile, string, cv::cudacodec::ColorFormat, cv::cudacodec::Codec);

#define COLOR_FORMAT Values(cv::cudacodec::ColorFormat::BGR, cv::cudacodec::ColorFormat::RGB, cv::cudacodec::ColorFormat::BGRA, \
cv::cudacodec::ColorFormat::RGBA, cv::cudacodec::ColorFormat::GRAY)
#define CODEC Values(cv::cudacodec::Codec::H264, cv::cudacodec::Codec::HEVC)

PERF_TEST_P(WriteToFile, VideoWriter, Combine(VIDEO_SRC, COLOR_FORMAT, CODEC))
{
    declare.time(30);
    const string inputFile = perf::TestBase::getDataPath(GET_PARAM(0));
    const cv::cudacodec::ColorFormat surfaceFormat = GET_PARAM(1);
    const cudacodec::Codec codec = GET_PARAM(2);
    const double fps = 25;
    const int nFrames = 20;
    cv::VideoCapture reader(inputFile);
    ASSERT_TRUE(reader.isOpened());
    Mat frameBgr;
    if (PERF_RUN_CUDA()) {
        const std::string ext = codec == cudacodec::Codec::H264 ? ".h264" : ".hevc";
        const string outputFile = cv::tempfile(ext.c_str());
        std::vector<GpuMat> frames;
        cv::Mat frameNewSf;
        cuda::Stream stream;
        ColorConversionCodes conversionCode = COLOR_COLORCVT_MAX;
        switch (surfaceFormat) {
        case cudacodec::ColorFormat::RGB:
            conversionCode = COLOR_BGR2RGB;
            break;
        case cudacodec::ColorFormat::BGRA:
            conversionCode = COLOR_BGR2BGRA;
            break;
        case cudacodec::ColorFormat::RGBA:
            conversionCode = COLOR_BGR2RGBA;
            break;
        case cudacodec::ColorFormat::GRAY:
            conversionCode = COLOR_BGR2GRAY;
        default:
            break;
        }
        for (int i = 0; i < nFrames; i++) {
            reader >> frameBgr;
            ASSERT_FALSE(frameBgr.empty());
            if (conversionCode == COLOR_COLORCVT_MAX)
                frameNewSf = frameBgr;
            else
                cv::cvtColor(frameBgr, frameNewSf, conversionCode);
            GpuMat frame; frame.upload(frameNewSf, stream);
            frames.push_back(frame);
        }
        stream.waitForCompletion();
        cv::Ptr<cv::cudacodec::VideoWriter> d_writer = cv::cudacodec::createVideoWriter(outputFile, frameBgr.size(), codec, fps, surfaceFormat, 0, stream);
        for (int i = 0; i < nFrames - 1; ++i) {
            startTimer();
            d_writer->write(frames[i]);
            stopTimer();
        }
        startTimer();
        d_writer->write(frames[nFrames - 1]);
        d_writer->release();
        stopTimer();

        ASSERT_EQ(0, remove(outputFile.c_str()));
    }
    else {
        if (surfaceFormat != cv::cudacodec::ColorFormat::BGR || codec != cv::cudacodec::Codec::H264)
            throw PerfSkipTestException();
        cv::VideoWriter writer;
        const string outputFile = cv::tempfile(".avi");
        for (int i = 0; i < nFrames-1; ++i) {
            reader >> frameBgr;
            ASSERT_FALSE(frameBgr.empty());
            if (!writer.isOpened())
                writer.open(outputFile, VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, frameBgr.size());
            startTimer();
            writer.write(frameBgr);
            stopTimer();
        }
        reader >> frameBgr;
        ASSERT_FALSE(frameBgr.empty());
        startTimer();
        writer.write(frameBgr);
        writer.release();
        stopTimer();

        ASSERT_EQ(0, remove(outputFile.c_str()));
    }
    SANITY_CHECK(frameBgr);
}

#endif
#endif
}} // namespace
