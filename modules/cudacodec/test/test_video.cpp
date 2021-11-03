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
#include "opencv2/cudaarithm.hpp"
namespace opencv_test {
    namespace {

#if defined(HAVE_NVCUVID) || defined(HAVE_NVCUVENC)
PARAM_TEST_CASE(Video, cv::cuda::DeviceInfo, std::string)
{
};

PARAM_TEST_CASE(VideoReadWrite, cv::cuda::DeviceInfo, std::string)
{
};

#if defined(HAVE_NVCUVID)
//////////////////////////////////////////////////////
// VideoReader

CUDA_TEST_P(Video, Reader)
{
    cv::cuda::setDevice(GET_PARAM(0).deviceID());

    // CUDA demuxer has to fall back to ffmpeg to process "cv/video/768x576.avi"
    if (GET_PARAM(1) == "cv/video/768x576.avi" && !videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend not found");

    std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../" + GET_PARAM(1);
    cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile);
    cv::cudacodec::FormatInfo fmt = reader->format();
    cv::cuda::GpuMat frame;
    for (int i = 0; i < 100; i++)
    {
        ASSERT_TRUE(reader->nextFrame(frame));
        if(!fmt.valid)
            fmt = reader->format();
        ASSERT_TRUE(frame.cols == fmt.width && frame.rows == fmt.height);
        ASSERT_FALSE(frame.empty());
    }
}

CUDA_TEST_P(VideoReadWrite, Reader)
{
    cv::cuda::setDevice(GET_PARAM(0).deviceID());

    // RTSP streaming is only supported by the FFmpeg back end
    if (!videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend not found");

    std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../" + GET_PARAM(1);
    const string fileNameOut = tempfile("test_container_stream");
    {
        cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile, fileNameOut);
        cv::cuda::GpuMat frame;
        for (int i = 0; i < 100; i++)
        {
            reader->writeToFile(fileNameOut.c_str());
            ASSERT_TRUE(reader->nextFrame(frame));
            ASSERT_FALSE(frame.empty());
        }
    }

    std::cout << "Checking written video stream: " << fileNameOut << std::endl;

    {
        cv::Ptr<cv::cudacodec::VideoReader> readerReference = cv::cudacodec::createVideoReader(inputFile);
        cv::Ptr<cv::cudacodec::VideoReader> readerActual = cv::cudacodec::createVideoReader(fileNameOut);
        cv::cuda::GpuMat reference, actual;
        cv::Mat referenceHost, actualHost;
        for (int i = 0; i < 100; i++)
        {
            ASSERT_TRUE(readerReference->nextFrame(reference));
            ASSERT_TRUE(readerActual->nextFrame(actual));
            actual.download(actualHost);
            reference.download(referenceHost);
            ASSERT_TRUE(cvtest::norm(actualHost, referenceHost, NORM_INF) == 0);
        }
    }

    ASSERT_EQ(0, remove(fileNameOut.c_str()));
}
#endif // HAVE_NVCUVID

#if defined(_WIN32) && defined(HAVE_NVCUVENC)
//////////////////////////////////////////////////////
// VideoWriter

CUDA_TEST_P(Video, Writer)
{
    cv::cuda::setDevice(GET_PARAM(0).deviceID());

    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "video/" + GET_PARAM(1);

    std::string outputFile = cv::tempfile(".avi");
    const double FPS = 25.0;

    cv::VideoCapture reader(inputFile);
    ASSERT_TRUE(reader.isOpened());

    cv::Ptr<cv::cudacodec::VideoWriter> d_writer;

    cv::Mat frame;
    cv::cuda::GpuMat d_frame;

    for (int i = 0; i < 10; ++i)
    {
        reader >> frame;
        ASSERT_FALSE(frame.empty());

        d_frame.upload(frame);

        if (d_writer.empty())
            d_writer = cv::cudacodec::createVideoWriter(outputFile, frame.size(), FPS);

        d_writer->write(d_frame);
    }

    reader.release();
    d_writer.release();

    reader.open(outputFile);
    ASSERT_TRUE(reader.isOpened());

    for (int i = 0; i < 5; ++i)
    {
        reader >> frame;
        ASSERT_FALSE(frame.empty());
    }
}

#endif // _WIN32, HAVE_NVCUVENC

#define VIDEO_SRC_R "highgui/video/big_buck_bunny.mp4", "cv/video/768x576.avi", "cv/video/1920x1080.avi", "highgui/video/big_buck_bunny.avi", \
    "highgui/video/big_buck_bunny.h264", "highgui/video/big_buck_bunny.h265", "highgui/video/big_buck_bunny.mpg"
INSTANTIATE_TEST_CASE_P(CUDA_Codec, Video, testing::Combine(
    ALL_DEVICES,
    testing::Values(VIDEO_SRC_R)));

#define VIDEO_SRC_RW "highgui/video/big_buck_bunny.h264", "highgui/video/big_buck_bunny.h265"

INSTANTIATE_TEST_CASE_P(CUDA_Codec, VideoReadWrite, testing::Combine(
    ALL_DEVICES,
    testing::Values(VIDEO_SRC_RW)));

#endif // HAVE_NVCUVID || HAVE_NVCUVENC
}} // namespace
