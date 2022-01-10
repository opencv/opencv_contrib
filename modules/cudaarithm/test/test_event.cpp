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

#include <cuda_runtime.h>

#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/ts/cuda_test.hpp"

namespace opencv_test { namespace {

struct AsyncEvent : testing::TestWithParam<cv::cuda::DeviceInfo>
{
    cv::cuda::HostMem src;
    cv::cuda::GpuMat d_src;

    cv::cuda::HostMem dst;
    cv::cuda::GpuMat d_dst;

    cv::cuda::Stream stream;

    virtual void SetUp()
    {
        cv::cuda::DeviceInfo devInfo = GetParam();
        cv::cuda::setDevice(devInfo.deviceID());

        src = cv::cuda::HostMem(cv::cuda::HostMem::PAGE_LOCKED);

        cv::Mat m = randomMat(cv::Size(128, 128), CV_8UC1);
        m.copyTo(src);
    }
};

void deviceWork(void* userData)
{
    AsyncEvent* test = reinterpret_cast<AsyncEvent*>(userData);
    test->d_src.upload(test->src, test->stream);
    test->d_src.convertTo(test->d_dst, CV_32S, test->stream);
    test->d_dst.download(test->dst, test->stream);
}

CUDA_TEST_P(AsyncEvent, WrapEvent)
{
    cudaEvent_t cuda_event = NULL;
    ASSERT_EQ(cudaSuccess, cudaEventCreate(&cuda_event));
    {
        cv::cuda::Event cudaEvent = cv::cuda::EventAccessor::wrapEvent(cuda_event);
        deviceWork(this);
        cudaEvent.record(stream);
        cudaEvent.waitForCompletion();
        cv::Mat dst_gold;
        src.createMatHeader().convertTo(dst_gold, CV_32S);
        ASSERT_MAT_NEAR(dst_gold, dst, 0);
    }
    ASSERT_EQ(cudaSuccess, cudaEventDestroy(cuda_event));
}

CUDA_TEST_P(AsyncEvent, WithFlags)
{
    cv::cuda::Event cudaEvent = cv::cuda::Event(cv::cuda::Event::CreateFlags::BLOCKING_SYNC);
    deviceWork(this);
    cudaEvent.record(stream);
    cudaEvent.waitForCompletion();
    cv::Mat dst_gold;
    src.createMatHeader().convertTo(dst_gold, CV_32S);
    ASSERT_MAT_NEAR(dst_gold, dst, 0);
}

CUDA_TEST_P(AsyncEvent, Timing)
{
    const std::vector<unsigned> eventFlags = { cv::cuda::Event::CreateFlags::BLOCKING_SYNC , cv::cuda::Event::CreateFlags::BLOCKING_SYNC | Event::CreateFlags::DISABLE_TIMING };
    const std::vector<bool> shouldFail = { false, true };
    for (size_t i = 0; i < eventFlags.size(); i++) {
        const auto& flags = eventFlags.at(i);
        cv::cuda::Event startEvent = cv::cuda::Event(flags);
        cv::cuda::Event stopEvent = cv::cuda::Event(flags);
        startEvent.record(stream);
        deviceWork(this);
        stopEvent.record(stream);
        stopEvent.waitForCompletion();
        cv::Mat dst_gold;
        src.createMatHeader().convertTo(dst_gold, CV_32S);
        ASSERT_MAT_NEAR(dst_gold, dst, 0);
        bool failed = false;
        try {
            const double elTimeMs = Event::elapsedTime(startEvent, stopEvent);
            ASSERT_GT(elTimeMs, 0);
        }
        catch (cv::Exception ex) {
            failed = true;
        }
        ASSERT_EQ(failed, shouldFail.at(i));
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_Event, AsyncEvent, ALL_DEVICES);

}} // namespace
#endif // HAVE_CUDA
