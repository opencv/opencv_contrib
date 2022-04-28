// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

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
    const std::vector<cv::cuda::Event::CreateFlags> eventFlags = { cv::cuda::Event::CreateFlags::BLOCKING_SYNC , cv::cuda::Event::CreateFlags::BLOCKING_SYNC | Event::CreateFlags::DISABLE_TIMING };
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
