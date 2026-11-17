// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Public GpuMat resize tests against the portable CPU reference, including
// copy and all four interpolation modes, CV_8U/CV_32F, C1/C3/C4, pitched ROIs,
// output reuse and guard bands. The linked cv::resize may select different
// optimized paths; cv::cuda::resize also has different numerical behavior.
#include "test_helpers.hpp"
#include "../validation/resize_cases.hpp"
#include <climits>
#include <thread>
namespace opencv_test
{
namespace
{

using namespace triton_test;

using namespace triton_test::resize_cases;

static std::string label(const RouteCase &c, int type, const char *layout)
{
    return cv::format("%dx%d -> %dx%d fx=%g fy=%g interp=%d type=%s %s", c.src.width, c.src.height,
                      c.dst.width, c.dst.height, c.fx, c.fy, c.interp, typeToString(type).c_str(),
                      layout);
}

// Every case on a non-default stream: whole images, new output buffer.
TEST(Triton_Resize, RoutesWholeImagesOnStream)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    uint64 seed = 1;
    cuda::Stream stream;
    for (const RouteCase &c : kCases)
        for (int type : kTypes)
        {
            SCOPED_TRACE(label(c, type, "whole"));
            const Mat host = randomImage(c.src, type, seed++);
            cuda::GpuMat src(host), dst;
            cv::triton::resize(src, dst, c.dst, c.fx, c.fy, c.interp, stream);
            stream.waitForCompletion();
            EXPECT_STREQ(c.route, td::runtimeStats().last_route.c_str());
            Mat got;
            dst.download(got);
            EXPECT_TRUE(meetsContract(got, oracle(host, c.dst, c.fx, c.fy, c.interp), c.interp));
        }
}

// ROI source and destination with padded steps and odd offsets: the output is reused (same
// address), the result equals the oracle, nothing outside the destination ROI is written.
TEST(Triton_Resize, RoutesRoiAndOutputReuse)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    uint64 seed = 100000;
    cuda::Stream stream;
    for (const RouteCase &c : kCases)
        for (int type : kTypes)
        {
            SCOPED_TRACE(label(c, type, "roi"));
            const Mat host = randomImage(c.src, type, seed++);
            const Size dsz = c.dst.empty()
                                 ? Size(cvRound(c.src.width * c.fx), cvRound(c.src.height * c.fy))
                                 : c.dst;
            const double guard = CV_MAT_DEPTH(type) == CV_8U ? 77.0 : 77.5;
            cuda::GpuMat srcParent, dstParent;
            const Point so(3, 1), dso(5, 2);
            cuda::GpuMat src = roiOf(srcParent, c.src, type, so, 7, 3, 0.0);
            cuda::GpuMat dst = roiOf(dstParent, dsz, type, dso, 9, 5, guard);
            src.upload(host);
            const void *before = dst.cudaPtr();
            cv::triton::resize(src, dst, c.dst, c.fx, c.fy, c.interp, stream);
            stream.waitForCompletion();
            EXPECT_EQ(before, dst.cudaPtr()) << "the output was reallocated";
            EXPECT_STREQ(c.route, td::runtimeStats().last_route.c_str());
            Mat got;
            dst.download(got);
            EXPECT_TRUE(meetsContract(got, oracle(host, c.dst, c.fx, c.fy, c.interp), c.interp));
            EXPECT_TRUE(guardsIntact(dstParent, Rect(dso, dsz), guard))
                << "wrote outside the destination ROI";
        }
}

// The default (null) stream takes the same path.
TEST(Triton_Resize, DefaultStream)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    uint64 seed = 200000;
    for (const RouteCase &c : kCases)
        for (int type : {CV_8UC3, CV_32FC4})
        {
            SCOPED_TRACE(label(c, type, "null stream"));
            const Mat host = randomImage(c.src, type, seed++);
            cuda::GpuMat src(host), dst;
            cv::triton::resize(src, dst, c.dst, c.fx, c.fy, c.interp);
            cuda::Stream::Null().waitForCompletion();
            Mat got;
            dst.download(got);
            EXPECT_TRUE(meetsContract(got, oracle(host, c.dst, c.fx, c.fy, c.interp), c.interp));
        }
}

TEST(Triton_Resize, CopyRouteIsNotALaunch)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    const Mat host = randomImage(Size(33, 21), CV_8UC3, 7);
    cuda::GpuMat src(host), dst;
    const td::RuntimeStats a = td::runtimeStats();
    cv::triton::resize(src, dst, host.size(), 0, 0, INTER_CUBIC);
    cuda::Stream::Null().waitForCompletion();
    const td::RuntimeStats b = td::runtimeStats();
    EXPECT_EQ(a.copies + 1, b.copies);
    EXPECT_EQ(a.launches, b.launches);
    Mat got;
    dst.download(got);
    EXPECT_EQ(0.0, cvtest::norm(got, host, NORM_INF));
}

TEST(Triton_Resize, RejectsWithoutFallback)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    cuda::GpuMat u8(32, 32, CV_8UC3), dst;
    const auto code = [&](const std::function<void()> &f) -> int
    {
        try
        {
            f();
        }
        catch (const cv::Exception &e)
        {
            return e.code;
        }
        return 0;
    };
    // Every declared interpolation is served; rejected requests still submit nothing.
    const td::RuntimeStats a = td::runtimeStats();
    EXPECT_EQ(Error::StsUnsupportedFormat, code(
                                               [&]
                                               {
                                                   cv::triton::resize(u8, dst, Size(50, 50), 0, 0,
                                                                      INTER_LANCZOS4);
                                               }));
    const td::RuntimeStats b = td::runtimeStats();
    EXPECT_EQ(a.launches, b.launches);
    EXPECT_EQ(a.copies, b.copies);
    // Unsupported input formats.
    cuda::GpuMat out;
    EXPECT_EQ(Error::StsUnsupportedFormat, code(
                                               [&]
                                               {
                                                   cv::triton::resize(u8, out, Size(16, 16), 0, 0,
                                                                      INTER_LANCZOS4);
                                               }));
    cuda::GpuMat u16(8, 8, CV_16UC1);
    EXPECT_EQ(Error::StsUnsupportedFormat, code(
                                               [&]
                                               {
                                                   cv::triton::resize(u16, out, Size(4, 4));
                                               }));
    cuda::GpuMat two(8, 8, CV_8UC2);
    EXPECT_EQ(Error::StsUnsupportedFormat, code(
                                               [&]
                                               {
                                                   cv::triton::resize(two, out, Size(4, 4));
                                               }));
    // Bad arguments.
    EXPECT_EQ(Error::StsBadArg, code(
                                    [&]
                                    {
                                        cv::triton::resize(u8, out, Size(), 0, 0);
                                    }));
    EXPECT_EQ(Error::StsBadArg, code(
                                    [&]
                                    {
                                        cv::triton::resize(u8, u8, Size(16, 16));
                                    }));
    cuda::GpuMat parent(64, 64, CV_8UC1);
    parent.setTo(Scalar::all(0));
    cuda::GpuMat s0 = parent(Rect(0, 0, 32, 32)), d0 = parent(Rect(16, 16, 16, 16));
    EXPECT_EQ(Error::StsBadArg, code(
                                    [&]
                                    {
                                        cv::triton::resize(s0, d0, Size(16, 16));
                                    }));
}

TEST(Triton_Resize, ScaleAndInterpolationValidationPrecedesOutputAllocation)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    const Mat host = randomImage(Size(12, 8), CV_8UC3, 901);
    cuda::GpuMat src(host), dst;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const td::RuntimeStats before = td::runtimeStats();
    for (double fx : {nan, inf, -inf, 1e300})
    {
        EXPECT_THROW(cv::triton::resize(src, dst, Size(), fx, 0.5), cv::Exception);
        EXPECT_TRUE(dst.empty()) << "invalid scale allocated dst";
    }
    for (int mode :
         {-1, INTER_LINEAR | WARP_INVERSE_MAP, int(INTER_NEAREST_EXACT), int(INTER_LINEAR_EXACT)})
    {
        EXPECT_THROW(cv::triton::resize(src, dst, Size(6, 4), 0, 0, mode), cv::Exception);
        EXPECT_TRUE(dst.empty()) << "invalid interpolation allocated dst";
    }
    EXPECT_EQ(before.launches, td::runtimeStats().launches);
    EXPECT_EQ(before.copies, td::runtimeStats().copies);
    // Explicit dsize ignores fx/fy, including their finiteness.
    cv::triton::resize(src, dst, Size(6, 4), nan, inf, INTER_LINEAR);
    Mat got;
    dst.download(got);
    EXPECT_TRUE(meetsContract(got, oracle(host, Size(6, 4), 0, 0, INTER_LINEAR), INTER_LINEAR));
}

TEST(Triton_Boundary, RejectsInvalidAllocationAndStrideBeforeLaunch)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    cuda::GpuMat allocation(8, 8, CV_8UC1);
    cuda::GpuMat tooLarge(128, 128, CV_8UC1, allocation.cudaPtr(), 128);
    cuda::GpuMat hugeStride(2, 8, CV_8UC1, allocation.cudaPtr(), static_cast<size_t>(INT_MAX));
    cuda::GpuMat out;
    const double matrix[6] = {1, 0, 0, 0, 1, 0};
    const Mat M(2, 3, CV_64F, const_cast<double *>(matrix));
    cuda::GpuMat unaligned(2, 4, CV_32FC1, static_cast<unsigned char *>(allocation.cudaPtr()) + 1,
                           32);
    for (const cuda::GpuMat *bad : {&tooLarge, &hugeStride, &unaligned})
    {
        const auto before = td::runtimeStats();
        EXPECT_THROW(cv::triton::resize(*bad, out, Size(4, 4)), cv::Exception);
        EXPECT_THROW(cv::triton::cvtColor(*bad, out, COLOR_GRAY2BGR), cv::Exception);
        EXPECT_THROW(cv::triton::warpAffine(*bad, out, M, Size(4, 4)), cv::Exception);
        EXPECT_EQ(before.launches, td::runtimeStats().launches);
        EXPECT_EQ(before.copies, td::runtimeStats().copies);
    }
}

TEST(Triton_Boundary, RejectsForeignBuffersAndStreamsBeforeAllocation)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    if (cuda::getCudaEnabledDeviceCount() < 2)
        throw SkipTestException("two devices required for actual context rejection");
    const double matrix[6] = {1, 0, 0, 0, 1, 0};
    const Mat M(2, 3, CV_64F, const_cast<double *>(matrix));
    cuda::setDevice(0);
    cuda::GpuMat local(16, 16, CV_8UC3), result;
    cuda::Stream localStream;
    cuda::setDevice(1);
    cuda::GpuMat foreign(16, 16, CV_8UC3);
    {
        cuda::Stream foreignStream;
        cuda::setDevice(0);
        const auto before = td::runtimeStats();
        EXPECT_THROW(cv::triton::resize(foreign, result, Size(8, 8)), cv::Exception);
        EXPECT_THROW(cv::triton::cvtColor(foreign, result, COLOR_BGR2GRAY), cv::Exception);
        EXPECT_THROW(cv::triton::warpAffine(foreign, result, M, Size(8, 8)), cv::Exception);
        const void *foreignPtr = foreign.cudaPtr();
        EXPECT_THROW(cv::triton::resize(local, foreign, Size(8, 8)), cv::Exception);
        EXPECT_THROW(cv::triton::cvtColor(local, foreign, COLOR_BGR2GRAY), cv::Exception);
        EXPECT_THROW(cv::triton::warpAffine(local, foreign, M, Size(8, 8)), cv::Exception);
        EXPECT_EQ(foreignPtr, foreign.cudaPtr());
        EXPECT_THROW(
            cv::triton::resize(local, result, Size(8, 8), 0, 0, INTER_LINEAR, foreignStream),
            cv::Exception);
        EXPECT_THROW(cv::triton::cvtColor(local, result, COLOR_BGR2GRAY, foreignStream),
                     cv::Exception);
        EXPECT_THROW(cv::triton::warpAffine(local, result, M, Size(8, 8), INTER_LINEAR,
                                            BORDER_CONSTANT, Scalar(), foreignStream),
                     cv::Exception);
        EXPECT_EQ(before.launches, td::runtimeStats().launches);
        EXPECT_EQ(before.copies, td::runtimeStats().copies);
        cuda::setDevice(1);
    }
    foreign.release();
    cuda::setDevice(0);
}

TEST(Triton_Boundary, ExplicitProducerConsumerAndSustainedReuse)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    const double matrix[6] = {1, 0, 0, 0, 1, 0};
    const Mat M(2, 3, CV_64F, const_cast<double *>(matrix));
    cuda::Stream producer, target, consumer;
    cuda::Event uploaded, completed;
    cuda::GpuMat src, resized, gray, warped;
    Mat a, b, c;
    for (int round = 0; round < 24; ++round)
    {
        const Mat host = randomImage(Size(48, 32), CV_8UC3, 500000 + round);
        src.upload(host, producer);
        uploaded.record(producer);
        target.waitEvent(uploaded);
        cv::triton::resize(src, resized, Size(24, 16), 0, 0, INTER_CUBIC, target);
        cv::triton::cvtColor(src, gray, COLOR_BGR2GRAY, target);
        cv::triton::warpAffine(src, warped, M, host.size(), INTER_NEAREST, BORDER_CONSTANT,
                               Scalar(), target);
        completed.record(target);
        consumer.waitEvent(completed);
        resized.download(a, consumer);
        gray.download(b, consumer);
        warped.download(c, consumer);
        consumer.waitForCompletion();
        EXPECT_TRUE(meetsContract(a, oracle(host, Size(24, 16), 0, 0, INTER_CUBIC), INTER_CUBIC));
        Mat grayRef;
        cv::cvtColor(host, grayRef, COLOR_BGR2GRAY);
        EXPECT_EQ(0.0, cvtest::norm(b, grayRef, NORM_INF));
        EXPECT_EQ(0.0, cvtest::norm(c, host, NORM_INF));
    }
    td::reapCompleted();
    EXPECT_EQ(0, td::runtimeStats().pending);
}

TEST(Triton_Boundary, BorrowedBufferOnFreshHostThread)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    cuda::setDevice(0);
    const Mat host(32, 48, CV_8UC3, Scalar(37, 37, 37));
    cuda::GpuMat src(host);
    bool passed = false;
    std::string failure;
    std::thread worker(
        [&]
        {
            try
            {
                cuda::GpuMat small, gray, warped;
                cv::triton::resize(src, small, Size(24, 16));
                cv::triton::cvtColor(src, gray, COLOR_BGR2GRAY);
                const double matrix[6] = {1, 0, 0, 0, 1, 0};
                Mat M(2, 3, CV_64F, const_cast<double *>(matrix));
                cv::triton::warpAffine(src, warped, M, src.size(), INTER_NEAREST);
                cuda::Stream::Null().waitForCompletion();
                Mat a, b, c;
                small.download(a);
                gray.download(b);
                warped.download(c);
                passed = cvtest::norm(a, Mat(a.size(), a.type(), Scalar::all(37)), NORM_INF) == 0 &&
                         cvtest::norm(b, Mat(b.size(), b.type(), Scalar::all(37)), NORM_INF) == 0 &&
                         cvtest::norm(c, host, NORM_INF) == 0;
            }
            catch (const cv::Exception &e)
            {
                failure = e.what();
            }
        });
    worker.join();
    EXPECT_TRUE(passed) << failure;
}

}
} // namespace
