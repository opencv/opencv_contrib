// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// cv::triton::cvtColor through its real public entry (GpuMat + C++ planner + binaries from the
// runtime JIT) against OpenCV's CPU cv::cvtColor.
//
// The comparison target is this build's own cv::cvtColor. That is a definite reference, not a
// claim that every build computes the same numbers. Integer/copy paths require exact results;
// float arithmetic uses the documented per-pixel bound. IPP, HAL and CPU dispatch can change
// the linked reference arithmetic, so the actual build is part of the evidence.
// ReferenceBuildIsCharacterised below records that configuration by value; it does not claim to
// prove which code executed. The codes whose CPU result is not even self-consistent (HSV2BGR,
// HLS -- float32, SIMD body vs scalar tail) are refused by the module and absent from this file.
//
// Coverage: every served code on random images (MatchesCpu), on hand-picked boundary colours
// (MatchesCpuOnBoundaries), across block-boundary widths and padded/offset layouts
// (MatchesCpuOnLayouts), plus artifact reuse and in-flight resource retention.
#include "test_helpers.hpp"
#include "../validation/cvtcolor_cases.hpp"
#include <chrono>
#include <thread>
#include <cuda.h>
namespace opencv_test
{
namespace
{

using namespace triton_test;
using namespace triton_test::cvt_cases;

static void CUDA_CB hostSleepCvt(void *ms)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(*static_cast<int *>(ms)));
}
static void blockStream(cuda::Stream &s, int *ms)
{
    ASSERT_EQ(CUDA_SUCCESS, cuLaunchHostFunc(cuda::StreamAccessor::getStream(s), hostSleepCvt, ms));
}
struct Restore
{
    ~Restore()
    {
        td::setCacheLimits(64, 256);
    }
};

static std::string label(const CvtCase &c, Size sz)
{
    return cv::format("%s %dx%d", c.name, sz.width, sz.height);
}

// Sizes chosen to hit the interesting boundaries of a flat pass over pixels: smaller than one
// BLOCK, exactly one BLOCK, one more, a row that is not a multiple of the block, and a real
// image size. The kernel's masking is the thing under test at the small end.
const Size kSizes[] = {Size(1, 1),   Size(13, 7),  Size(256, 1),
                       Size(257, 3), Size(97, 61), Size(640, 480)};

// The 4:2:0 codes have their own size rule (even width, source height divisible by 3), so the
// general list above contains shapes the CPU itself rejects. These are the same boundary ideas
// (smaller than a block, exactly a block, not a multiple of one, a real size) restricted to legal
// 4:2:0 sources.
const Size kSizes420[] = {Size(2, 2),    Size(4, 2),   Size(16, 4),
                          Size(130, 10), Size(256, 6), Size(640, 480)};

TEST(Triton_CvtColor, MatchesCpu)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    uint64 seed = 1;
    cuda::Stream stream;
    for (const CvtCase &c : kCases)
        for (const Size &sz : kSizes)
        {
            if (c.yuv420)
                continue; // covered by MatchesCpu420 with its own sizes
            // The 4:2:2 CONVERSIONS require an even width; the CPU asserts
            // (`sz.width % 2 == 0`, color.simd_helpers.hpp:119) rather than producing an answer,
            // so an odd width from the shared list is not a legal input for them. The two
            // YUV2GRAY_* codes are the exception -- extractChannel has no parity assert -- so the
            // test keeps them on the full list.
            const bool yuv422conv = c.scn == 2 && c.dcn != 1;
            if (yuv422conv && (sz.width % 2) != 0)
                continue;
            SCOPED_TRACE(label(c, sz));
            // A 4:2:0 source is (3H/2) x W and decodes to H x W, so the fixture builds the taller
            // buffer and expects the shorter result. Everything else keeps the size, and the
            // assertion below still pins that.
            const Size srcSz = sz;
            const Mat host = randomImage(srcSz, CV_MAKETYPE(c.depth, c.scn), seed++);
            cuda::GpuMat src(host), dst;
            cv::triton::cvtColor(src, dst, c.code, stream);
            stream.waitForCompletion();
            EXPECT_STREQ(c.name, td::runtimeStats().last_route.c_str());
            Mat got;
            dst.download(got);
            ASSERT_EQ(got.size(), host.size());
            ASSERT_EQ(got.channels(), c.dcn);
            // Integer arithmetic on both sides: exact, not "within a tolerance".
            EXPECT_EQ(0.0, cvtest::norm(got, cpuCvt(host, c.code), NORM_INF));
        }
}

// Full finite 32F coverage, including mixed per-pixel scales.
TEST(Triton_CvtColor, MatchesCpu32F)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    uint64 seed = 7000;
    cuda::Stream stream;
    for (const CvtCase &c : kCases32F)
        for (const Size &sz : kSizes)
        {
            SCOPED_TRACE(cv::format("%s %dx%d", c.name, sz.width, sz.height));
            Mat host = randomImage(sz, CV_MAKETYPE(CV_32F, c.scn), seed++);
            // Adjacent small and large pixels expose a mistakenly image-global bound.
            for (int y = 0; y < host.rows; ++y)
                for (int x = 0; x < host.cols; ++x)
                    if ((x + y) % 3 == 0)
                        for (int k = 0; k < c.scn; ++k)
                            host.ptr<float>(y)[x * c.scn + k] *= 32767.5f;
            cuda::GpuMat src(host), dst;
            cv::triton::cvtColor(src, dst, c.code, stream);
            stream.waitForCompletion();
            // Not compared against c.name: several codes are enum ALIASES of one another
            // (GRAY2BGRA == GRAY2RGBA == 9, BGR2RGB/BGR2BGRA have RGB-order twins), so the route
            // reports the name of whichever row the table holds, not the caller's spelling.
            // MatchesCpu's name check does hold for the non-aliased codes it covers.
            Mat got;
            dst.download(got);
            ASSERT_EQ(got.size(), host.size());
            ASSERT_EQ(got.type(), CV_MAKETYPE(CV_32F, c.dcn));
            // Two different claims, and the test must not blur them. The bit-copy codes write each
            // source float into a destination slot unchanged, so equality holds (including the
            // alpha = 1.0f the inserted channel gets at 32F -- it is 255 at 8U, which is why the
            // kernel derives it). The arithmetic codes have several CPU answers, so they are
            // checked against the per-pixel bound, not for equality.
            const bool arithmetic = cvt_cases::floatArithmetic(c.code);
            const Mat ref = cpuCvt(host, c.code);
            EXPECT_TRUE(meetsCvtContract(got, ref, host, arithmetic,
                                         arithmetic && cv_triton_color::float_cpu_separate4()));
        }
}

// The 4:2:0 decode half. A separate test rather than more rows in MatchesCpu because the size
// rule differs: the source is (3H/2) x W and the result is H x W, and the CPU itself rejects
// shapes from the general list (it asserts an even width and a height divisible by 3).
TEST(Triton_CvtColor, MatchesCpu420)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    uint64 seed = 900;
    cuda::Stream stream;
    for (const CvtCase &c : kCases)
    {
        if (!c.yuv420)
            continue;
        for (const Size &sz : kSizes420)
        {
            SCOPED_TRACE(cv::format("%s %dx%d", c.name, sz.width, sz.height));
            const Mat host =
                randomImage(Size(sz.width, sz.height * 3 / 2), CV_MAKETYPE(c.depth, c.scn), seed++);
            cuda::GpuMat src(host), dst;
            cv::triton::cvtColor(src, dst, c.code, stream);
            stream.waitForCompletion();
            EXPECT_STREQ(c.name, td::runtimeStats().last_route.c_str());
            Mat got;
            dst.download(got);
            // The size really did change: this is the property that distinguishes the family, and
            // The destination must use the decoded dimensions rather than src.size().
            ASSERT_EQ(got.size(), sz) << "4:2:0 decode must produce H x W, not the source shape";
            ASSERT_EQ(got.channels(), c.dcn);
            EXPECT_EQ(0.0, cvtest::norm(got, cpuCvt(host, c.code), NORM_INF));
        }
    }
}

// Bayer uses the same integer algorithm at 8U and 16U. Reuse the image sizes
// to check full-range 16U arithmetic and element-size-dependent addressing.
// Other supported 16U families are covered by the shared depth fixtures.
TEST(Triton_CvtColor, MatchesCpu16U)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    uint64 seed = 4000;
    cuda::Stream stream;
    for (const CvtCase &c : kCases16U)
        for (const Size &sz : kSizes)
        {
            SCOPED_TRACE(label(c, sz) + " (16U)");
            const Mat host = randomImage(sz, CV_MAKETYPE(c.depth, c.scn), seed++);
            // Prove the fixture is really 16U with the full range filled: a CV_8U image here
            // would make every comparison an 8U comparison and pass while covering nothing.
            ASSERT_EQ(CV_16U, host.depth());
            double lo, hi;
            cv::minMaxLoc(host, &lo, &hi);
            ASSERT_GT(hi, 256.0) << "fixture: 16U values never leave the 8-bit range";
            cuda::GpuMat src(host), dst;
            cv::triton::cvtColor(src, dst, c.code, stream);
            stream.waitForCompletion();
            EXPECT_STREQ(c.name, td::runtimeStats().last_route.c_str());
            Mat got;
            dst.download(got);
            ASSERT_EQ(got.size(), host.size());
            ASSERT_EQ(got.channels(), c.dcn);
            EXPECT_EQ(CV_16U, got.depth()) << "the destination changed depth";
            EXPECT_EQ(0.0, cvtest::norm(got, cpuCvt(host, c.code), NORM_INF));
        }
}

// The default stream and output reuse: a second call into the same destination must not
// reallocate it, and must give the same answer.
TEST(Triton_CvtColor, ReusesOutputOnDefaultStream)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    const Mat host = randomImage(Size(131, 97), CV_8UC3, 7);
    cuda::GpuMat src(host), dst;
    cv::triton::cvtColor(src, dst, COLOR_BGR2YUV);
    cuda::Stream::Null().waitForCompletion();
    const void *before = dst.cudaPtr();
    cv::triton::cvtColor(src, dst, COLOR_BGR2YUV);
    cuda::Stream::Null().waitForCompletion();
    EXPECT_EQ(before, dst.cudaPtr()) << "the output was reallocated on the second call";
    Mat got;
    dst.download(got);
    EXPECT_EQ(0.0, cvtest::norm(got, cpuCvt(host, COLOR_BGR2YUV), NORM_INF));
}

// ROI source and destination: the result equals the CPU conversion of the source ROI, and
// nothing outside the destination ROI is written.
TEST(Triton_CvtColor, RoutesRoi)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    uint64 seed = 5000;
    cuda::Stream stream;
    for (const CvtCase &c : kCases)
    {
        SCOPED_TRACE(c.name);
        // Even width satisfies packed 4:2:2 input. A row count divisible by
        // three also permits packed 4:2:0 input, whose output has 2/3 as many
        // rows. Other conversions preserve size. Both ROIs retain parent pitch.
        const Size sz(72, 45);
        const Size dsz = (c.yuv420) ? Size(sz.width, sz.height * 2 / 3) : sz;
        const Mat host = randomImage(sz, CV_MAKETYPE(c.depth, c.scn), seed++);
        cuda::GpuMat srcParent, dstParent;
        cuda::GpuMat src =
            roiOf(srcParent, sz, CV_MAKETYPE(c.depth, c.scn), Point(3, 1), 7, 3, 0.0);
        cuda::GpuMat dst =
            roiOf(dstParent, dsz, CV_MAKETYPE(c.depth, c.dcn), Point(5, 2), 9, 5, 77.0);
        src.upload(host);
        const void *before = dst.cudaPtr();
        cv::triton::cvtColor(src, dst, c.code, stream);
        stream.waitForCompletion();
        EXPECT_EQ(before, dst.cudaPtr()) << "the output was reallocated";
        Mat got;
        dst.download(got);
        EXPECT_EQ(0.0, cvtest::norm(got, cpuCvt(host, c.code), NORM_INF));
        EXPECT_TRUE(guardsIntact(dstParent, Rect(Point(5, 2), dsz), 77.0))
            << "wrote outside the dst ROI";
    }
}

// Value boundaries, not random images. Random 8-bit data almost never lands on the inputs where
// a fixed-point formula saturates, where a hue branch is exactly zero, or where two channels tie
// for the maximum -- which is where the CPU's own code takes its special cases. Each case below
// is a set of literal colours chosen for that, so a failure names one real input.
struct BoundaryCase
{
    const char *name;
    std::vector<Vec3b> colours;
};

static std::vector<BoundaryCase> boundaryCases()
{
    std::vector<BoundaryCase> v;
    v.push_back({"extremes",
                 {Vec3b(0, 0, 0), Vec3b(255, 255, 255), Vec3b(0, 0, 255), Vec3b(255, 0, 0),
                  Vec3b(0, 255, 0), Vec3b(255, 255, 0), Vec3b(0, 255, 255), Vec3b(255, 0, 255)}});
    v.push_back({"grey",
                 {Vec3b(1, 1, 1), Vec3b(2, 2, 2), Vec3b(127, 127, 127), Vec3b(128, 128, 128),
                  Vec3b(254, 254, 254), Vec3b(255, 255, 255)}});
    // Channel maxima equal: d == 0, so the CPU divides by a zero diff and the s/h branches are
    // the d==0 special case, not a formula.
    v.push_back({"tie_max", {Vec3b(10, 10, 10), Vec3b(200, 200, 200), Vec3b(0, 0, 0)}});
    // Hue at each primary/secondary sector and each boundary between sectors.
    v.push_back({"hue_edges",
                 {Vec3b(255, 0, 0), Vec3b(0, 255, 0), Vec3b(0, 0, 255), Vec3b(255, 255, 0),
                  Vec3b(0, 255, 255), Vec3b(255, 0, 255), Vec3b(1, 0, 0), Vec3b(0, 1, 0),
                  Vec3b(0, 0, 1), Vec3b(255, 1, 0), Vec3b(254, 255, 0), Vec3b(0, 255, 254)}});
    // Low saturation: d == 1 and d == 2 at both ends of V -- the case where s and h are most
    // sensitive to the table entries and where saturation_cast is most likely to clip.
    v.push_back({"low_sat",
                 {Vec3b(100, 100, 101), Vec3b(100, 101, 100), Vec3b(101, 100, 100),
                  Vec3b(254, 255, 255), Vec3b(0, 0, 1), Vec3b(1, 1, 0)}});
    // Saturated chroma with a large luma spread: the YUV/YCrCb chroma terms reach their clamp.
    v.push_back({"chroma_clamp",
                 {Vec3b(0, 0, 255), Vec3b(255, 0, 0), Vec3b(0, 255, 0), Vec3b(0, 128, 255),
                  Vec3b(255, 128, 0)}});
    // Grey and near-grey YUV: the inverse transform's chroma is 128 +- 0 or 1, where the
    // (v - 128) rounding of the inverse path is decided.
    return v;
}

TEST(Triton_CvtColor, MatchesCpuOnBoundaries)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    cuda::Stream stream;
    const Size sz(61, 37);
    for (const CvtCase &c : kCases)
    {
        // Codes whose source shape is constrained (odd width forbidden, or a 4:2:0 height rule)
        // cannot run on this fixture's 61x37 buffer: the CPU itself asserts. They have their own
        // dedicated coverage (MatchesCpu420, and the 4:2:2 rows in MatchesCpu).
        if (c.yuv420 || c.scn == 2)
            continue;
        for (const BoundaryCase &b : boundaryCases())
        {
            SCOPED_TRACE(label(c, sz) + " / " + b.name);
            Mat host(sz, CV_MAKETYPE(c.depth, c.scn), Scalar::all(0));
            // Use the conversion's channel count to address interleaved pixels.
            for (int i = 0; i < (int)b.colours.size(); ++i)
            {
                const Vec3b &v = b.colours[i];
                uchar *px = host.ptr(i / sz.width) + (i % sz.width) * c.scn;
                // Only `scn` bytes belong to this pixel; guard every additional
                // channel write to avoid modifying neighboring pixels.
                px[0] = v[0];
                if (c.scn > 1)
                    px[1] = v[1];
                if (c.scn > 2)
                    px[2] = v[2];
                if (c.scn == 4)
                    px[3] = (uchar)(i * 37); // alpha: carried, never computed
            }
            // Prove the fixture before using it: the colours really are where they were meant
            // to go, and (for the 4-channel codes) alpha really is set. Without this the test
            // compares the kernel with the CPU on the same wrong input and passes.
            for (int i = 0; i < (int)b.colours.size(); ++i)
            {
                const Vec3b &v = b.colours[i];
                const uchar *px = host.ptr(i / sz.width) + (i % sz.width) * c.scn;
                ASSERT_EQ(v[0], px[0]) << "fixture: colour " << i << " byte 0";
                if (c.scn > 1)
                    ASSERT_EQ(v[1], px[1]) << "fixture: colour " << i << " byte 1";
                if (c.scn > 2)
                    ASSERT_EQ(v[2], px[2]) << "fixture: colour " << i << " byte 2";
                if (c.scn == 4)
                    ASSERT_EQ((uchar)(i * 37), px[3]) << "fixture: alpha " << i;
            }
            // The inverse codes read [Y, Cr, Cb] / [Y, U, V]: sweeping their input space is the
            // same exercise as sweeping a colour, so feed the same literal tuples either way.
            cuda::GpuMat src(host), dst;
            cv::triton::cvtColor(src, dst, c.code, stream);
            stream.waitForCompletion();
            Mat got;
            dst.download(got);
            EXPECT_EQ(0.0, cvtest::norm(got, cpuCvt(host, c.code), NORM_INF));
        }
    }
}

// Every size near a block boundary and every source/destination geometry whose row pitch is not
// the natural one. The kernel addresses row*pitch + col*ch, so the pitch is the thing that can be
// wrong, and it is wrong exactly at the sizes where the allocator pads.
TEST(Triton_CvtColor, MatchesCpuOnLayouts)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    const int block = 256; // cv_triton_color::block_size: at cols-1, cols, cols+1
    const Size sizes[] = {Size(block - 1, 5), Size(block, 5), Size(block + 1, 5), Size(1, 257),
                          Size(3, 1),         Size(3, 2),     Size(4, 3)};
    uint64 seed = 900;
    for (const CvtCase &c : kCases)
        for (const Size &sz : sizes)
            for (int padX : {0, 1, 5, 17})
            {
                // The CPU REJECTS these shapes itself, so there is no reference answer to compare
                // against and the failure would be in the harness, not the kernel: 4:2:0 needs an
                // even width and a source height divisible by 3, and a 4:2:2 CONVERSION needs an
                // even width (color.simd_helpers.hpp:119). The YUV2GRAY_* 4:2:2 codes are
                // extractChannel and stay in.
                if (c.yuv420)
                    continue;
                if (c.scn == 2 && c.dcn != 1 && (sz.width % 2) != 0)
                    continue;
                SCOPED_TRACE(label(c, sz) + cv::format(" padX=%d", padX));
                const Mat host = randomImage(sz, CV_MAKETYPE(c.depth, c.scn), seed++);
                cuda::GpuMat srcParent, dstParent;
                cuda::GpuMat src = roiOf(srcParent, sz, CV_MAKETYPE(c.depth, c.scn), Point(0, 0),
                                         sz.width + padX, 2, 0.0);
                src.upload(host);
                // The destination is an offset ROI inside a padded parent, so both its pitch and
                // its begin pointer differ from src's, and the guard outside it must survive.
                cuda::GpuMat dst = roiOf(dstParent, sz, CV_MAKETYPE(c.depth, c.dcn), Point(3, 1),
                                         sz.width + padX + 9, 4, 77.0);
                cv::triton::cvtColor(src, dst, c.code);
                cuda::Stream::Null().waitForCompletion();
                Mat got;
                dst.download(got);
                EXPECT_EQ(0.0, cvtest::norm(got, cpuCvt(host, c.code), NORM_INF));
                EXPECT_TRUE(guardsIntact(dstParent, Rect(Point(3, 1), sz), 77.0))
                    << "wrote outside the destination ROI";
            }
}

// Everything the module must refuse, with the error code it must use. A call that returns has
// run the Triton kernel: there is no quiet fallback to the CPU implementation.
TEST(Triton_CvtColor, RejectsWithoutFallback)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    cuda::GpuMat src(32, 32, CV_8UC3), dst;
    const auto code = [&](const std::function<void()> &f) -> int
    {
        const auto before = td::runtimeStats();
        try
        {
            f();
        }
        catch (const cv::Exception &e)
        {
            EXPECT_EQ(before.launches, td::runtimeStats().launches)
                << "a rejected request launched work";
            return e.code;
        }
        return 0;
    };
    // The float32 conversions: not served, refused by name rather than under a tolerance.
    EXPECT_EQ(Error::StsUnsupportedFormat, code(
                                               [&]
                                               {
                                                   cv::triton::cvtColor(src, dst, COLOR_HSV2BGR);
                                               }));
    EXPECT_EQ(Error::StsUnsupportedFormat, code(
                                               [&]
                                               {
                                                   cv::triton::cvtColor(src, dst, COLOR_BGR2HLS);
                                               }));
    // Forward linear-Luv is unsupported. The CPU SIMD body and scalar tail can
    // round differently at channel boundaries; this is a distinct restriction
    // from the unsupported CV_32F conversion families checked below.
    EXPECT_EQ(Error::StsUnsupportedFormat, code(
                                               [&]
                                               {
                                                   cv::triton::cvtColor(src, dst, COLOR_LBGR2Luv);
                                               }));
    EXPECT_EQ(Error::StsUnsupportedFormat, code(
                                               [&]
                                               {
                                                   cv::triton::cvtColor(src, dst, COLOR_LRGB2Luv);
                                               }));
    // Supported 32F conversions are the bit-copy codes and the
    // three arithmetic families (colour->gray, YCrCb, YUV). Everything else at 32F is refused.
    cuda::GpuMat f32(32, 32, CV_32FC3);
    EXPECT_EQ(Error::StsUnsupportedFormat, code(
                                               [&]
                                               {
                                                   cv::triton::cvtColor(f32, dst, COLOR_BGR2XYZ);
                                               }));
    EXPECT_EQ(Error::StsUnsupportedFormat, code(
                                               [&]
                                               {
                                                   cv::triton::cvtColor(f32, dst, COLOR_BGR2HSV);
                                               }));
    EXPECT_EQ(Error::StsUnsupportedFormat, code(
                                               [&]
                                               {
                                                   cv::triton::cvtColor(f32, dst, COLOR_BGR2Lab);
                                               }));
    // ...and the arithmetic ones really are served, not merely not-refused.
    {
        cuda::GpuMat g32, d32;
        EXPECT_NO_THROW(cv::triton::cvtColor(f32, d32, COLOR_BGR2GRAY));
        EXPECT_EQ(CV_32FC1, d32.type());
    }
    // extractChannel itself supports floats, but cvtColorYUV2Gray_ch explicitly
    // restricts these conversion codes to CV_8U. Both real public entries reject.
    Mat extractHost(3, 7, CV_32FC2, Scalar::all(1));
    cuda::GpuMat extractGpu(extractHost);
    for (int extractCode : {COLOR_YUV2GRAY_UYVY, COLOR_YUV2GRAY_YUY2})
    {
        Mat ref;
        EXPECT_THROW(cv::cvtColor(extractHost, ref, extractCode), cv::Exception);
        EXPECT_EQ(Error::StsUnsupportedFormat, code(
                                                   [&]
                                                   {
                                                       cv::triton::cvtColor(extractGpu, dst,
                                                                            extractCode);
                                                   }));
    }
    // A source with the wrong channel count for the code.
    cuda::GpuMat c1(32, 32, CV_8UC1);
    EXPECT_EQ(Error::StsUnsupportedFormat, code(
                                               [&]
                                               {
                                                   cv::triton::cvtColor(c1, dst, COLOR_BGR2GRAY);
                                               }));
    // In-place.
    cuda::GpuMat inplace(32, 32, CV_8UC3);
    EXPECT_EQ(Error::StsBadArg, code(
                                    [&]
                                    {
                                        cv::triton::cvtColor(inplace, inplace, COLOR_BGR2RGB);
                                    }));
    // Bayer VNG: 8U-only on the CPU AND a gradient search rather than the bilinear taps, so it
    // is not in the table. Refused, not served as "close enough" (demosaicing.cpp:1806 rejects
    // 16U for it and :1825-1830 routes it to a different kernel entirely).
    cuda::GpuMat bayer8(32, 32, CV_8UC1);
    EXPECT_EQ(Error::StsUnsupportedFormat, code(
                                               [&]
                                               {
                                                   cv::triton::cvtColor(bayer8, dst,
                                                                        COLOR_BayerBG2BGR_VNG);
                                               }));
    // The Bayer family at a depth the CPU does not support there either: 32F.
    cuda::GpuMat bayer32f(32, 32, CV_32FC1);
    EXPECT_EQ(Error::StsUnsupportedFormat, code(
                                               [&]
                                               {
                                                   cv::triton::cvtColor(bayer32f, dst,
                                                                        COLOR_BayerBG2BGR);
                                               }));
    // 16U is admitted for Bayer, gray, YUV/YCrCb and gray -> colour -- the families whose CPU
    // path is the same integer formula at both depths. Everything else is still 8U-only, and
    // these are the cases that would slip through if the gate were widened to "any op at 16U":
    // a colour permute (excluded by the scn == 1 condition), HSV, and the packed format.
    cuda::GpuMat g16(32, 32, CV_16UC3);
    EXPECT_EQ(Error::StsUnsupportedFormat, code(
                                               [&]
                                               {
                                                   cv::triton::cvtColor(g16, dst, COLOR_BGR2BGRA);
                                               }));
    EXPECT_EQ(Error::StsUnsupportedFormat, code(
                                               [&]
                                               {
                                                   cv::triton::cvtColor(g16, dst, COLOR_BGR2HSV);
                                               }));
    cuda::GpuMat g16packed(32, 32, CV_16UC2);
    EXPECT_EQ(Error::StsUnsupportedFormat, code(
                                               [&]
                                               {
                                                   cv::triton::cvtColor(g16packed, dst,
                                                                        COLOR_BGR5652BGR);
                                               }));
    // ... and a served code really is served at 16U through the same entry, so the assertion
    // above cannot pass by everything being refused.
    cv::triton::cvtColor(g16, dst, COLOR_BGR2GRAY);
    cuda::Stream::Null().waitForCompletion();
    EXPECT_EQ(CV_16U, dst.depth()) << "BGR2GRAY at 16U did not produce a 16U destination";
    // ... and src/dst depths must agree: the CPU creates dst at src's depth, so a mixed pair is
    // not something it can produce.
    // src/dst depth cannot disagree THROUGH THIS ENTRY: the adapter re-creates dst as
    // CV_MAKETYPE(src.depth(), dcn), so an existing 8U dst handed a 16U source is replaced, not
    // mixed. normalize()'s matching-depth check is therefore an internal invariant, not
    // something a caller can trip here -- and the observable behaviour is the depth following
    // the SOURCE.
    cuda::GpuMat dst8(32, 32, CV_8UC3);
    cuda::GpuMat bayer16(32, 32, CV_16UC1);
    cv::triton::cvtColor(bayer16, dst8, COLOR_BayerBG2BGR);
    cuda::Stream::Null().waitForCompletion();
    EXPECT_EQ(CV_16U, dst8.depth()) << "the destination did not follow the source depth";
    // A Bayer source must be 1-channel: 8UC3 is rejected by the CPU too
    // (CV_Assert(scn == 1 && (dcn == 3 || dcn == 4), demosaicing.cpp:1787).
    EXPECT_EQ(Error::StsUnsupportedFormat, code(
                                               [&]
                                               {
                                                   cv::triton::cvtColor(src, dst,
                                                                        COLOR_BayerBG2BGR);
                                               }));
    // A destination that already exists with the wrong channel count is made to fit, not
    // rejected: BGR2BGRA into a 3-channel buffer must produce 4 channels.
    cuda::GpuMat wrong(32, 32, CV_8UC3);
    cv::triton::cvtColor(src, wrong, COLOR_BGR2BGRA);
    cuda::Stream::Null().waitForCompletion();
    EXPECT_EQ(4, wrong.channels());
}

// Malformed shapes must stop before compilation or launch. Do not exercise an
// unsafe odd-width 422 kernel to discover the missing validation.
TEST(Triton_CvtColor, RejectsMalformedVideoBeforeLaunch)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    const int decode422[] = {COLOR_YUV2BGR_UYVY, COLOR_YUV2RGB_UYVY, COLOR_YUV2BGR_YUY2,
                             COLOR_YUV2RGB_YUY2, COLOR_YUV2BGR_YVYU, COLOR_YUV2RGB_YVYU};
    for (int code : decode422)
        for (int width : {1, 3, 31})
        {
            SCOPED_TRACE(cv::format("code=%d width=%d", code, width));
            cuda::GpuMat src(1, width, CV_8UC2), dst;
            const auto before = td::runtimeStats();
            EXPECT_THROW(cv::triton::cvtColor(src, dst, code), cv::Exception);
            EXPECT_EQ(before.launches, td::runtimeStats().launches);
        }
    for (const Size &size : {Size(31, 24), Size(32, 23)})
    {
        cuda::GpuMat src(size, CV_8UC1), dst;
        const auto before = td::runtimeStats();
        EXPECT_THROW(cv::triton::cvtColor(src, dst, COLOR_YUV2BGR_NV12), cv::Exception);
        EXPECT_EQ(before.launches, td::runtimeStats().launches);
    }
    // Odd-width Y extraction remains legal, and really launches.
    Mat host = randomImage(Size(3, 1), CV_8UC2, 12);
    cuda::GpuMat src(host), dst;
    const auto before = td::runtimeStats();
    cv::triton::cvtColor(src, dst, COLOR_YUV2GRAY_UYVY);
    cuda::Stream::Null().waitForCompletion();
    EXPECT_EQ(before.launches + 1, td::runtimeStats().launches);
    Mat got;
    dst.download(got);
    EXPECT_EQ(0.0, cvtest::norm(got, cpuCvt(host, COLOR_YUV2GRAY_UYVY), NORM_INF));
}

// CPU dispatch may use IPP or HAL substitutions in color dispatch files.
// Build information identifies configured substitutions but cannot detect a
// runtime-registered HAL. Numeric tests compare the actual linked cv::cvtColor.
static std::string buildField(const std::string &key)
{
    const std::string info = cv::getBuildInformation();
    for (std::size_t p = 0; p < info.size();)
    {
        const std::size_t e = info.find('\n', p);
        const std::string line = info.substr(p, e == std::string::npos ? std::string::npos : e - p);
        const std::size_t c = line.find(':');
        if (c != std::string::npos)
        {
            std::string k = line.substr(0, c);
            const std::size_t b = k.find_first_not_of(" \t");
            std::string v = line.substr(c + 1);
            const std::size_t vb = v.find_first_not_of(" \t");
            if (b != std::string::npos && k.substr(b) == key)
                return vb == std::string::npos ? std::string() : v.substr(vb);
        }
        if (e == std::string::npos)
            break;
        p = e + 1;
    }
    return std::string();
}

TEST(Triton_CvtColor, ReferenceBuildIsCharacterised)
{
    // Report the configuration values used by the CPU reference diagnostic.
    const std::string ipp = buildField("Intel IPP");
    const std::string hal = buildField("Custom HAL");
    const std::string base = buildField("Baseline");
    if (base.empty())
        throw SkipTestException("cannot read the build configuration");

    // IPP and custom HAL configurations are legitimate OpenCV builds. Report
    // them as a skipped diagnostic instead of rejecting the build. Accuracy
    // tests still use the linked cv::cvtColor; this check cannot establish its
    // equivalence to a generic CPU implementation.
    const bool isReference =
        (ipp == "NO" || ipp.empty()) && hal.compare(0, 2, "NO") == 0 && !cv::ipp::useIPP();
    std::cout << "cvtColor reference build: Baseline=" << base << " Intel IPP=" << ipp
              << " Custom HAL=" << hal << " ipp_active=" << (cv::ipp::useIPP() ? 1 : 0)
              << " is_reference=" << (isReference ? 1 : 0) << std::endl;
    if (!isReference)
        throw SkipTestException(
            "this build is not the generic reference configuration (Intel IPP: " + ipp +
            ", Custom HAL: " + hal +
            "); generic-path equivalence is not established for this configuration. "
            "Accuracy tests compare the linked implementation. "
            "The diagnostic cannot identify runtime-registered HAL substitutions");
}

// The same kernel reused: no second compile, no second module load, the tables uploaded once.
// The counters start at zero but the caches do not -- an earlier test in this binary may have
// already uploaded these tables -- so the invariant is "six calls upload at most once and hit
// otherwise", not "the first call uploads".
TEST(Triton_CvtColor, ReusesCompiledArtifacts)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    td::reapCompleted();
    td::resetRuntimeStats();
    cuda::Stream stream;
    const Mat host = randomImage(Size(203, 111), CV_8UC3, 77);
    cuda::GpuMat src(host), dst;
    for (int i = 0; i < 6; ++i)
    {
        cv::triton::cvtColor(src, dst, COLOR_BGR2YUV, stream);
        stream.waitForCompletion();
    }
    const td::RuntimeStats st = td::runtimeStats();
    EXPECT_EQ(6, st.launches);
    EXPECT_LE(st.module_loads, 1) << "reloaded the module on every call";
    EXPECT_LE(st.helper_compiles, 1) << "compiled again";
    EXPECT_GE(st.module_hits, 5) << "the module cache was not used";
    EXPECT_LE(st.table_uploads, 1) << "re-uploaded the div tables";
    EXPECT_GE(st.table_hits, 5) << "the table cache was not used";
    EXPECT_EQ(0, st.disk_rejected);
    Mat got;
    dst.download(got);
    EXPECT_EQ(0.0, cvtest::norm(got, cpuCvt(host, COLOR_BGR2YUV), NORM_INF));
}

// Check asynchronous resource retention with deterministic eviction. Empty
// unrelated cache entries, warm two codes, then evict a table with the second
// code and a module by shrinking the limit. Reset counters after setup so
// subsequent frees refer to the resources under test. Keeping the required
// module resident until launch also avoids a context-draining module reload.
TEST(Triton_CvtColor, HoldsResourcesWhileInFlight)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    Restore restore;
    const Mat host = randomImage(Size(157, 93), CV_8UC3, 91);
    const Mat want = cpuCvt(host, COLOR_BGR2HSV);
    cuda::GpuMat src(host), dst, other;
    cuda::Stream a(cudaStreamNonBlocking), b(cudaStreamNonBlocking);

    // Empty the caches of whatever earlier tests left, then let them grow again.
    td::setCacheLimits(1, 1);
    td::setCacheLimits(64, 256);
    td::reapCompleted();
    // Warm exactly two codes: BGR2HSV (the one that will be left pending) and BGR2YUV.
    cv::triton::cvtColor(src, other, COLOR_BGR2YUV, b);
    cv::triton::cvtColor(src, dst, COLOR_BGR2HSV, a);
    a.waitForCompletion();
    b.waitForCompletion();
    td::reapCompleted();
    // Make this test's own footprint exact: one module slot, one table slot, with BGR2HSV the
    // most recently used entry so the shrink keeps it. The leftover evicted here is not under
    // test, and its release is why the counters are reset afterwards.
    td::setCacheLimits(2, 1);
    td::resetRuntimeStats();

    int ms = 400;
    blockStream(a, &ms);
    const auto t0 = std::chrono::steady_clock::now();
    cv::triton::cvtColor(src, dst, COLOR_BGR2HSV, a); // module hit and table hit: no load
    const double returnedA =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    td::RuntimeStats st = td::runtimeStats();
    // The pending state is only meaningful if the call did not have to load or compile: either
    // would drain the context and run the block. Asserted, not assumed.
    EXPECT_EQ(0, st.module_loads) << "the measured call loaded a module";
    EXPECT_EQ(0, st.helper_compiles) << "the measured call compiled";
    EXPECT_LT(returnedA, 0.5 * ms) << "the measured call stalled the host, so the launch was "
                                      "not pending and the scenario proves nothing";
    EXPECT_EQ(1, st.launches);
    EXPECT_EQ(1, st.pending);
    const long long freesAtStart = st.module_frees + st.table_frees;
    EXPECT_EQ(0, freesAtStart);

    // Eviction 1, tables: the second code needs its own table set, and the cache holds one.
    // BGR2YUV's module is still cached (two module slots), so no load.
    cv::triton::cvtColor(src, other, COLOR_BGR2YUV, b);
    b.waitForCompletion();
    st = td::runtimeStats();
    EXPECT_EQ(0, st.module_loads) << "the second call loaded a module and drained the context";
    EXPECT_EQ(1, st.table_evictions) << "the pending call's tables were not evicted";
    EXPECT_GE(st.deferred, 1) << "the evicted tables were not counted as held by a pending launch";
    EXPECT_EQ(0, st.table_frees) << "tables a pending launch is reading were freed";

    // Eviction 2, module: the limit shrink evicts the least recently used module, which is the
    // one the pending call is using.
    td::setCacheLimits(1, 1);
    st = td::runtimeStats();
    EXPECT_GE(st.module_evictions, 1) << "no module was evicted";
    EXPECT_GE(st.deferred, 2) << "the module was evicted but not counted as held";
    EXPECT_EQ(0, st.module_frees) << "a module still in use was unloaded";

    // Drain: now the held resources must be released, and the output must be right.
    a.waitForCompletion();
    td::reapCompleted();
    st = td::runtimeStats();
    EXPECT_GE(st.module_frees + st.table_frees, 2)
        << "evicted resources were never released after the launch completed";
    EXPECT_EQ(0, st.pending);
    Mat got;
    dst.download(got);
    EXPECT_EQ(0.0, cvtest::norm(got, want, NORM_INF));
}

// The test table above must cover the module's own code table. Without this, adding a
// conversion to the planner without a test row would pass silently.
TEST(Triton_CvtColor, CoversEverySupportedCode)
{
    const auto &table = cv_triton_color::all();
    EXPECT_EQ(sizeof(kCases) / sizeof(kCases[0]), table.size());
    for (const cv_triton_color::CodeSpec &spec : table)
    {
        bool found = false;
        for (const CvtCase &c : kCases)
            if (c.code == spec.code)
            {
                found = true;
                break;
            }
        EXPECT_TRUE(found) << "code " << spec.name << " (" << spec.code << ") has no test row";
    }
}

// Depth inventories must track every admitted planner family, including copies
// whose code names are aliases. Exact counts name numeric rows, not aliases.
TEST(Triton_CvtColor, CoversEverySupportedDepth)
{
    int counts[] = {0, 0, 0};
    for (const auto &spec : cv_triton_color::all())
        for (int index = 0; index < 3; ++index)
        {
            const int depth = index == 0 ? CV_8U : index == 1 ? CV_16U : CV_32F;
            cv_triton_runtime::DeviceView src;
            src.ptr = 4096;
            src.rows = spec.op == cv_triton_color::Op::Yuv420 ? 48 : 32;
            src.cols = 32;
            src.channels = spec.scn;
            src.type = depth == CV_8U    ? cv_triton_runtime::ElemType::U8
                       : depth == CV_16U ? cv_triton_runtime::ElemType::U16
                                         : cv_triton_runtime::ElemType::F32;
            src.step = src.cols * src.channels * cv_triton_runtime::elem_size(src.type);
            cv_triton_color::Output output;
            const bool admitted = cv_triton_color::resolve_output(src, spec.code, output).ok();
            const CvtCase *row = cvt_cases::find(spec.code, depth);
            EXPECT_EQ(admitted, row != nullptr) << spec.name << " depth=" << depth;
            if (!admitted)
                continue;
            ++counts[index];
            ASSERT_TRUE(row != nullptr);
            EXPECT_EQ(spec.scn, row->scn);
            EXPECT_EQ(spec.dcn, row->dcn);
            EXPECT_EQ(row->yuv420 ? 32 : src.rows, output.rows);
            EXPECT_EQ(src.cols, output.cols);
        }
    EXPECT_EQ(84, counts[0]);
    EXPECT_EQ(20, counts[1]);
    EXPECT_EQ(20, counts[2]);
    EXPECT_EQ(size_t(counts[0]), sizeof(kCases) / sizeof(kCases[0]));
    EXPECT_EQ(size_t(counts[1]), sizeof(kCases16U) / sizeof(kCases16U[0]));
    EXPECT_EQ(size_t(counts[2]), sizeof(kCases32F) / sizeof(kCases32F[0]));
}

// The public adapters call this source-only owner before allocating output.
// Synthetic addresses are never dereferenced; malformed views must be refused here.
TEST(Triton_CvtColor, ResolvesAndRefusesBeforeOutputAllocation)
{
    cv_triton_runtime::DeviceView source;
    source.ptr = 4096;
    source.rows = 48;
    source.cols = 32;
    source.channels = 1;
    source.step = 32;
    cv_triton_color::Output output;
    ASSERT_TRUE(cv_triton_color::resolve_output(source, COLOR_YUV2BGR_NV12, output).ok());
    EXPECT_EQ(32, output.rows);
    EXPECT_EQ(32, output.cols);
    EXPECT_EQ(3, output.spec->dcn);
    auto reject = [&](const cv_triton_runtime::DeviceView &src)
    {
        EXPECT_FALSE(cv_triton_color::resolve_output(src, COLOR_YUV2BGR_NV12, output).ok());
    };
    auto bad = source;
    bad.ptr = 0;
    reject(bad);
    bad = source;
    bad.rows = 0;
    reject(bad);
    bad = source;
    bad.cols = 31;
    reject(bad);
    bad = source;
    bad.rows = 47;
    reject(bad);
    bad = source;
    bad.step = 31;
    reject(bad);
    bad = source;
    bad.type = cv_triton_runtime::ElemType::F32;
    reject(bad);
    bad = source;
    bad.rows = 2147483646;
    bad.cols = 2;
    bad.step = 2;
    reject(bad);
    bad = source;
    bad.rows = 6;
    bad.step = 536870912;
    reject(bad);
}

// Check coefficient rows against coeffs_for and verify the kernel's slot order:
// BLUE-side, green, RED-side for BGR; RED-side, green, BLUE-side for RGB.
TEST(Triton_CvtColor, CoefficientRowsMatchTheConstructor)
{
    for (const cv_triton_color::CodeSpec &spec : cv_triton_color::all())
    {
        if (spec.op != cv_triton_color::Op::Xyz && spec.op != cv_triton_color::Op::Lab &&
            spec.op != cv_triton_color::Op::Luv)
            continue;
        int k[9];
        cv_triton_color::coeffs_for(spec.code, k);
        for (int i = 0; i < 9; ++i)
            EXPECT_EQ(k[i], spec.k[i]) << spec.name << " coefficient " << i;
    }
    // Spot values with a meaning that can be checked by hand. BGR2XYZ row 0 is 2^12*M[0]/... in
    // the CPU's BGR column order, i.e. {b,g,r} = {0.180423,0.357580,0.412453}*4096 rounded; the
    // BGR2Lab rows are the white-point-scaled sRGB2XYZ matrix, and their sum is exactly 2^12
    // (the CPU's own constructor asserts it, color_lab.cpp:1583).
    const cv_triton_color::CodeSpec *x = cv_triton_color::find(COLOR_BGR2XYZ);
    ASSERT_TRUE(x != nullptr);
    EXPECT_EQ(739, x->k[0]);
    EXPECT_EQ(1465, x->k[1]);
    EXPECT_EQ(1689, x->k[2]);
    const cv_triton_color::CodeSpec *lab = cv_triton_color::find(COLOR_BGR2Lab);
    ASSERT_TRUE(lab != nullptr);
    EXPECT_EQ(778, lab->k[0]);
    EXPECT_EQ(1777, lab->k[2]);
    for (int r = 0; r < 3; ++r)
        EXPECT_EQ(4096, lab->k[3 * r] + lab->k[3 * r + 1] + lab->k[3 * r + 2])
            << "RGB2Lab_b row " << r << " must sum to 1 << lab_shift";
    // Reverse matrix rows follow destination-slot order. BGR and RGB variants
    // carry the same rows in opposite order.
    const cv_triton_color::CodeSpec *labr = cv_triton_color::find(COLOR_Lab2BGR);
    ASSERT_TRUE(labr != nullptr);
    EXPECT_EQ(217, labr->k[0]);   // row 2 of the untransposed matrix: B from the lab triple
    EXPECT_EQ(12615, labr->k[6]); // row 0: R
    const cv_triton_color::CodeSpec *labrrgb = cv_triton_color::find(COLOR_Lab2RGB);
    ASSERT_TRUE(labrrgb != nullptr);
    EXPECT_EQ(12615, labrrgb->k[0]); // the RGB destination wants R first, so the rows swap
    EXPECT_EQ(217, labrrgb->k[6]);
    // Luv's reverse matrix is the same XYZ2sRGB matrix without the white-point factor.
    const cv_triton_color::CodeSpec *luvr = cv_triton_color::find(COLOR_Luv2BGR);
    ASSERT_TRUE(luvr != nullptr);
    EXPECT_EQ(228, luvr->k[0]);
    EXPECT_EQ(13273, luvr->k[6]);
}

}
} // namespace
