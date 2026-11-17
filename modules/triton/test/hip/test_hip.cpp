// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Supplemental HIP accuracy/lifecycle checks through the public DeviceImage overloads.
//
// Standard shared cases run in opencv_test_triton on either backend. CUDA-specific fixtures
// also use GpuMat/driver APIs; this separate executable exercises HIP allocations, streams
// and lifecycle boundaries without gtest/opencv_ts. Calls use the production runtime JIT
// and the same numerical-reference owners as the shared tests:
//   resize      the portable CPU oracle (test/resize_reference.cpp, compiled into this
//               binary as a separate source); CV_8U bit-exact, CV_32F within f32Atol;
//   cvtColor    linked cv::cvtColor; integer/copy paths exact, float arithmetic per-pixel bound;
//   warpAffine  cv::warpAffine's vector path cropped to dsize (NEAREST/LINEAR) or the natural
//               call (CUBIC), as test/test_warpaffine.cpp defines it. This supplemental
//               integer fixture requires exact results (stricter than the public 16U
//               one-level bound); CV_32F uses min(5e-5 * max(|ref|, 1), 1e-4).
// Each accepted case also checks that only dst was written: the destination is a ROI of a
// padded parent whose border bytes must be unchanged. A refused case must raise the documented
// error and leave dst untouched. Scalar bounds use validation/numerics.hpp, the
// same owner as the standard fixtures, without requiring gtest / opencv_ts.
//
// Plain main, no gtest: the target builds in a minimal core,imgproc,triton tree.
// Its gtest-style filters select five registered group tests, not individual inner checks.
// Each group runs only when selected. XML lists those same group names and records
// the inner-check count separately. Exit is 0 for passing nonempty execution, 1 on
// failed checks, 2 on invalid/empty selection, and 77 when no usable device exists.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include <hip/hip_runtime_api.h>
#include "opencv2/triton.hpp"
#include "opencv2/core.hpp"
#include "../../src/runtime/runtime.hpp"
#include "../../src/runtime/cvtcolor.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/triton/device.hpp"
#include "../resize_reference.hpp"
#include "../../validation/numerics.hpp"
namespace ref = cv_triton_resize;
namespace tr = cv::triton;
#define HIPOK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call, hipGetErrorName(e_)); std::exit(200); } } while (0)
static int g_fail = 0, g_pass = 0;

// Inner checks remain individually visible in stdout. XML aggregates them under
// the five registered groups, whose names also own list/filter behavior.
struct Result
{
    std::string suite, name, failure;
};
static std::vector<Result> g_results;
static std::string g_suite = "Triton_Hip";

static void report(bool ok, const std::string &what)
{
    std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    (ok ? g_pass : g_fail)++;
    g_results.push_back({g_suite, what, ok ? std::string() : what});
}

static std::string xmlEscape(const std::string &t)
{
    std::string o;
    for (char c : t)
        switch (c)
        {
        case '&':
            o += "&amp;";
            break;
        case '<':
            o += "&lt;";
            break;
        case '>':
            o += "&gt;";
            break;
        case '"':
            o += "&quot;";
            break;
        default:
            o += c;
        }
    return o;
}

// This standalone target intentionally registers five coarse tests. Standard
// OpenCV gtest fixtures provide fine-grained parameter selection elsewhere.
struct GroupTest
{
    const char *suite;
    const char *name;
};
static const GroupTest g_inventory[] = {{"Triton_HipResize", "Accuracy"},
                                        {"Triton_HipCvtColor", "Accuracy"},
                                        {"Triton_HipWarpAffine", "Accuracy"},
                                        {"Triton_HipRefusals", "NoFallback"},
                                        {"Triton_HipRuntime", "AsyncAndCapture"}};
static std::string g_filter = "*";

static bool globMatches(const std::string &pattern, const std::string &value)
{
    std::size_t p = 0, v = 0, star = std::string::npos, retry = 0;
    while (v < value.size())
    {
        if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == value[v]))
        {
            ++p;
            ++v;
        }
        else if (p < pattern.size() && pattern[p] == '*')
        {
            star = p++;
            retry = v;
        }
        else if (star != std::string::npos)
        {
            p = star + 1;
            v = ++retry;
        }
        else
            return false;
    }
    while (p < pattern.size() && pattern[p] == '*')
        ++p;
    return p == pattern.size();
}

static bool anyPatternMatches(const std::string &patterns, const std::string &name)
{
    std::size_t begin = 0;
    do
    {
        const std::size_t end = patterns.find(':', begin);
        if (globMatches(patterns.substr(begin, end == std::string::npos ? end : end - begin), name))
            return true;
        if (end == std::string::npos)
            break;
        begin = end + 1;
    } while (begin <= patterns.size());
    return false;
}

static bool groupSelected(const GroupTest &group)
{
    const std::size_t minus = g_filter.find('-');
    std::string positive = g_filter.substr(0, minus);
    if (minus != std::string::npos && positive.empty())
        positive = "*";
    const std::string name = std::string(group.suite) + "." + group.name;
    return anyPatternMatches(positive, name) &&
           (minus == std::string::npos || !anyPatternMatches(g_filter.substr(minus + 1), name));
}

static bool selected(const char *suite)
{
    for (const auto &group : g_inventory)
        if (std::string(group.suite) == suite)
            return groupSelected(group);
    return false;
}

static bool writeXml(const std::string &path)
{
    FILE *f = std::fopen(path.c_str(), "w");
    if (!f)
        return false;
    std::size_t groups = 0, failedGroups = 0;
    for (const auto &group : g_inventory)
        if (groupSelected(group))
        {
            ++groups;
            bool failed = false, ran = false;
            for (const auto &result : g_results)
                if (result.suite == group.suite)
                {
                    ran = true;
                    failed |= !result.failure.empty();
                }
            failedGroups += failed || !ran;
        }
    std::fprintf(
        f,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<testsuites tests=\"%zu\" failures=\"%zu\" checks=\"%zu\" check_failures=\"%d\">\n",
        groups, failedGroups, g_results.size(), g_fail);
    for (const auto &group : g_inventory)
        if (groupSelected(group))
        {
            std::size_t checks = 0, failures = 0;
            std::string details;
            for (const auto &result : g_results)
                if (result.suite == group.suite)
                {
                    ++checks;
                    if (!result.failure.empty())
                    {
                        ++failures;
                        details += result.failure + "\n";
                    }
                }
            if (!checks)
            {
                ++failures;
                details = "selected group executed no checks";
            }
            std::fprintf(
                f,
                "  <testsuite name=\"%s\" tests=\"1\" failures=\"%d\" checks=\"%zu\">\n"
                "    <testcase classname=\"%s\" name=\"%s\" status=\"run\" checks=\"%zu\" check_failures=\"%zu\"",
                group.suite, failures ? 1 : 0, checks, group.suite, group.name, checks, failures);
            if (failures)
                std::fprintf(f, ">\n      <failure message=\"%s\"/>\n    </testcase>\n",
                             xmlEscape(details).c_str());
            else
                std::fprintf(f, "/>\n");
            std::fprintf(f, "  </testsuite>\n");
        }
    std::fprintf(f, "</testsuites>\n");
    return std::fclose(f) == 0;
}

// ---------------------------------------------------------------- bounds (owned elsewhere)

static double resizeF32Atol(int interp)
{
    return triton_validation::numeric::resizeF32Atol(interp);
}

static double warpF32Tol(double r)
{
    return triton_validation::numeric::warpF32Tolerance(r);
}

// ------------------------------------------------------------------- device images

static cv::Mat randomImage(cv::Size sz, int type, uint64 seed)
{
    cv::RNG rng(seed);
    cv::Mat m(sz, type);
    const int d = CV_MAT_DEPTH(type);
    if (d == CV_8U)
        rng.fill(m, cv::RNG::UNIFORM, 0, 256);
    else if (d == CV_16U)
        rng.fill(m, cv::RNG::UNIFORM, 0, 65536);
    else
        rng.fill(m, cv::RNG::UNIFORM, -2.0, 2.0);
    return m;
}

// A device ROI inside a padded parent, so an out-of-bounds write is detectable: `pad` pixels on
// every side, filled with a sentinel byte that no case writes.
struct DevImg
{
    void *base = nullptr;
    std::size_t step = 0, bytes = 0;
    int rows = 0, cols = 0, type = 0, pad = 0;
    tr::DeviceImage view() const
    {
        tr::DeviceImage v;
        v.ptr = reinterpret_cast<std::uintptr_t>(base) + pad * step + pad * CV_ELEM_SIZE(type);
        v.rows = rows;
        v.cols = cols;
        v.type = type;
        v.step = step;
        return v;
    }
    ~DevImg()
    {
        if (base)
            (void)hipFree(base);
    }
};

static void alloc(DevImg &d, int rows, int cols, int type, int pad, unsigned char fill)
{
    d.rows = rows;
    d.cols = cols;
    d.type = type;
    d.pad = pad;
    d.step = static_cast<std::size_t>(cols + 2 * pad) * CV_ELEM_SIZE(type) + 64; // non-dense pitch
    d.bytes = d.step * static_cast<std::size_t>(rows + 2 * pad);
    HIPOK(hipMalloc(&d.base, d.bytes));
    HIPOK(hipMemset(d.base, fill, d.bytes));
}

static void upload(DevImg &d, const cv::Mat &m)
{
    const tr::DeviceImage v = d.view();
    HIPOK(hipMemcpy2D(reinterpret_cast<void *>(v.ptr), v.step, m.data, m.step[0],
                      static_cast<std::size_t>(m.cols) * m.elemSize(), m.rows,
                      hipMemcpyHostToDevice));
}

static cv::Mat downloadParent(const DevImg &d)
{
    cv::Mat whole(d.rows + 2 * d.pad, static_cast<int>(d.step), CV_8U);
    HIPOK(hipMemcpy(whole.data, d.base, d.bytes, hipMemcpyDeviceToHost));
    return whole;
}

static cv::Mat roiOf(const cv::Mat &parent, const DevImg &d)
{
    const int es = CV_ELEM_SIZE(d.type);
    cv::Mat out(d.rows, d.cols, d.type);
    for (int y = 0; y < d.rows; ++y)
        std::memcpy(out.ptr(y), parent.ptr(y + d.pad) + d.pad * es,
                    static_cast<std::size_t>(d.cols) * es);
    return out;
}

// True when every byte of the parent OUTSIDE the ROI still holds `fill`.
static bool untouchedOutside(const cv::Mat &parent, const DevImg &d, unsigned char fill)
{
    const int es = CV_ELEM_SIZE(d.type);
    for (int y = 0; y < parent.rows; ++y)
        for (int x = 0; x < static_cast<int>(d.step); ++x)
        {
            const bool inside =
                y >= d.pad && y < d.pad + d.rows && x >= d.pad * es && x < (d.pad + d.cols) * es;
            if (!inside && parent.at<unsigned char>(y, x) != fill)
                return false;
        }
    return true;
}

// Elementwise comparison: exact for integer depths, `tol(ref)` for CV_32F (NaN matches NaN).
static bool compare(const cv::Mat &got, const cv::Mat &want,
                    const std::function<double(double)> &tol, std::string &detail)
{
    if (got.size() != want.size() || got.type() != want.type())
    {
        detail = "shape/type";
        return false;
    }
    const cv::Mat g = got.reshape(1), w = want.reshape(1);
    long bad = 0;
    double worst = 0;
    for (int y = 0; y < g.rows; ++y)
        for (int x = 0; x < g.cols; ++x)
        {
            double a, b;
            switch (g.depth())
            {
            case CV_8U:
                a = g.at<uchar>(y, x);
                b = w.at<uchar>(y, x);
                break;
            case CV_16U:
                a = g.at<ushort>(y, x);
                b = w.at<ushort>(y, x);
                break;
            default:
                a = g.at<float>(y, x);
                b = w.at<float>(y, x);
                break;
            }
            if (std::isnan(a) && std::isnan(b))
                continue;
            const double d = std::abs(a - b);
            const double lim = g.depth() == CV_32F ? tol(b) : 0.0;
            if (!(d <= lim))
            {
                ++bad;
                worst = std::max(worst, d);
            }
        }
    if (bad)
    {
        detail = cv::format("%ld elements out of bound, worst |diff| %.3g", bad, worst);
        return false;
    }
    return true;
}

static void sync()
{
    HIPOK(hipDeviceSynchronize());
}

// --------------------------------------------------------------------------- resize

static cv::Mat resizeOracle(const cv::Mat &src, cv::Size dsize, int interp)
{
    const ref::Size2 z = ref::resolve_size(src.rows, src.cols, dsize.width, dsize.height, 0, 0);
    cv::Mat dst(z.dst_h, z.dst_w, src.type());
    ref::ImageRef s{src.data,       src.rows,    src.cols,
                    src.channels(), src.step[0], src.depth() == CV_32F};
    ref::ImageRef d{dst.data,       dst.rows,    dst.cols,
                    dst.channels(), dst.step[0], src.depth() == CV_32F};
    ref::resize(s, d, z.inv_y, z.inv_x, interp);
    return dst;
}

// All canonical resize routes are served; the oracle classifier still identifies each route.
static bool servedRoute(ref::Route r)
{
    return r == ref::ROUTE_COPY || r == ref::ROUTE_NEAREST || r == ref::ROUTE_LINEAR ||
           r == ref::ROUTE_CUBIC || r == ref::ROUTE_AREA_FAST || r == ref::ROUTE_AREA;
}

static void resizeCase(cv::Size ssz, cv::Size dsz, int type, int interp, hipStream_t stream,
                       uint64 seed)
{
    const unsigned char fill = 0xA5;
    const cv::Mat host = randomImage(ssz, type, seed);
    DevImg s, d;
    alloc(s, ssz.height, ssz.width, type, seed % 2 ? 4 : 2, 0);
    alloc(d, dsz.height, dsz.width, type, seed % 2 ? 4 : 3, fill);
    upload(s, host);
    const ref::Size2 z = ref::resolve_size(ssz.height, ssz.width, dsz.width, dsz.height, 0, 0);
    const ref::Route route =
        ref::route(ssz.height, ssz.width, z.dst_h, z.dst_w, z.inv_y, z.inv_x, interp).kind;
    const std::string name =
        cv::format("resize %dx%d->%dx%d %s interp=%d route=%s", ssz.width, ssz.height, dsz.width,
                   dsz.height, cv::typeToString(type).c_str(), interp, ref::route_name(route));
    if (!servedRoute(route))
    {
        int code = 0;
        try
        {
            tr::resize(s.view(), d.view(), dsz, interp, 0,
                       reinterpret_cast<std::uintptr_t>(stream));
            HIPOK(hipStreamSynchronize(stream));
        }
        catch (const cv::Exception &e)
        {
            code = e.code;
        }
        const cv::Mat parent = downloadParent(d);
        bool untouched = true;
        for (int y = 0; y < parent.rows && untouched; ++y)
            for (int x = 0; x < parent.cols; ++x)
                if (parent.at<unsigned char>(y, x) != fill)
                {
                    untouched = false;
                    break;
                }
        report(code == cv::Error::StsUnsupportedFormat && untouched,
               name + cv::format(": refused with code %d (want %d)%s", code,
                                 cv::Error::StsUnsupportedFormat,
                                 untouched ? "" : ", but dst was written"));
        return;
    }
    try
    {
        tr::resize(s.view(), d.view(), dsz, interp, 0, reinterpret_cast<std::uintptr_t>(stream));
        HIPOK(hipStreamSynchronize(stream));
    }
    catch (const cv::Exception &e)
    {
        report(false, name + ": " + e.what());
        return;
    }
    const cv::Mat parent = downloadParent(d);
    std::string why;
    const bool ok = compare(
        roiOf(parent, d), resizeOracle(host, dsz, interp),
        [&](double)
        {
            return resizeF32Atol(interp);
        },
        why);
    const bool clean = untouchedOutside(parent, d, fill);
    report(ok && clean, name + (ok ? "" : ": " + why) + (clean ? "" : ": wrote outside dst"));
}

// ------------------------------------------------------------------------- cvtColor

static void cvtCase(cv::Size sz, const cv_triton_color::CodeSpec &spec, int depth,
                    hipStream_t stream, uint64 seed, bool boundaries = false, int sourcePad = 1,
                    int destinationPad = 2)
{
    const unsigned char fill = 0x5A;
    const int sourceHeight = spec.op == cv_triton_color::Op::Yuv420 ? sz.height / 2 * 3 : sz.height;
    cv::Mat host = randomImage({sz.width, sourceHeight}, CV_MAKETYPE(depth, spec.scn), seed);
    if (boundaries)
    {
        cv::Mat flat = host.reshape(1);
        const double values[] = {0, 1, 2, 15, 16, 127, 128, 235, 240, 254, 255, 65535};
        for (int y = 0; y < flat.rows; ++y)
            for (int x = 0; x < flat.cols; ++x)
            {
                const double value = values[(x + 7 * y) % 12];
                if (depth == CV_8U)
                    flat.at<uchar>(y, x) = cv::saturate_cast<uchar>(value);
                else if (depth == CV_16U)
                    flat.at<ushort>(y, x) = static_cast<ushort>(value);
                else
                    flat.at<float>(y, x) = static_cast<float>((x % 3 == 0 ? -1 : 1) * value);
            }
    }
    if (depth == CV_32F && !boundaries)
        for (int y = 0; y < host.rows; ++y)
            for (int x = 0; x < host.cols; x += 3)
                for (int c = 0; c < spec.scn; ++c)
                    host.ptr<float>(y)[x * spec.scn + c] *= 32767.5f;
    DevImg src, dst;
    alloc(src, sourceHeight, sz.width, host.type(), sourcePad, 0xC3);
    alloc(dst, sz.height, sz.width, CV_MAKETYPE(depth, spec.dcn), destinationPad, fill);
    upload(src, host);
    const std::string name = cv::format("cvtColor %s depth=%d %dx%d%s", spec.name, depth, sz.width,
                                        sz.height, boundaries ? " boundaries" : "");
    try
    {
        tr::cvtColor(src.view(), dst.view(), spec.code, 0,
                     reinterpret_cast<std::uintptr_t>(stream));
        HIPOK(hipStreamSynchronize(stream));
    }
    catch (const cv::Exception &e)
    {
        report(false, name + ": " + e.what());
        return;
    }
    cv::Mat want;
    cv::cvtColor(host, want, spec.code);
    const cv::Mat parent = downloadParent(dst), got = roiOf(parent, dst);
    std::string why;
    bool ok = true;
    if (depth != CV_32F || spec.op == cv_triton_color::Op::Permute)
        ok = compare(
            got, want,
            [](double)
            {
                return 0.0;
            },
            why);
    else
    {
        // Scale is local to the pixel's input channels, never the image maximum.
        for (int y = 0; y < got.rows; ++y)
            for (int x = 0; x < got.cols; ++x)
            {
                double scale = 1;
                for (int c = 0; c < spec.scn; ++c)
                    scale = std::max(
                        scale, std::abs(static_cast<double>(host.ptr<float>(y)[x * spec.scn + c])));
                for (int c = 0; c < spec.dcn; ++c)
                {
                    const double a = got.ptr<float>(y)[x * spec.dcn + c],
                                 b = want.ptr<float>(y)[x * spec.dcn + c];
                    if (!std::isfinite(a) ||
                        std::abs(a - b) > triton_validation::numeric::cvtF32Tolerance(
                                              b, scale, cv_triton_color::float_cpu_separate4()))
                    {
                        ok = false;
                        why = cv::format("pixel(%d,%d,%d) got=%.12g ref=%.12g", y, x, c, a, b);
                    }
                }
            }
    }
    const cv::Mat sourceParent = downloadParent(src);
    const bool sourceClean = cv::norm(roiOf(sourceParent, src), host, cv::NORM_INF) == 0 &&
                             untouchedOutside(sourceParent, src, 0xC3);
    const bool clean = untouchedOutside(parent, dst, fill);
    report(ok && clean && sourceClean, name + (ok ? "" : ": " + why) +
                                           (clean ? "" : ": wrote outside dst") +
                                           (sourceClean ? "" : ": modified src"));
}

static void cvtCoverage(hipStream_t stream)
{
    uint64 seed = 10001;
    int rows8 = 0, rows16 = 0, rows32 = 0;
    for (const auto &spec : cv_triton_color::all())
        for (int depth : {CV_8U, CV_16U, CV_32F})
        {
            using cv_triton_color::Op;
            const bool word = spec.op == Op::Bayer || spec.op == Op::Gray || spec.op == Op::Yuv ||
                              spec.op == Op::YCrCb || (spec.op == Op::Permute && spec.scn == 1);
            const bool floating = spec.op == Op::Gray || spec.op == Op::Yuv ||
                                  spec.op == Op::YCrCb || (spec.op == Op::Permute && spec.scn != 2);
            if ((depth == CV_16U && !word) || (depth == CV_32F && !floating))
                continue;
            if (depth == CV_8U)
                ++rows8;
            else if (depth == CV_16U)
                ++rows16;
            else
                ++rows32;
            const std::vector<cv::Size> sizes =
                spec.op == Op::Yuv420
                    ? std::vector<cv::Size>{{2, 2}, {14, 6}, {16, 4}, {18, 8}, {258, 6}}
                : spec.op == Op::Yuv422
                    ? std::vector<cv::Size>{{2, 1}, {14, 3}, {16, 5}, {18, 7}, {258, 3}}
                : spec.op == Op::Bayer
                    ? std::vector<cv::Size>{{1, 1}, {2, 2}, {3, 3}, {7, 5}, {258, 9}}
                    : std::vector<cv::Size>{{1, 1}, {7, 3}, {16, 5}, {17, 7}, {257, 3}};
            for (const auto &size : sizes)
                cvtCase(size, spec, depth, stream, seed++);
            cvtCase(sizes.back(), spec, depth, stream, seed++, true);
        }
    report(rows8 == 84 && rows16 == 20 && rows32 == 20,
           cv::format("cvtColor table coverage: %d uint8 / %d uint16 / %d float32 code-depth rows",
                      rows8, rows16, rows32));
}

// ----------------------------------------------------------------------- warpAffine

static cv::Mat warpReference(const cv::Mat &src, const cv::Mat &M, cv::Size dsize, int flags,
                             int border, const cv::Scalar &bv)
{
    cv::Mat out;
    if ((flags & cv::INTER_MAX) == cv::INTER_CUBIC)
    {
        cv::warpAffine(src, out, M, dsize, flags, border, bv);
        return out;
    }
    cv::Mat wide;
    const int wide_w = ((dsize.width + 128 + 127) / 128) * 128;
    cv::warpAffine(src, wide, M, cv::Size(wide_w, dsize.height), flags, border, bv);
    return wide(cv::Rect(0, 0, dsize.width, dsize.height)).clone();
}

static void warpCase(cv::Size ssz, cv::Size dsz, int type, int flags, int border, const double m[6],
                     const cv::Scalar &bv, hipStream_t stream, uint64 seed)
{
    const unsigned char fill = 0xC3;
    const cv::Mat host = randomImage(ssz, type, seed);
    DevImg s, d;
    alloc(s, ssz.height, ssz.width, type, 2, 0);
    alloc(d, dsz.height, dsz.width, type, 2, fill);
    upload(s, host);
    const std::string name =
        cv::format("warpAffine %dx%d->%dx%d %s flags=%d border=%d", ssz.width, ssz.height,
                   dsz.width, dsz.height, cv::typeToString(type).c_str(), flags, border);
    try
    {
        tr::warpAffine(s.view(), d.view(), m, flags, border, bv, 0,
                       reinterpret_cast<std::uintptr_t>(stream));
        HIPOK(hipStreamSynchronize(stream));
    }
    catch (const cv::Exception &e)
    {
        report(false, name + ": " + e.what());
        return;
    }
    const cv::Mat M(2, 3, CV_64F, const_cast<double *>(m));
    const cv::Mat parent = downloadParent(d);
    std::string why;
    const bool ok =
        compare(roiOf(parent, d), warpReference(host, M, dsz, flags, border, bv), warpF32Tol, why);
    const bool clean = untouchedOutside(parent, d, fill);
    report(ok && clean, name + (ok ? "" : ": " + why) + (clean ? "" : ": wrote outside dst"));
}

// TRANSPARENT reads the old destination for every missing tap. The selected CPU
// vector-path reference must therefore start with the same ROI contents as the
// device target; padding an empty destination would read uninitialised pixels.
static cv::Mat warpInitializedReference(const cv::Mat &src, const double m[6],
                                        const cv::Mat &initial, int flags, int border)
{
    const int wide_w = ((initial.cols + 128 + 127) / 128) * 128;
    cv::Mat wide = randomImage({wide_w, initial.rows}, initial.type(), 30001);
    initial.copyTo(wide(cv::Rect(0, 0, initial.cols, initial.rows)));
    const cv::Mat M(2, 3, CV_64F, const_cast<double *>(m));
    cv::warpAffine(src, wide, M, wide.size(), flags, border);
    return wide(cv::Rect(0, 0, initial.cols, initial.rows)).clone();
}

static void warpTransparentCase(cv::Size size, int type, int flags, const double m[6], int geometry,
                                hipStream_t stream, uint64 seed)
{
    const unsigned char fill = 0xC3;
    const cv::Mat host = randomImage(size, type, seed);
    const cv::Mat initial = randomImage(size, type, seed + 1);
    DevImg s, d;
    alloc(s, size.height, size.width, type, 2, 0);
    alloc(d, size.height, size.width, type, 3, fill);
    upload(s, host);
    upload(d, initial);
    const std::string name =
        cv::format("warpAffine transparent %dx%d %s flags=%d geometry=%d", size.width, size.height,
                   cv::typeToString(type).c_str(), flags, geometry);
    try
    {
        tr::warpAffine(s.view(), d.view(), m, flags, cv::BORDER_TRANSPARENT, cv::Scalar(), 0,
                       reinterpret_cast<std::uintptr_t>(stream));
        HIPOK(hipStreamSynchronize(stream));
    }
    catch (const cv::Exception &e)
    {
        report(false, name + ": " + e.what());
        return;
    }
    const cv::Mat parent = downloadParent(d), got = roiOf(parent, d);
    const cv::Mat want = warpInitializedReference(host, m, initial, flags, cv::BORDER_TRANSPARENT);
    std::string why;
    const bool ok = compare(got, want, warpF32Tol, why);
    const bool clean = untouchedOutside(parent, d, fill);
    // The fully outside geometry must preserve every old destination bit.
    const bool preserved = geometry != 2 || cv::norm(got, initial, cv::NORM_INF) == 0.0;
    report(ok && clean && preserved, name + (ok ? "" : ": " + why) +
                                         (clean ? "" : ": wrote outside dst") +
                                         (preserved ? "" : ": outside dst changed"));
}

static void warpAreaAliasCase(int type, int border, int inverse, hipStream_t stream, uint64 seed)
{
    const cv::Size size(37, 53);
    const unsigned char fill = 0xC3;
    const cv::Mat host = randomImage(size, type, seed);
    const cv::Mat initial = randomImage(size, type, seed + 1);
    const double m[6] = {0.9, -0.43589, 4, 0.43589, 0.9, -2};
    DevImg s, linear, area;
    alloc(s, size.height, size.width, type, 2, 0);
    alloc(linear, size.height, size.width, type, 3, fill);
    alloc(area, size.height, size.width, type, 3, fill);
    upload(s, host);
    upload(linear, initial);
    upload(area, initial);
    const std::string name = cv::format("warpAffine AREA aliases LINEAR %s border=%d inverse=%d",
                                        cv::typeToString(type).c_str(), border, inverse);
    try
    {
        tr::warpAffine(s.view(), linear.view(), m, cv::INTER_LINEAR | inverse, border, cv::Scalar(),
                       0, reinterpret_cast<std::uintptr_t>(stream));
        tr::warpAffine(s.view(), area.view(), m, cv::INTER_AREA | inverse, border, cv::Scalar(), 0,
                       reinterpret_cast<std::uintptr_t>(stream));
        HIPOK(hipStreamSynchronize(stream));
    }
    catch (const cv::Exception &e)
    {
        report(false, name + ": " + e.what());
        return;
    }
    const cv::Mat linearParent = downloadParent(linear), areaParent = downloadParent(area);
    const cv::Mat gotLinear = roiOf(linearParent, linear), gotArea = roiOf(areaParent, area);
    const cv::Mat cpuLinear =
        warpInitializedReference(host, m, initial, cv::INTER_LINEAR | inverse, border);
    const cv::Mat cpuArea =
        warpInitializedReference(host, m, initial, cv::INTER_AREA | inverse, border);
    std::string why;
    const bool alias = cv::norm(gotArea, gotLinear, cv::NORM_INF) == 0.0;
    const bool cpuAlias = cv::norm(cpuArea, cpuLinear, cv::NORM_INF) == 0.0;
    const bool ok = compare(gotArea, cpuArea, warpF32Tol, why);
    const bool clean =
        untouchedOutside(linearParent, linear, fill) && untouchedOutside(areaParent, area, fill);
    report(alias && cpuAlias && ok && clean, name + (alias ? "" : ": device alias differs") +
                                                 (cpuAlias ? "" : ": CPU alias differs") +
                                                 (ok ? "" : ": " + why) +
                                                 (clean ? "" : ": wrote outside dst"));
}

// --------------------------------------------------------------------------- refusals

template <class F> static void refusal(const std::string &name, int expect_code, F call)
{
    try
    {
        call();
        sync();
        report(false, name + ": accepted, expected a refusal");
    }
    catch (const cv::Exception &e)
    {
        report(e.code == expect_code,
               name + cv::format(" -> code %d (want %d)", e.code, expect_code));
    }
}

// Shared-runtime integration: resize and cvtColor in one process, with real async
// uploads, LRU eviction and a captured graph replayed after input mutation.
static void runtimeChecks(hipStream_t stream)
{
    namespace detail = cv::triton::detail;
    const cv::Mat host = randomImage({96, 64}, CV_8UC1, 991);
    DevImg src, a, b;
    alloc(src, 64, 96, CV_8UC1, 4, 0);
    alloc(a, 47, 63, CV_8UC1, 4, 0xA5);
    alloc(b, 47, 63, CV_8UC1, 4, 0xA5);
    upload(src, host);
    hipStream_t other = nullptr;
    HIPOK(hipStreamCreate(&other));
    detail::setPoisonNewTables(true);
    detail::setCacheLimits(64, 256);
    const auto before = detail::runtimeStats();
    tr::resize(src.view(), a.view(), {63, 47}, cv::INTER_AREA, 0,
               reinterpret_cast<std::uintptr_t>(stream));
    tr::resize(src.view(), b.view(), {63, 47}, cv::INTER_AREA, 0,
               reinterpret_cast<std::uintptr_t>(other));
    HIPOK(hipDeviceSynchronize());
    std::string why;
    const cv::Mat reference = resizeOracle(host, {63, 47}, cv::INTER_AREA);
    const auto after = detail::runtimeStats();
    report(after.cross_stream_waits > before.cross_stream_waits &&
               compare(
                   roiOf(downloadParent(a), a), reference,
                   [&](double)
                   {
                       return 0.0;
                   },
                   why) &&
               compare(
                   roiOf(downloadParent(b), b), reference,
                   [&](double)
                   {
                       return 0.0;
                   },
                   why),
           "fractional AREA tables obey upload-to-consumer dependencies on two streams");
    hipGraph_t graph = nullptr;
    hipGraphExec_t executable = nullptr;
    HIPOK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    tr::resize(src.view(), a.view(), {63, 47}, cv::INTER_AREA, 0,
               reinterpret_cast<std::uintptr_t>(stream));
    HIPOK(hipStreamEndCapture(stream, &graph));
    HIPOK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    detail::setCacheLimits(0, 0);
    const cv::Mat changed = randomImage({96, 64}, CV_8UC1, 992);
    upload(src, changed);
    HIPOK(hipMemset(b.base, 0xA5, b.bytes));
    HIPOK(hipMemset(a.base, 0xA5, a.bytes));
    HIPOK(hipGraphLaunch(executable, stream));
    HIPOK(hipStreamSynchronize(stream));
    const cv::Mat parent = downloadParent(a);
    const bool correct = compare(
        roiOf(parent, a), resizeOracle(changed, {63, 47}, cv::INTER_AREA),
        [&](double)
        {
            return 0.0;
        },
        why);
    report(correct && untouchedOutside(parent, a, 0xA5),
           "captured fractional AREA executes on new input after module/table LRU eviction");
    HIPOK(hipGraphExecDestroy(executable));
    HIPOK(hipGraphDestroy(graph));
    cv::triton::releaseGraphResources();
    detail::reapCompleted();
    report(detail::runtimeStats().capture_pin_holds == 0,
           "public releaseGraphResources releases pins after graph destruction");
    detail::setCacheLimits(64, 256);
    detail::setPoisonNewTables(false);
    HIPOK(hipStreamDestroy(other));
}
#include "test_hip_cvt_runtime.hpp"
#include "test_hip_ownership.hpp"
int main(int argc, char **argv)
{
    std::string xml;
    bool list = false;
    if (const char *filter = std::getenv("GTEST_FILTER"))
        g_filter = filter;
    if (std::getenv("GTEST_SHARD_INDEX") || std::getenv("GTEST_TOTAL_SHARDS"))
    {
        std::fprintf(stderr, "standalone HIP group tests do not support sharding\n");
        return 2;
    }
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a.rfind("--gtest_output=xml:", 0) == 0 && a.size() > 19)
            xml = a.substr(19);
        else if (a.rfind("--gtest_filter=", 0) == 0)
            g_filter = a.substr(15);
        else if (a == "--gtest_list_tests")
            list = true;
        else
        {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 2;
        }
    }
    std::size_t groupCount = 0;
    for (const auto &group : g_inventory)
        if (groupSelected(group))
            ++groupCount;
    if (!groupCount)
    {
        std::fprintf(stderr, "filter selected zero registered HIP groups: %s\n", g_filter.c_str());
        return 2;
    }
    if (list)
    {
        for (const auto &group : g_inventory)
            if (groupSelected(group))
                std::printf("%s.\n  %s\n", group.suite, group.name);
        return 0;
    }
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev < 1 || !tr::haveDevice(0))
    {
        std::printf("SKIP: no usable hip device 0 (haveDevice=%d)\n", tr::haveDevice(0) ? 1 : 0);
        return 77; // the conventional "skipped" exit code, distinct from pass (0) and fail
    }
    HIPOK(hipSetDevice(0));
    hipStream_t stream = nullptr;
    HIPOK(hipStreamCreate(&stream));

    // Each section starts from its own fixed seed, so a filtered run gives every case the
    // same input a full run does.
    uint64 seed = 0;
    g_suite = "Triton_HipResize";
    if (selected("Triton_HipResize"))
    {
        seed = 1;
        std::printf("== resize vs the portable CPU oracle\n");
        const int rtypes[] = {CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4};
        const int interps[] = {cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC,
                               cv::INTER_AREA};
        // Sizes chosen so every served route is reached and the refused ones are exercised too:
        // exact 2x / integer ratios (area_fast), non-integer up and down (linear; fractional area for
        // INTER_AREA), identity (copy), a 1-pixel axis, and a full HD frame.
        const cv::Size rsizes[][2] = {
            {{64, 48}, {32, 24}},  {{97, 61}, {131, 53}},      {{40, 40}, {40, 40}},
            {{1, 7}, {5, 3}},      {{1920, 1080}, {640, 360}}, {{129, 96}, {43, 32}},
            {{100, 80}, {37, 29}}, {{512, 514}, {256, 257}},   {{514, 516}, {257, 258}}};
        for (int t : rtypes)
            for (int i : interps)
                for (const auto &sz : rsizes)
                    resizeCase(sz[0], sz[1], t, i, stream, seed++);
    }

    g_suite = "Triton_HipCvtColor";
    if (selected("Triton_HipCvtColor"))
    {
        std::printf("== cvtColor all supported code/depth rows vs linked CPU reference\n");
        cvtCoverage(stream);
        // Aligned pitched ROIs exercise the HIP vector path at its size thresholds;
        // odd widths and an unaligned source exercise the generic fallback.
        for (cv::Size size :
             {cv::Size(1024, 511), cv::Size(1024, 512), cv::Size(1028, 513), cv::Size(1025, 513)})
            cvtCase(size, *cv_triton_color::find(cv::COLOR_BGR2GRAY), CV_8U, stream, 70901, false,
                    4, 4);
        cvtCase({1024, 512}, *cv_triton_color::find(cv::COLOR_BGR2GRAY), CV_8U, stream, 70903,
                false, 1, 4);
    }

    g_suite = "Triton_HipWarpAffine";
    if (selected("Triton_HipWarpAffine"))
    {
        seed = 20001;
        std::printf("== warpAffine vs cv::warpAffine's selected reference path\n");
        const double rot[6] = {0.92, -0.38, 12.5, 0.38, 0.92, -6.25};
        const double scl[6] = {1.1, 0.0, -1.5, 0.0, 1.1, -1.0};
        const double shear[6] = {1.0, 1.9, -20.0, 0.4, 1.0, -10.0};
        const double *mats[] = {rot, scl, shear};
        const int wtypes[] = {CV_8UC1,  CV_8UC3,  CV_8UC4,  CV_16UC1, CV_16UC3,
                              CV_16UC4, CV_32FC1, CV_32FC3, CV_32FC4};
        const int wmodes[] = {cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC};
        const int borders[] = {cv::BORDER_CONSTANT, cv::BORDER_REPLICATE, cv::BORDER_REFLECT,
                               cv::BORDER_REFLECT_101, cv::BORDER_WRAP};
        for (int t : wtypes)
            for (int mode : wmodes)
                for (const double *m : mats)
                    for (int b : borders)
                        for (int inverse : {0, static_cast<int>(cv::WARP_INVERSE_MAP)})
                            warpCase({53, 37}, {61, 43}, t, mode | inverse, b, m,
                                     cv::Scalar(7, 11, 13, 17), stream, seed++);
        const double outside[6] = {1, 0, 10000, 0, 1, 10000};
        for (int t : wtypes)
            for (int mode : wmodes)
                warpCase({53, 37}, {61, 43}, t, mode | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT,
                         outside, cv::Scalar(-9.5, 255.5, 70000, 65535.6), stream, seed++);
        warpCase({53, 37}, {61, 43}, CV_8UC1, cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                 cv::BORDER_CONSTANT, rot, cv::Scalar(), stream, seed++);
        // CUBIC's four taps start at floor(sx)-1. These columns cover the
        // floor(sx)==-3 strip and cross sx==-2, where the last tap
        // first enters the source: ix0+3<0 is outside; ix0+3==0 still samples.
        const double cubicLeftBoundary[6] = {0.25, 0, -3.25, 0, 1, 2.25};
        for (int type : {CV_8UC1, CV_16UC1, CV_32FC1})
            warpCase({53, 37}, {7, 5}, type, cv::INTER_CUBIC | cv::WARP_INVERSE_MAP,
                     cv::BORDER_CONSTANT, cubicLeftBoundary, cv::Scalar(251.25), stream, seed++);

        // Check AREA aliasing and
        // transparent old-destination semantics for every supported depth/channel pair.
        seed = 30001;
        for (int t : wtypes)
            for (int border : {cv::BORDER_CONSTANT, cv::BORDER_REPLICATE, cv::BORDER_REFLECT,
                               cv::BORDER_REFLECT_101, cv::BORDER_WRAP, cv::BORDER_TRANSPARENT})
                for (int inverse : {0, static_cast<int>(cv::WARP_INVERSE_MAP)})
                    warpAreaAliasCase(t, border, inverse, stream, seed += 2);
        const cv::Size transparentSizes[] = {{128, 48}, {37, 53}, {1, 1}, {1, 9}, {7, 1}};
        for (int t : wtypes)
            for (const cv::Size &size : transparentSizes)
                for (int geometry = 0; geometry < 3; ++geometry)
                    for (int mode : {cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_AREA})
                        for (int inverse : {0, static_cast<int>(cv::WARP_INVERSE_MAP)})
                        {
                            // Half-pixel taps crossing an edge distinguish substitution of
                            // previous dst per missing tap from weight renormalisation.
                            double m[6] = {1, 0, 0.5, 0, 1, -0.5};
                            if (geometry == 1)
                            {
                                const double partial[6] = {0.9, 0.43589, -14, -0.43589, 0.9, 10};
                                std::copy(partial, partial + 6, m);
                            }
                            if (geometry == 2)
                            {
                                m[2] = size.width + 4;
                                m[5] = size.height + 4;
                            }
                            warpTransparentCase(size, t, mode | inverse, m, geometry, stream,
                                                seed += 2);
                        }
    }

    g_suite = "Triton_HipRefusals";
    if (selected("Triton_HipRefusals"))
    {
        std::printf("== refusals: documented error, nothing written\n");
        {
            DevImg s, d;
            alloc(s, 16, 16, CV_16UC1, 0, 0);
            alloc(d, 8, 8, CV_16UC1, 0, 0);
            refusal("resize CV_16U", cv::Error::StsUnsupportedFormat,
                    [&]
                    {
                        tr::resize(s.view(), d.view(), {8, 8}, cv::INTER_LINEAR, 0, 0);
                    });
            DevImg a, b;
            alloc(a, 16, 16, CV_8UC1, 0, 0);
            alloc(b, 8, 8, CV_8UC1, 0, 0);
            refusal("resize LANCZOS4", cv::Error::StsUnsupportedFormat,
                    [&]
                    {
                        tr::resize(a.view(), b.view(), {8, 8}, cv::INTER_LANCZOS4, 0, 0);
                    });
            refusal("resize in place", cv::Error::StsBadArg,
                    [&]
                    {
                        tr::resize(a.view(), a.view(), {16, 16}, cv::INTER_LINEAR, 0, 0);
                    });
            DevImg c3, c1;
            alloc(c3, 16, 16, CV_8UC3, 0, 0);
            alloc(c1, 16, 16, CV_8UC1, 0, 0);
            refusal("cvtColor HSV2BGR (float path, refused)", cv::Error::StsUnsupportedFormat,
                    [&]
                    {
                        tr::cvtColor(c3.view(), c3.view(), cv::COLOR_HSV2BGR, 0, 0);
                    });
            refusal("cvtColor wrong dst channels", cv::Error::StsBadArg,
                    [&]
                    {
                        tr::cvtColor(c3.view(), c3.view(), cv::COLOR_BGR2GRAY, 0, 0);
                    });
            const double id[6] = {1, 0, 0, 0, 1, 0};
            refusal("warpAffine LANCZOS4", cv::Error::StsUnsupportedFormat,
                    [&]
                    {
                        tr::warpAffine(a.view(), b.view(), id, cv::INTER_LANCZOS4,
                                       cv::BORDER_CONSTANT, cv::Scalar(), 0, 0);
                    });
            refusal("warpAffine in place", cv::Error::StsBadArg,
                    [&]
                    {
                        tr::warpAffine(a.view(), a.view(), id, cv::INTER_LINEAR,
                                       cv::BORDER_CONSTANT, cv::Scalar(), 0, 0);
                    });
            refusal("warpAffine unknown flags", cv::Error::StsUnsupportedFormat,
                    [&]
                    {
                        tr::warpAffine(a.view(), b.view(), id, cv::INTER_LINEAR | (1 << 12),
                                       cv::BORDER_REPLICATE, cv::Scalar(), 0, 0);
                    });
            refusal("warpAffine null matrix", cv::Error::StsBadArg,
                    [&]
                    {
                        tr::warpAffine(a.view(), b.view(), nullptr, cv::INTER_LINEAR,
                                       cv::BORDER_REPLICATE, cv::Scalar(), 0, 0);
                    });
            const double huge[6] = {1e100, 0, 0, 0, 1, 0};
            refusal("warpAffine nonrepresentable fp32 matrix", cv::Error::StsBadArg,
                    [&]
                    {
                        tr::warpAffine(a.view(), b.view(), huge,
                                       cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                                       cv::BORDER_REPLICATE, cv::Scalar(), 0, 0);
                    });
            const double overflow[6] = {1e200, 1e200, 0, 1e200, 1e200, 0};
            refusal("warpAffine nonfinite inverse", cv::Error::StsBadArg,
                    [&]
                    {
                        tr::warpAffine(a.view(), b.view(), overflow, cv::INTER_LINEAR,
                                       cv::BORDER_REPLICATE, cv::Scalar(), 0, 0);
                    });
            for (int depth : {CV_8U, CV_16U, CV_32F})
                for (int channels : {1, 3, 4})
                {
                    const int type = CV_MAKETYPE(depth, channels);
                    const cv::Mat initial = randomImage({17, 13}, type, 40001);
                    DevImg source, target;
                    alloc(source, 13, 17, type, 2, 0);
                    alloc(target, 13, 17, type, 3, 0xC3);
                    upload(source, initial);
                    upload(target, initial);
                    bool refused = false;
                    try
                    {
                        tr::warpAffine(source.view(), target.view(), id, cv::INTER_CUBIC,
                                       cv::BORDER_TRANSPARENT, cv::Scalar(), 0,
                                       reinterpret_cast<std::uintptr_t>(stream));
                    }
                    catch (const cv::Exception &e)
                    {
                        refused = e.code == cv::Error::StsUnsupportedFormat;
                    }
                    HIPOK(hipStreamSynchronize(stream));
                    const cv::Mat parent = downloadParent(target);
                    const bool unchanged =
                        cv::norm(roiOf(parent, target), initial, cv::NORM_INF) == 0.0 &&
                        untouchedOutside(parent, target, 0xC3);
                    report(refused && unchanged,
                           "warpAffine CUBIC+TRANSPARENT refuses without writing " +
                               cv::typeToString(type));
                }
        }
    }

    // Depth, channel and video-layout refusals preserve the existing destination.
    g_suite = "Triton_HipCvtColor";
    if (selected("Triton_HipCvtColor"))
    {
        struct BadCvt
        {
            int code, depth, scn, dcn, width, height;
        };
        const BadCvt bad[] = {{cv::COLOR_BGR2HSV, CV_32F, 3, 3, 17, 7},
                              {cv::COLOR_BGR2XYZ, CV_16U, 3, 3, 17, 7},
                              {cv::COLOR_BGR2Lab, CV_32F, 3, 3, 17, 7},
                              {cv::COLOR_BGR2BGRA, CV_16U, 3, 4, 17, 7},
                              {cv::COLOR_YUV2GRAY_UYVY, CV_32F, 2, 1, 17, 7},
                              {cv::COLOR_YUV2BGR_UYVY, CV_8U, 2, 3, 17, 7},
                              {cv::COLOR_YUV2BGR_NV12, CV_8U, 1, 3, 17, 6},
                              {cv::COLOR_YUV2BGR_NV21, CV_8U, 1, 3, 18, 7},
                              {cv::COLOR_BGR2GRAY, CV_8U, 1, 1, 17, 7},
                              {cv::COLOR_BGR2GRAY, CV_8U, 3, 4, 17, 7},
                              {cv::COLOR_HSV2BGR, CV_8U, 3, 3, 17, 7},
                              {cv::COLOR_LBGR2Luv, CV_8U, 3, 3, 17, 7},
                              {cv::COLOR_BGR2YUV_I420, CV_8U, 3, 1, 18, 6}};
        for (const auto &c : bad)
        {
            DevImg source, target;
            alloc(source, c.height, c.width, CV_MAKETYPE(c.depth, c.scn), 1, 0);
            const int targetHeight =
                c.code == cv::COLOR_YUV2BGR_NV12 || c.code == cv::COLOR_YUV2BGR_NV21
                    ? (c.height / 3) * 2
                    : c.height;
            alloc(target, targetHeight, c.width, CV_MAKETYPE(c.depth, c.dcn), 2, 0x5A);
            bool rejected = false;
            try
            {
                tr::cvtColor(source.view(), target.view(), c.code, 0,
                             reinterpret_cast<std::uintptr_t>(stream));
            }
            catch (const cv::Exception &e)
            {
                rejected =
                    e.code == cv::Error::StsBadArg || e.code == cv::Error::StsUnsupportedFormat;
            }
            HIPOK(hipStreamSynchronize(stream));
            const cv::Mat parent = downloadParent(target);
            bool preserved = true;
            for (size_t i = 0; i < parent.total(); ++i)
                preserved = preserved && parent.data[i] == 0x5A;
            report(rejected && preserved,
                   cv::format("cvtColor refused code=%d depth=%d C%d->C%d %dx%d without writing",
                              c.code, c.depth, c.scn, c.dcn, c.width, c.height));
        }
    }

    g_suite = "Triton_HipRuntime";
    if (selected("Triton_HipRuntime"))
    {
        runtimeChecks(stream);
        cvtRuntimeChecks(stream);
        ownershipChecks(stream);
    }
    HIPOK(hipStreamDestroy(stream));
    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    if (!xml.empty() && !writeXml(xml))
    {
        std::fprintf(stderr, "cannot write %s\n", xml.c_str());
        return 2;
    }
    if (g_results.empty())
    {
        std::fprintf(stderr, "selected groups executed zero checks\n");
        return 2;
    }
    for (const auto &group : g_inventory)
        if (groupSelected(group))
        {
            bool ran = false;
            for (const auto &result : g_results)
                ran |= result.suite == group.suite;
            if (!ran)
            {
                std::fprintf(stderr, "selected group executed no checks: %s.%s\n", group.suite,
                             group.name);
                return 2;
            }
        }
    return g_fail ? 1 : 0;
}
