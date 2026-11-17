// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_TRITON_VALIDATION_REFERENCE_HPP
#define OPENCV_TRITON_VALIDATION_REFERENCE_HPP
#include "opencv2/ts.hpp"
#include "opencv2/imgproc.hpp"
#include "../test/resize_reference.hpp"
#include "numerics.hpp"
#include "../src/runtime/cvtcolor.hpp"
#include <algorithm>
#include <cmath>
namespace opencv_test
{
namespace triton_test
{
namespace ref = cv_triton_resize;
inline Mat randomImage(Size sz, int type, uint64 seed)
{
    RNG rng(seed);
    Mat m(sz, type);
    if (CV_MAT_DEPTH(type) == CV_8U)
        rng.fill(m, RNG::UNIFORM, 0, 256);
    else if (CV_MAT_DEPTH(type) == CV_16U)
        // The full 16-bit range: a fill of 0..2 would pass every test while exercising none
        // of the range where a saturating store can go wrong.
        rng.fill(m, RNG::UNIFORM, 0, 65536);
    else
        rng.fill(m, RNG::UNIFORM, -2.0, 2.0);
    return m;
}

/** cvtColor uses this build's actual cv::cvtColor. Integer and bit-copy paths are exact;
 * float arithmetic uses the per-pixel input-channel-scale bound. This is not the
 * portable resize oracle and is not a promise about a different IPP/HAL/dispatch build.
 */
inline Mat cpuCvt(const Mat &src, int code)
{
    Mat dst;
    cv::cvtColor(src, dst, code);
    return dst;
}

/** Resize's answer: its portable CPU oracle, independent of this build's cv::resize.
 * warpAffine does not use this helper: test_warpaffine.cpp selects this build's padded
 * CPU vector path for NN/LINEAR/AREA and natural-width CPU call for CUBIC.
 */
inline Mat oracle(const Mat &src, Size dsize, double fx, double fy, int interp)
{
    CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
    const ref::Size2 z = ref::resolve_size(src.rows, src.cols, dsize.width, dsize.height, fx, fy);
    Mat dst(z.dst_h, z.dst_w, src.type());
    ref::ImageRef s{src.data,       src.rows,    src.cols,
                    src.channels(), src.step[0], src.depth() == CV_32F};
    ref::ImageRef d{dst.data,       dst.rows,    dst.cols,
                    dst.channels(), dst.step[0], src.depth() == CV_32F};
    ref::resize(s, d, z.inv_y, z.inv_x, interp);
    return dst;
}

inline double f32Atol(int interp) // Derived from the reference algorithm's float32 rounding
{
    return triton_validation::numeric::resizeF32Atol(interp);
}

/** CV_8U exact; CV_32F within the derived bound, NaN matched by NaN. */
inline ::testing::AssertionResult meetsContract(const Mat &got, const Mat &ref_, int interp)
{
    if (got.size() != ref_.size() || got.type() != ref_.type())
        return ::testing::AssertionFailure() << "size/type " << got.size() << "/" << got.type()
                                             << " vs " << ref_.size() << "/" << ref_.type();
    const int W = got.cols * got.channels();
    long long bad = 0;
    double worst = 0;
    for (int y = 0; y < got.rows; ++y)
    {
        if (got.depth() == CV_8U)
        {
            const uchar *a = got.ptr<uchar>(y);
            const uchar *b = ref_.ptr<uchar>(y);
            for (int x = 0; x < W; ++x)
                if (a[x] != b[x])
                {
                    ++bad;
                    worst = std::max(worst, std::abs(double(a[x]) - b[x]));
                }
        }
        else
        {
            const float *a = got.ptr<float>(y);
            const float *b = ref_.ptr<float>(y);
            for (int x = 0; x < W; ++x)
            {
                if (std::isnan(a[x]) && std::isnan(b[x]))
                    continue;
                const double d = std::abs(double(a[x]) - b[x]);
                if (!(d <= f32Atol(interp)))
                {
                    ++bad;
                    if (!(d <= worst))
                        worst = d;
                }
            }
        }
    }
    if (bad)
        return ::testing::AssertionFailure()
               << bad << " elements outside the accuracy bound, worst |diff| " << worst;
    return ::testing::AssertionSuccess();
}

// Selected CPU reference, bound to this linked build. CUBIC uses natural width;
// other modes add at least 128 columns and round up to a multiple of 128, then
// crop. The extra width keeps compared pixels away from scalar-tail sampling.
// This deliberately selects the vector-path result, not natural-width equivalence
// or a cross-ISA guarantee.
inline Mat warpReference(const Mat &src, const Mat &matrix, Size size, int flags, int border,
                         const Scalar &value = Scalar())
{
    Mat output;
    if ((flags & INTER_MAX) == INTER_CUBIC)
        cv::warpAffine(src, output, matrix, size, flags, border, value);
    else
    {
        const int width = ((size.width + 128 + 127) / 128) * 128;
        Mat wide;
        cv::warpAffine(src, wide, matrix, Size(width, size.height), flags, border, value);
        output = wide(Rect(0, 0, size.width, size.height)).clone();
    }
    return output;
}
inline double warpF32Tolerance(double value)
{
    return triton_validation::numeric::warpF32Tolerance(value);
}
inline double cvtF32Tolerance(double reference, double inputScale, bool absoluteCap = false)
{
    return triton_validation::numeric::cvtF32Tolerance(reference, inputScale, absoluteCap);
}
/** cvtColor's numeric owner, shared by accuracy/perf and device fixtures.
 * Exact paths retain the existing NORM_INF == 0 comparison; floating arithmetic
 * uses only the input channels of the corresponding pixel, never an image max.
 */
inline ::testing::AssertionResult meetsCvtContract(const Mat &got, const Mat &reference,
                                                   const Mat &input, bool arithmetic,
                                                   bool absoluteCap = false)
{
    if (got.size() != reference.size() || got.type() != reference.type())
        return ::testing::AssertionFailure() << "cvtColor output size/type mismatch";
    if (got.depth() != CV_32F || !arithmetic)
    {
        const double difference = cv::norm(got, reference, NORM_INF);
        if (difference == 0)
            return ::testing::AssertionSuccess();
        return ::testing::AssertionFailure() << "cvtColor exact path |diff| " << difference;
    }
    if (input.depth() != CV_32F || input.size() != got.size())
        return ::testing::AssertionFailure() << "cvtColor float input size/depth mismatch";
    long long bad = 0;
    double worst = 0;
    for (int y = 0; y < got.rows; ++y)
        for (int x = 0; x < got.cols; ++x)
        {
            double inputScale = 0;
            for (int c = 0; c < input.channels(); ++c)
            {
                const double value = input.ptr<float>(y)[x * input.channels() + c];
                if (!std::isfinite(value))
                    return ::testing::AssertionFailure()
                           << "non-finite cvtColor input at " << x << "," << y;
                inputScale = std::max(inputScale, std::abs(value));
            }
            for (int c = 0; c < got.channels(); ++c)
            {
                const double expected = reference.ptr<float>(y)[x * got.channels() + c];
                const double actual = got.ptr<float>(y)[x * got.channels() + c];
                if (!std::isfinite(expected) || !std::isfinite(actual))
                    return ::testing::AssertionFailure()
                           << "non-finite cvtColor result at " << x << "," << y;
                const double ratio = std::abs(actual - expected) /
                                     cvtF32Tolerance(expected, inputScale, absoluteCap);
                worst = std::max(worst, ratio);
                if (ratio > 1)
                    ++bad;
            }
        }
    if (bad)
        return ::testing::AssertionFailure()
               << bad << " cvtColor elements outside per-pixel bound, worst ratio " << worst;
    return ::testing::AssertionSuccess();
}

}
}
#endif
