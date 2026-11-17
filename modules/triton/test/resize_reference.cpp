// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Adapted from modules/imgproc/src/resize.cpp; retain its original license notice.
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
// Copyright (C) 2000-2008, 2017, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
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

// Test-only scalar resize evaluation. Preserve the OpenCV generic implementation
// operation order, including its vector-prefix/scalar-tail distinction.
#include "resize_reference.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#if defined(FLT_EVAL_METHOD) && FLT_EVAL_METHOD != 0
#error "the reference needs IEEE single-precision float evaluation (FLT_EVAL_METHOD == 0)"
#endif
namespace cv_triton_resize
{

namespace
{

constexpr int COEF_BITS = 11;
constexpr int COEF_SCALE = 1 << COEF_BITS;
// The 128-bit SIMD rounding order in the reference: VResizeCubic's uint8 vector loop covers
// (W / 8) * 8 elements, its float32 one (W / 4) * 4; the float32 2x2 AREA vector
// loop covers (dwidth1 / 4) * 4. These sizes are fixed independently of the host vector width.
constexpr int CUBIC_U8_VEC = 8;
constexpr int CUBIC_F32_VEC = 4;
constexpr int AREA2X2_F32_VEC = 4;

inline double rint_even(double v)
{
    return std::nearbyint(v);
}

inline uint8_t store_u8(float v)
{ // saturate_cast<uchar>(float)
    double r = rint_even(v);
    return static_cast<uint8_t>(std::max(0.0, std::min(255.0, r)));
}

template <typename T> inline const T *row_ptr(const ImageRef &im, int y)
{
    return reinterpret_cast<const T *>(static_cast<const char *>(im.data) +
                                       static_cast<std::size_t>(y) * im.step);
}
template <typename T> inline T *row_ptr_mut(const ImageRef &im, int y)
{
    return reinterpret_cast<T *>(static_cast<char *>(im.data) +
                                 static_cast<std::size_t>(y) * im.step);
}

} // namespace

// ------------------------------------------------------------------ evaluation

namespace
{

template <typename T> void eval_copy(const ImageRef &s, const ImageRef &d)
{
    const std::size_t bytes = static_cast<std::size_t>(s.cols) * s.channels * sizeof(T);
    for (int y = 0; y < s.rows; ++y)
        std::memcpy(row_ptr_mut<T>(d, y), row_ptr<T>(s, y), bytes);
}

template <typename T>
void eval_nearest(const ImageRef &s, const ImageRef &d, double inv_y, double inv_x)
{
    const double ifx = 1.0 / inv_x, ify = 1.0 / inv_y; // resizeNN: 1./fx, not src/dst
    const int cn = s.channels;
    std::vector<int> sx(d.cols);
    for (int x = 0; x < d.cols; ++x)
        sx[x] = std::min(static_cast<int>(std::floor(x * ifx)), s.cols - 1);
    for (int y = 0; y < d.rows; ++y)
    {
        const T *sr = row_ptr<T>(s, std::min(static_cast<int>(std::floor(y * ify)), s.rows - 1));
        T *dr = row_ptr_mut<T>(d, y);
        for (int x = 0; x < d.cols; ++x)
            for (int c = 0; c < cn; ++c)
                dr[x * cn + c] = sr[sx[x] * cn + c];
    }
}

// Generic separable interpolation: horizontal pass per source row, vertical pass per destination row.
template <typename T>
void eval_generic(const ImageRef &s, const ImageRef &d, double inv_y, double inv_x, int ksize,
                  bool area_mode)
{
    constexpr bool U8 = std::is_same<T, uint8_t>::value;
    const int cn = s.channels, W = d.cols * cn, sh = s.rows, sw = s.cols;
    const Axis ax = generic_axis(d.cols, sw, inv_x, ksize, area_mode, true);
    const Axis ay = generic_axis(d.rows, sh, inv_y, ksize, area_mode, false);
    const int t0 = -(ksize / 2 - 1); // cubic: taps -1..2; linear: 0..1
    const std::vector<int> axi = fixed_coeffs(ax.coeffs), ayi = fixed_coeffs(ay.coeffs);

    // H[row][e]: int64 for uint8 (fixed point), float for float32. Only rows that
    // the vertical pass reads are computed.
    std::vector<char> need(sh, 0);
    for (int y = 0; y < d.rows; ++y)
        for (int t = 0; t < ksize; ++t)
            need[std::clamp(ay.ofs[y] + t0 + t, 0, sh - 1)] = 1;
    std::vector<int64_t> Hi(U8 ? static_cast<std::size_t>(sh) * W : 0);
    std::vector<float> Hf(U8 ? 0 : static_cast<std::size_t>(sh) * W);
    for (int r = 0; r < sh; ++r)
    {
        if (!need[r])
            continue;
        const T *sr = row_ptr<T>(s, r);
        for (int x = 0; x < d.cols; ++x)
        {
            int cols[4];
            for (int t = 0; t < ksize; ++t)
                cols[t] = std::clamp(ax.ofs[x] + t0 + t, 0, sw - 1);
            const bool beyond = ksize == 2 && x >= ax.xmax; // D = S[sx]*ONE beyond xmax
            for (int c = 0; c < cn; ++c)
            {
                const std::size_t o = static_cast<std::size_t>(r) * W + x * cn + c;
                if (U8)
                {
                    int64_t g[4];
                    for (int t = 0; t < ksize; ++t)
                        g[t] = sr[cols[t] * cn + c];
                    if (beyond)
                    {
                        Hi[o] = g[0] * COEF_SCALE;
                        continue;
                    }
                    int64_t h = g[0] * axi[x * ksize];
                    for (int t = 1; t < ksize; ++t)
                        h = h + g[t] * axi[x * ksize + t];
                    Hi[o] = h;
                }
                else
                {
                    float g[4];
                    for (int t = 0; t < ksize; ++t)
                        g[t] = static_cast<float>(sr[cols[t] * cn + c]);
                    if (beyond)
                    {
                        Hf[o] = g[0];
                        continue;
                    }
                    float h = g[0] * ax.coeffs[x * ksize];
                    for (int t = 1; t < ksize; ++t)
                        h = h + g[t] * ax.coeffs[x * ksize + t]; // ((t0+t1)+t2)+t3
                    Hf[o] = h;
                }
            }
        }
    }

    const float sc = 1.0f / static_cast<float>(COEF_SCALE * COEF_SCALE);
    for (int y = 0; y < d.rows; ++y)
    {
        int rr[4];
        for (int t = 0; t < ksize; ++t)
            rr[t] = std::clamp(ay.ofs[y] + t0 + t, 0, sh - 1); // sy clipped
        T *dr = row_ptr_mut<T>(d, y);
        for (int e = 0; e < W; ++e)
        {
            if (ksize == 2)
            {
                if (U8)
                {
                    const int64_t b0 = ayi[y * 2], b1 = ayi[y * 2 + 1];
                    const int64_t r0 = Hi[static_cast<std::size_t>(rr[0]) * W + e],
                                  r1 = Hi[static_cast<std::size_t>(rr[1]) * W + e];
                    const int64_t v =
                        (((b0 * (r0 >> 4)) >> 16) + ((b1 * (r1 >> 4)) >> 16) + 2) >> 2;
                    dr[e] = static_cast<T>(v & 0xFF); // uchar(...) truncates
                }
                else
                {
                    const float r0 = Hf[static_cast<std::size_t>(rr[0]) * W + e],
                                r1 = Hf[static_cast<std::size_t>(rr[1]) * W + e];
                    const float p0 = r0 * ay.coeffs[y * 2], p1 = r1 * ay.coeffs[y * 2 + 1];
                    dr[e] = static_cast<T>(p0 + p1);
                }
            }
            else if (U8)
            {
                int64_t R[4];
                for (int t = 0; t < 4; ++t)
                    R[t] = Hi[static_cast<std::size_t>(rr[t]) * W + e];
                if (e < (W / CUBIC_U8_VEC) * CUBIC_U8_VEC)
                { // SIMD lanes: float path
                    float bf[4];
                    for (int t = 0; t < 4; ++t)
                        bf[t] = static_cast<float>(ayi[y * 4 + t]) * sc;
                    float acc = static_cast<float>(R[3]) * bf[3];
                    acc = static_cast<float>(R[2]) * bf[2] + acc;
                    acc = static_cast<float>(R[1]) * bf[1] + acc;
                    acc = static_cast<float>(R[0]) * bf[0] + acc;
                    dr[e] = store_u8(acc);
                }
                else
                { // scalar tail: FixedPtCast<int, uchar, 22>
                    int64_t sum = 0;
                    for (int t = 0; t < 4; ++t)
                        sum += R[t] * ayi[y * 4 + t];
                    dr[e] = static_cast<T>(std::clamp<int64_t>((sum + (1 << 21)) >> 22, 0, 255));
                }
            }
            else
            {
                float p[4];
                for (int t = 0; t < 4; ++t)
                    p[t] = Hf[static_cast<std::size_t>(rr[t]) * W + e] * ay.coeffs[y * 4 + t];
                if (e < (W / CUBIC_F32_VEC) * CUBIC_F32_VEC)
                    dr[e] = static_cast<T>(p[0] + (p[1] + (p[2] + p[3]))); // v_muladd nesting
                else
                    dr[e] = static_cast<T>(((p[0] + p[1]) + p[2]) + p[3]); // scalar loop
            }
        }
    }
}

// Integer-ratio AREA: resizeAreaFast_ for an exact integer ratio, with its count-based edges.
template <typename T> void eval_area_fast(const ImageRef &s, const ImageRef &d, int ix, int iy)
{
    constexpr bool U8 = std::is_same<T, uint8_t>::value;
    const int cn = s.channels, sh = s.rows, sw = s.cols, W = d.cols * cn;
    const int dwidth1 = (sw / ix) * cn, area = ix * iy;
    const float scale = 1.0f / static_cast<float>(area);
    for (int dy = 0; dy < d.rows; ++dy)
    {
        T *dr = row_ptr_mut<T>(d, dy);
        const int sy0 = dy * iy;
        if (sy0 >= sh)
        {
            for (int e = 0; e < W; ++e)
                dr[e] = T(0);
            continue;
        } // D[dx] = 0
        const int w = sy0 + iy <= sh ? dwidth1 : 0;
        for (int e = 0; e < w; ++e)
        {
            const int px = e / cn, c = e % cn;
            auto at = [&](int k)
            {
                return row_ptr<T>(s, sy0 + k / ix)[(px * ix + k % ix) * cn + c];
            };
            if (U8)
            {
                int64_t sum = 0;
                for (int k = 0; k < area; ++k)
                    sum += at(k);
                if (ix == 2 && iy == 2 && (cn == 1 || cn == 3 || cn == 4))
                    dr[e] = static_cast<T>((sum + 2) >> 2);
                else
                    dr[e] = static_cast<T>(store_u8(static_cast<float>(sum) * scale));
            }
            else
            {
                float acc = 0.0f;
                int k = 0;
                for (; k + 4 <= area; k += 4) // unrolled: sum += a + b + c + d
                    acc = acc + (((static_cast<float>(at(k)) + at(k + 1)) + at(k + 2)) + at(k + 3));
                for (; k < area; ++k)
                    acc = acc + static_cast<float>(at(k));
                float row = acc * scale;
                if (ix == 2 && iy == 2 && (cn == 1 || cn == 4) &&
                    e < (w / AREA2X2_F32_VEC) * AREA2X2_F32_VEC)
                    row = ((static_cast<float>(at(0)) + at(1)) +
                           (static_cast<float>(at(2)) + at(3))) *
                          scale; // ResizeAreaFastVec_SIMD_32f
                dr[e] = static_cast<T>(row);
            }
        }
        for (int e = w; e < W; ++e)
        { // count-based edge
            const int px = e / cn, c = e % cn;
            const int sx0 = px * ix * cn + c;
            int64_t itotal = 0;
            float ftotal = 0.0f;
            int count = 0;
            for (int sy = 0; sy < iy && sy0 + sy < sh; ++sy)
            {
                const T *sr = row_ptr<T>(s, sy0 + sy);
                for (int sx = 0; sx < ix * cn && sx0 + sx < sw * cn; sx += cn)
                {
                    if (U8)
                        itotal += sr[sx0 + sx];
                    else
                        ftotal = ftotal + static_cast<float>(sr[sx0 + sx]);
                    ++count;
                }
            }
            if (U8)
                dr[e] = count == 0 ? T(0)
                                   : static_cast<T>(store_u8(static_cast<float>(itotal) /
                                                             static_cast<float>(count)));
            else
                dr[e] = static_cast<T>(count == 0 ? std::numeric_limits<float>::quiet_NaN()
                                                  : ftotal / static_cast<float>(count));
        }
    }
}

// Fractional-ratio AREA: resizeArea_ -- h = h + S*alpha over a pixel's x taps in table order, then
// acc = acc + beta*h over its y taps. Padding taps are skipped, never multiplied by zero.
template <typename T>
void eval_area(const ImageRef &s, const ImageRef &d, double scale_x, double scale_y)
{
    constexpr bool U8 = std::is_same<T, uint8_t>::value;
    const int cn = s.channels;
    const AreaTab tx = area_tab(s.cols, d.cols, scale_x), ty = area_tab(s.rows, d.rows, scale_y);
    for (int dy = 0; dy < d.rows; ++dy)
    {
        T *dr = row_ptr_mut<T>(d, dy);
        for (int dx = 0; dx < d.cols; ++dx)
        {
            for (int c = 0; c < cn; ++c)
            {
                float acc = 0.0f;
                for (int ky = 0; ky < ty.count[dy]; ++ky)
                {
                    const T *sr =
                        row_ptr<T>(s, ty.src_index[static_cast<std::size_t>(dy) * ty.taps + ky]);
                    float h = 0.0f;
                    for (int kx = 0; kx < tx.count[dx]; ++kx)
                    {
                        const std::size_t i = static_cast<std::size_t>(dx) * tx.taps + kx;
                        h = h + static_cast<float>(sr[tx.src_index[i] * cn + c]) * tx.alpha[i];
                    }
                    acc = acc + ty.alpha[static_cast<std::size_t>(dy) * ty.taps + ky] * h;
                }
                dr[dx * cn + c] = U8 ? static_cast<T>(store_u8(acc)) : static_cast<T>(acc);
            }
        }
    }
}

template <typename T>
void dispatch(const ImageRef &s, const ImageRef &d, double inv_y, double inv_x, int interp)
{
    const RouteInfo r = route(s.rows, s.cols, d.rows, d.cols, inv_y, inv_x, interp);
    switch (r.kind)
    {
    case ROUTE_COPY:
        eval_copy<T>(s, d);
        break;
    case ROUTE_NEAREST:
        eval_nearest<T>(s, d, inv_y, inv_x);
        break;
    case ROUTE_AREA_FAST:
        eval_area_fast<T>(s, d, r.ix, r.iy);
        break;
    case ROUTE_AREA:
        eval_area<T>(s, d, 1.0 / inv_x, 1.0 / inv_y);
        break;
    case ROUTE_LINEAR:
        eval_generic<T>(s, d, inv_y, inv_x, 2, r.area_mode);
        break;
    case ROUTE_CUBIC:
        eval_generic<T>(s, d, inv_y, inv_x, 4, false);
        break;
    }
}

} // namespace

void resize(const ImageRef &src, const ImageRef &dst, double inv_y, double inv_x, int interp)
{
    if (!src.data || !dst.data || src.rows <= 0 || src.cols <= 0 || dst.rows <= 0 || dst.cols <= 0)
        throw std::invalid_argument("empty image");
    if (src.channels != dst.channels || src.is_f32 != dst.is_f32 || src.channels < 1)
        throw std::invalid_argument("src and dst must have the same type and channel count");
    if (!(inv_x > 0) || !(inv_y > 0))
        throw std::invalid_argument("inverse scales must be > 0");
    if (src.is_f32)
        dispatch<float>(src, dst, inv_y, inv_x, interp);
    else
        dispatch<uint8_t>(src, dst, inv_y, inv_x, interp);
}

} // namespace cv_triton_resize
