// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright 2026, The FlagOS Contributors.
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

// Parameter resolution and interpolation tables follow the generic cv::resize
// implementation in modules/imgproc/src/resize.cpp. Float32 multiply/add operations
// must round separately (-ffp-contract=off).
// Standalone CPU controls and test support do not use library-private headers.
#ifdef __OPENCV_BUILD
#include "../precomp.hpp"
#endif
#include "resize_plan.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#if defined(FLT_EVAL_METHOD) && FLT_EVAL_METHOD != 0
#error "resize tables require IEEE single-precision float evaluation (FLT_EVAL_METHOD == 0)"
#endif
namespace cv_triton_resize
{

namespace
{

constexpr int COEF_BITS = 11;
constexpr int COEF_SCALE = 1 << COEF_BITS;
inline float f32(double v)
{
    return static_cast<float>(v);
}

// rint with ties to even (the default rounding mode), as cvRound.
inline double rint_even(double v)
{
    return std::nearbyint(v);
}

inline int sat_short(float v)
{ // saturate_cast<short>(float)
    double r = rint_even(v);
    return static_cast<int>(std::max(-32768.0, std::min(32767.0, r)));
}

void cubic_coeffs(float x, float *c)
{ // interpolateCubic, float32 op by op
    const float A = -0.75f, one = 1.0f;
    float xp1 = x + one;
    float c0 = ((A * xp1 - 5.0f * A) * xp1 + 8.0f * A) * xp1 - 4.0f * A;
    float c1 = ((A + 2.0f) * x - (A + 3.0f)) * x * x + one;
    float omx = one - x;
    float c2 = ((A + 2.0f) * omx - (A + 3.0f)) * omx * omx + one;
    float c3 = one - c0 - c1 - c2;
    c[0] = c0;
    c[1] = c1;
    c[2] = c2;
    c[3] = c3;
}

} // namespace

const char *route_name(Route r)
{
    switch (r)
    {
    case ROUTE_COPY:
        return "copy";
    case ROUTE_NEAREST:
        return "nearest";
    case ROUTE_LINEAR:
        return "linear";
    case ROUTE_CUBIC:
        return "cubic";
    case ROUTE_AREA_FAST:
        return "area_fast";
    case ROUTE_AREA:
        return "area";
    }
    return "?";
}

// ------------------------------------------------------------------ parameters

Size2 resolve_size(int src_h, int src_w, int dst_w, int dst_h, double fx, double fy)
{ // resolve_size
    if (dst_w > 0 && dst_h > 0)
        return {dst_h, dst_w, static_cast<double>(dst_h) / src_h,
                static_cast<double>(dst_w) / src_w};
    if (!(fx > 0 && fy > 0))
        throw std::invalid_argument("either dsize or both fx and fy (> 0) must be given");
    return {static_cast<int>(rint_even(src_h * fy)), static_cast<int>(rint_even(src_w * fx)), fy,
            fx};
}

RouteInfo route(int src_h, int src_w, int dst_h, int dst_w, double inv_y, double inv_x, int interp)
{ // route
    if (interp < INTER_NEAREST || interp > INTER_AREA)
        throw std::invalid_argument("unsupported interpolation");
    RouteInfo r;
    if (dst_h == src_h && dst_w == src_w)
    {
        r.kind = ROUTE_COPY;
        return r;
    }
    if (interp == INTER_NEAREST)
    {
        r.kind = ROUTE_NEAREST;
        return r;
    }
    double scale_x = 1.0 / inv_x, scale_y = 1.0 / inv_y;
    int isx = static_cast<int>(rint_even(scale_x)), isy = static_cast<int>(rint_even(scale_y));
    bool area_fast =
        std::fabs(scale_x - isx) < DBL_EPSILON && std::fabs(scale_y - isy) < DBL_EPSILON;
    int mode = interp;
    if (mode == INTER_LINEAR && area_fast && isx == 2 && isy == 2)
        mode = INTER_AREA;
    if (mode == INTER_AREA && scale_x >= 1 && scale_y >= 1)
    {
        if (area_fast)
        {
            r.kind = ROUTE_AREA_FAST;
            r.ix = isx;
            r.iy = isy;
            return r;
        }
        r.kind = ROUTE_AREA;
        return r;
    }
    r.kind = mode == INTER_CUBIC ? ROUTE_CUBIC : ROUTE_LINEAR;
    r.area_mode = mode == INTER_AREA;
    return r;
}

// ---------------------------------------------------------------- coefficients

Axis generic_axis(int dst, int src, double inv_scale, int ksize, bool area_mode, bool horizontal)
{ // generic_axis
    Axis a;
    a.ksize = ksize;
    a.xmin = 0;
    a.xmax = dst;
    a.ofs.resize(dst);
    a.coeffs.resize(static_cast<std::size_t>(dst) * ksize);
    const double scale = 1.0 / inv_scale;
    const int ksize2 = ksize / 2;
    const bool cubic = ksize == 4;
    for (int d = 0; d < dst; ++d)
    {
        float f;
        long s;
        if (!area_mode)
        {
            f = f32((d + 0.5) * scale - 0.5);
            s = static_cast<long>(std::floor(f));
            f = f - static_cast<float>(s);
        }
        else
        {
            s = static_cast<long>(std::floor(d * scale));
            f = f32((d + 1) - (s + 1) * inv_scale);
            f = f <= 0 ? 0.0f : f - static_cast<float>(std::floor(f));
        }
        if (horizontal)
        {
            if (s < ksize2 - 1)
            {
                a.xmin = d + 1;
                if (s < 0 && !cubic)
                {
                    f = 0.0f;
                    s = 0;
                }
            }
            if (s + ksize2 >= src)
            {
                a.xmax = std::min(a.xmax, d);
                if (s >= src - 1 && !cubic)
                {
                    f = 0.0f;
                    s = src - 1;
                }
            }
        }
        a.ofs[d] = static_cast<int>(s);
        float *c = &a.coeffs[static_cast<std::size_t>(d) * ksize];
        if (cubic)
            cubic_coeffs(f, c);
        else
        {
            c[0] = 1.0f - f;
            c[1] = f;
        }
    }
    return a;
}

std::vector<int> fixed_coeffs(const std::vector<float> &coeffs)
{ // fixed
    std::vector<int> out(coeffs.size());
    for (std::size_t i = 0; i < coeffs.size(); ++i)
        out[i] = sat_short(coeffs[i] * static_cast<float>(COEF_SCALE));
    return out;
}

AreaTab area_tab(int src, int dst, double scale)
{ // area_tab + padded_area_tab
    std::vector<std::vector<std::pair<int, float>>> tab(dst);
    for (int dx = 0; dx < dst; ++dx)
    {
        double fsx1 = dx * scale, fsx2 = fsx1 + scale;
        double cell = std::min(scale, src - fsx1);
        long sx1 = static_cast<long>(std::ceil(fsx1)), sx2 = static_cast<long>(std::floor(fsx2));
        sx2 = std::min<long>(sx2, src - 1);
        sx1 = std::min(sx1, sx2);
        auto &t = tab[dx];
        if (sx1 - fsx1 > 1e-3)
            t.push_back({static_cast<int>(sx1 - 1), f32((sx1 - fsx1) / cell)});
        for (long sx = sx1; sx < sx2; ++sx)
            t.push_back({static_cast<int>(sx), f32(1.0 / cell)});
        if (fsx2 - sx2 > 1e-3)
            t.push_back(
                {static_cast<int>(sx2), f32(std::min(std::min(fsx2 - sx2, 1.0), cell) / cell)});
    }
    AreaTab a;
    for (auto &t : tab)
        a.taps = std::max<int>(a.taps, static_cast<int>(t.size()));
    // Use a fixed row width for GPU table indexing; count masks unused taps.
    // Zero-filled coefficients and safe source index 0 cover the padded slots.
    a.src_index.assign(static_cast<std::size_t>(dst) * a.taps, 0);
    a.alpha.assign(static_cast<std::size_t>(dst) * a.taps, 0.0f);
    a.count.resize(dst);
    for (int d = 0; d < dst; ++d)
    {
        a.count[d] = static_cast<int>(tab[d].size());
        for (int k = 0; k < a.count[d]; ++k)
        {
            a.src_index[static_cast<std::size_t>(d) * a.taps + k] = tab[d][k].first;
            a.alpha[static_cast<std::size_t>(d) * a.taps + k] = tab[d][k].second;
        }
    }
    return a;
}

} // namespace cv_triton_resize
