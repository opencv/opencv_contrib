// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright 2026, The FlagOS Contributors.
// Pure parts of the provider protocol: request normalisation and validation, the explicit
// compile key, launch geometry. No device access. See provider.hpp.
// Standalone CPU controls and test support do not use library-private headers.
#ifdef __OPENCV_BUILD
#include "../precomp.hpp"
#endif
#include "provider.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include "dispatch_table.hpp"
namespace cv_triton_runtime
{

namespace ref = cv_triton_resize;

const char *reason_name(Reason r)
{
    switch (r)
    {
    case Reason::None:
        return "none";
    case Reason::DepthOrChannels:
        return "depth_or_channels";
    case Reason::Interpolation:
        return "interpolation";
    case Reason::EmptyOrSize:
        return "empty_or_size";
    case Reason::TypeMismatch:
        return "type_mismatch";
    case Reason::StepLayout:
        return "step_layout";
    case Reason::IndexRange:
        return "index_range";
    case Reason::InPlaceOrOverlap:
        return "in_place_or_overlap";
    case Reason::ContextMismatch:
        return "context_mismatch";
    case Reason::RouteNotEnabled:
        return "route_not_enabled";
    case Reason::ToolchainUnavailable:
        return "toolchain_unavailable";
    case Reason::CompileFailed:
        return "compile_failed";
    case Reason::CompileTimeout:
        return "compile_timeout";
    case Reason::ManifestInvalid:
        return "manifest_invalid";
    case Reason::LoadFailed:
        return "load_failed";
    case Reason::DeviceAlloc:
        return "device_alloc";
    case Reason::LaunchFailed:
        return "launch_failed";
    }
    return "?";
}

namespace
{

Status prepared()
{
    Status s;
    s.code = Code::Prepared;
    return s;
}

Status unsupported(Reason r, std::string msg)
{
    Status s;
    s.code = Code::Unsupported;
    s.reason = r;
    s.message = std::move(msg);
    return s;
}

constexpr long long I32_MAX = std::numeric_limits<int>::max();

bool fits_i32(long long v)
{
    return v >= 0 && v <= I32_MAX;
}

Status check_view(const DeviceView &v, const char *which)
{
    const std::string label(which);
    if (v.ptr == 0 || v.rows <= 0 || v.cols <= 0 || v.channels <= 0)
        return unsupported(Reason::EmptyOrSize, label + " is empty");
    const long long width = static_cast<long long>(v.cols) * v.channels;
    if (!fits_i32(width))
        return unsupported(Reason::IndexRange, label + ": row width exceeds int32 elements");
    const std::size_t es = elem_size(v.type), row = static_cast<std::size_t>(width) * es;
    if (v.ptr % es != 0 || v.step % es != 0 || v.step < row)
        return unsupported(Reason::StepLayout,
                           label +
                               ": pointer/step must be element-aligned and step >= the packed row");
    if (v.step / es > static_cast<std::size_t>(I32_MAX))
        return unsupported(Reason::IndexRange, label + ": row stride exceeds int32 elements");
    // Every address expression in the current kernels is int32 element indexing.
    const std::size_t rows = static_cast<std::size_t>(v.rows - 1), stride = v.step / es;
    if (rows &&
        stride > (static_cast<std::size_t>(I32_MAX) - static_cast<std::size_t>(width)) / rows)
        return unsupported(Reason::IndexRange,
                           label + ": view span exceeds int32 element indexing");
    const std::size_t span = rows * v.step + row;
    const auto maxptr = std::numeric_limits<std::uintptr_t>::max();
    if (v.ptr > maxptr - span)
        return unsupported(Reason::IndexRange, label + ": address range overflows");
    if (v.alloc_base != 0 && (v.alloc_base > maxptr - v.alloc_bytes || v.ptr < v.alloc_base ||
                              v.ptr + span > v.alloc_base + v.alloc_bytes))
        return unsupported(Reason::StepLayout, label + " extends beyond its allocation");
    return prepared();
}

const char *tname(ElemType t)
{
    return t == ElemType::U8 ? "u8"
                             : (t == ElemType::U16 ? "u16" : (t == ElemType::S16 ? "i16" : "fp32"));
}

// Runtime arguments of the resize JIT functions in kernels/resize_cpu.py, in
// Triton type strings. Constexpr parameters are excluded. Integer arguments are
// i32; normalize() validates their ranges before compilation or launch.
std::string signature(ref::Route r, ElemType t)
{
    const std::string T = tname(t);
    const std::string io = "src:*" + T + ", dst:*" + T + ", ";
    const std::string tail = "src_rs:i32, dst_rs:i32, src_bs:i32, dst_bs:i32";
    switch (r)
    {
    case ref::ROUTE_NEAREST:
        return io + "xofs:*i32, yofs:*i32, dst_h:i32, W:i32, " + tail;
    case ref::ROUTE_LINEAR:
    case ref::ROUTE_CUBIC:
    {
        const std::string w =
            t == ElemType::U8 ? "i32" : "fp32"; // fixed-point vs float coefficients
        return io + "xofs:*i32, ax:*" + w + ", yofs:*i32, by:*" + w +
               ", byf:*fp32, "
               "src_h:i32, src_w:i32, dst_h:i32, W:i32, xmax:i32, " +
               tail;
    }
    case ref::ROUTE_AREA_FAST:
        return io +
               "src_h:i32, src_w:i32, dst_h:i32, W:i32, dwidth1:i32, scale:fp32, IX:i32, IY:i32, " +
               tail;
    case ref::ROUTE_AREA:
        return io +
               "xs:*i32, xa:*fp32, xc:*i32, ys:*i32, ya:*fp32, yc:*i32, KX:i32, KY:i32, dst_h:i32, W:i32, " +
               tail;
    case ref::ROUTE_COPY:
        break;
    }
    return "";
}

} // namespace

Status validate_view(const DeviceView &view, const char *name)
{
    return check_view(view, name);
}

Status validate_output_shape(int rows, int cols, int channels)
{
    if (rows <= 0 || cols <= 0 || channels <= 0)
        return unsupported(Reason::EmptyOrSize,
                           "destination dimensions and channels must be positive");
    const long long width = static_cast<long long>(cols) * channels;
    if (width > I32_MAX || rows > I32_MAX / width)
        return unsupported(Reason::IndexRange,
                           "packed destination span exceeds int32 element indexing");
    return prepared();
}

bool views_overlap(const DeviceView &first, const DeviceView &second)
{
    if (first.ptr >= second.end() || second.ptr >= first.end())
        return false;
    const std::size_t firstWidth = first.row_bytes(), secondWidth = second.row_bytes();
    int firstRow = 0, secondRow = 0;
    std::uintptr_t firstBegin = first.ptr, secondBegin = second.ptr;
    while (firstRow < first.rows && secondRow < second.rows)
    {
        if (firstBegin < secondBegin + secondWidth && secondBegin < firstBegin + firstWidth)
            return true;
        if (firstBegin + firstWidth <= secondBegin)
        {
            ++firstRow;
            if (firstRow < first.rows)
                firstBegin += first.step;
        }
        else
        {
            ++secondRow;
            if (secondRow < second.rows)
                secondBegin += second.step;
        }
    }
    return false;
}

Status resolve_output(const ResizeRequest &req, ref::Size2 &out)
{
    const DeviceView &s = req.src;
    if (s.type != ElemType::U8 && s.type != ElemType::F32)
        return unsupported(Reason::DepthOrChannels, "resize serves U8 and F32 only");
    if (req.interpolation < ref::INTER_NEAREST || req.interpolation > ref::INTER_AREA)
        return unsupported(Reason::Interpolation,
                           "interpolation must be NEAREST, LINEAR, CUBIC or AREA");
    if (s.channels != 1 && s.channels != 3 && s.channels != 4)
        return unsupported(Reason::DepthOrChannels, "channels must be 1, 3 or 4");
    if (s.rows <= 0 || s.cols <= 0)
        return unsupported(Reason::EmptyOrSize, "src is empty");
    // Validate scale-derived dimensions before the size resolver's double-to-int conversion.
    // Explicit dsize keeps cv::resize precedence, even if ignored fx/fy are nonfinite.
    if (!(req.dsize_w > 0 && req.dsize_h > 0))
    {
        if (!std::isfinite(req.fx) || !std::isfinite(req.fy) || !(req.fx > 0 && req.fy > 0))
            return unsupported(Reason::EmptyOrSize,
                               "either dsize or both finite fx and fy (> 0) must be given");
        const double width = s.cols * req.fx, height = s.rows * req.fy;
        if (!std::isfinite(width) || !std::isfinite(height) || width > I32_MAX || height > I32_MAX)
            return unsupported(Reason::IndexRange, "scale-derived size exceeds int32 dimensions");
    }
    try
    {
        out = ref::resolve_size(s.rows, s.cols, req.dsize_w, req.dsize_h, req.fx, req.fy);
    }
    catch (const std::invalid_argument &e)
    {
        return unsupported(Reason::EmptyOrSize, e.what());
    }
    if (out.dst_h <= 0 || out.dst_w <= 0)
        return unsupported(Reason::EmptyOrSize, "resolved size is empty");
    return validate_output_shape(out.dst_h, out.dst_w, s.channels);
}

Status normalize(const ResizeRequest &req, Normalized &out)
{
    const DeviceView &s = req.src, &d = req.dst;
    ref::Size2 z;
    Status resolved = resolve_output(req, z);
    if (!resolved.ok())
        return resolved;
    if (s.type != d.type || s.channels != d.channels)
        return unsupported(Reason::TypeMismatch,
                           "src and dst must have the same depth and channel count");
    if (!(s.ctx == d.ctx))
        return unsupported(Reason::ContextMismatch, "src and dst belong to different contexts");
    Status st = check_view(s, "src");
    if (!st.ok())
        return st;
    st = check_view(d, "dst");
    if (!st.ok())
        return st;

    if (d.rows != z.dst_h || d.cols != z.dst_w)
        return unsupported(Reason::EmptyOrSize, "dst size differs from the resolved size");
    if (views_overlap(s, d))
        return unsupported(Reason::InPlaceOrOverlap,
                           "src and dst overlap (in-place is not supported)");

    out.src_h = s.rows;
    out.src_w = s.cols;
    out.dst_h = z.dst_h;
    out.dst_w = z.dst_w;
    out.channels = s.channels;
    out.type = s.type;
    out.interpolation = req.interpolation;
    out.inv_y = z.inv_y;
    out.inv_x = z.inv_x;
    out.route = ref::route(out.src_h, out.src_w, out.dst_h, out.dst_w, out.inv_y, out.inv_x,
                           out.interpolation);
    if (out.route.kind == ref::ROUTE_AREA)
    { // tap counts are kernel scalars too
        const ref::AreaTab tx = ref::area_tab(out.src_w, out.dst_w, 1.0 / out.inv_x);
        const ref::AreaTab ty = ref::area_tab(out.src_h, out.dst_h, 1.0 / out.inv_y);
        if (!fits_i32(static_cast<long long>(tx.taps) * out.dst_w) ||
            !fits_i32(static_cast<long long>(ty.taps) * out.dst_h))
            return unsupported(Reason::IndexRange, "area tap tables exceed int32 indexing");
    }
    return prepared();
}

std::string KernelKey::text() const
{
    std::ostringstream o;
    o << "kernel=" << kernel << ";sig=" << signature << ";cx=";
    for (std::size_t i = 0; i < constexprs.size(); ++i)
        o << (i ? "," : "") << constexprs[i].first << "=" << constexprs[i].second;
    o << ";nw=" << cfg.num_warps << ";ns=" << cfg.num_stages << ";target=" << target.backend << ":"
      << target.arch << ":" << target.warp_size << ";compiler=" << toolchain.compiler
      << ";source=" << toolchain.source_sha256;
    return o.str();
}

LaunchConfig default_config(const Normalized &n, const Target *target)
{
    LaunchConfig cfg;
    cfg.num_stages = DISPATCH_DEFAULT_NUM_STAGES;
    // The class comes from the route the request actually takes (n.route, already resolved
    // from the caller's fx/fy by normalize()). Re-deriving it from the rounded sizes would
    // pick a different table row for fx/fy requests: fx = 1/3 runs area_fast while its
    // rounded sizes look like area, and the two rows differ. Preserve the caller's
    // inverse scales when fx/fy determined the output size.
    const ref::RouteInfo look = n.route;
    const char *want = ref::route_name(look.kind);
    const char *ydir = n.dst_h > n.src_h ? "up" : "down";
    const bool two = look.kind == ref::ROUTE_AREA_FAST && look.ix == 2 && look.iy == 2;
    const bool f32 = n.type == ElemType::F32;
    // The (route, cn, dtype, ydir[, two_by_two]) row; for area_fast a miss falls back to the
    // same row with two_by_two = false. Only "down" rows exist, and the index is
    // (route, cn, is_f32, yd) -- an absent "up" row is a miss, not a down row.
    auto match = [&](bool two_by_two) -> const DispatchRow *
    {
        for (const DispatchRow &r : DISPATCH_TABLE)
            if (std::strcmp(r.route, want) == 0 && r.cn == n.channels && r.f32 == f32 &&
                std::strcmp(r.ydir, ydir) == 0 && r.two_by_two == two_by_two)
                return &r;
        return nullptr;
    };
    const DispatchRow *row = match(two);
    if (!row && look.kind == ref::ROUTE_AREA_FAST)
        row = match(false);
    if (row)
    {
        cfg.block_h = row->block_h;
        cfg.block_w = row->block_w;
        cfg.num_warps = row->num_warps;
    }
    else
    {
        cfg.block_h = DISPATCH_FALLBACK_BLOCK_H;
        cfg.block_w = DISPATCH_FALLBACK_BLOCK_W;
        cfg.num_warps = DISPATCH_FALLBACK_NUM_WARPS;
    }
    // HIP uses launch defaults by route, channel layout and scale direction;
    // BLOCK_W counts channel elements for nearest/separable kernels, pixels for AREA.
    // These are tuning choices, not sampling rules or proven hardware mechanisms.
    // They apply to every HIP target, not an architecture-specific optimum; benchmark
    // the actual device before carrying over a performance claim. CUDA uses the table.
    if (target && target->backend == "hip" && static_cast<long long>(n.dst_h) * n.dst_w >= 65536)
    {
        const bool f32 = n.type == ElemType::F32;
        const double sy = 1.0 / n.inv_y, sx = 1.0 / n.inv_x;
        const bool point = !f32 && look.kind == ref::ROUTE_LINEAR && !look.area_mode &&
                           std::max(n.src_h, n.src_w) < (1 << 23) && sy >= 1 && sx >= 1 &&
                           sy == std::floor(sy) && sx == std::floor(sx) &&
                           std::fmod(sy, 2.0) == 1 && std::fmod(sx, 2.0) == 1;
        if (f32 && n.dst_h <= n.src_h && n.dst_w <= n.src_w)
        {
            // Aligned C1 nearest and C3 linear downscales use distinct row/width tiles.
            if (look.kind == ref::ROUTE_NEAREST && n.channels == 1 && n.src_w % 128 == 0 &&
                n.dst_w % 128 == 0)
            {
                cfg.block_h = 2;
                cfg.block_w = 512;
                cfg.num_warps = 2;
            }
            else if (look.kind == ref::ROUTE_LINEAR && n.channels == 3)
            {
                cfg.block_h = 4;
                cfg.block_w = 256;
                cfg.num_warps = 1;
            }
        }
        if (!f32)
        {
            // Odd-integer LINEAR ratios select a point sample; exact 2x2 windows
            // have dedicated channel layouts. Other LINEAR rows distinguish up/downscale.
            if (point)
            {
                cfg.block_h = 1;
                cfg.block_w = 512;
                cfg.num_warps = 4;
            }
            else if (two && n.channels == 1)
            {
                cfg.block_h = 1;
                cfg.block_w = 1024;
                cfg.num_warps = 4;
            }
            else if (two && n.channels == 4)
            {
                cfg.block_h = 1;
                cfg.block_w = 512;
                cfg.num_warps = 8;
            }
            else if (look.kind == ref::ROUTE_LINEAR && n.channels == 1 && n.dst_h <= n.src_h)
            {
                cfg.block_h = 4;
                cfg.block_w = 256;
                cfg.num_warps = 2;
            }
            else if (look.kind == ref::ROUTE_LINEAR && n.channels == 3 && n.dst_h > n.src_h)
            {
                cfg.block_h = 8;
                cfg.block_w = 256;
                cfg.num_warps = 2;
            }
        }
    }
    // Shrink BLOCK_H, then BLOCK_W, until the launch has at least
    // DISPATCH_MIN_PROGRAMS programs (or neither can shrink). It counts programs on
    // dst_w for axis 1 whatever the route; that only errs on the small side for C3/C4
    // element-major tiles, which is the safe direction. Copies keep the fallback tile.
    if (look.kind != ref::ROUTE_COPY)
    {
        auto programs = [&](const LaunchConfig &c)
        {
            return static_cast<long long>((n.dst_h + c.block_h - 1) / c.block_h) *
                   ((n.dst_w + c.block_w - 1) / c.block_w);
        };
        while (programs(cfg) < DISPATCH_MIN_PROGRAMS && cfg.block_h > 1)
            cfg.block_h = std::max(1, cfg.block_h / 2);
        while (programs(cfg) < DISPATCH_MIN_PROGRAMS && cfg.block_w > 64)
            cfg.block_w = std::max(64, cfg.block_w / 2);
    }
    // RY is only a constexpr of the area_fast kernel; the JIT passes cfg.get("RY", 1).
    cfg.ry = 1;
    return cfg;
}

LaunchShape launch_shape(const Normalized &n, const LaunchConfig &requested, const Target &t,
                         const Toolchain &tc, int batch, const DeviceView *src,
                         const DeviceView *dst)
{
    LaunchShape ls;
    const ref::Route r = n.route.kind;
    if (r == ref::ROUTE_COPY)
    {
        ls.is_copy = true;
        return ls;
    }
    // Fields the caller left at 0 come from the JIT's dispatch table. An explicitly supplied
    // value is never replaced, so the compile protocol's (block_h, block_w, num_warps) pass
    // through unchanged.
    LaunchConfig cfg = requested;
    const LaunchConfig d = default_config(n, &t);
    if (cfg.block_h <= 0)
        cfg.block_h = d.block_h;
    if (cfg.block_w <= 0)
        cfg.block_w = d.block_w;
    if (cfg.num_warps <= 0)
        cfg.num_warps = d.num_warps;
    if (cfg.num_stages <= 0)
        cfg.num_stages = d.num_stages;
    if (cfg.ry <= 0)
        cfg.ry = d.ry;
    KernelKey &k = ls.key;
    k.cfg = cfg;
    k.target = t;
    k.toolchain = tc;
    k.signature = signature(r, n.type);
    const long u8 = n.type == ElemType::U8 ? 1 : 0;
    std::vector<std::pair<std::string, long>> &c = k.constexprs;
    c = {{"BLOCK_H", cfg.block_h}, {"BLOCK_W", cfg.block_w}, {"cn", n.channels}};
    switch (r)
    {
    case ref::ROUTE_NEAREST:
        k.kernel = "_resize_cpu_nearest_kernel";
        break;
    case ref::ROUTE_LINEAR:
    case ref::ROUTE_CUBIC:
        k.kernel = "_resize_cpu_sep_kernel";
        c.push_back({"KSIZE", r == ref::ROUTE_CUBIC ? 4 : 2});
        c.push_back({"U8", u8});
        break;
    case ref::ROUTE_AREA_FAST:
        k.kernel = "_resize_cpu_area_fast_kernel";
        c.push_back({"RY", cfg.ry});
        c.push_back({"TWO_BY_TWO", n.route.ix == 2 && n.route.iy == 2 ? 1 : 0});
        c.push_back({"U8", u8});
        break;
    case ref::ROUTE_AREA:
        k.kernel = "_resize_cpu_area_kernel";
        c.push_back({"U8", u8});
        break;
    case ref::ROUTE_COPY:
        break;
    }
    if (t.backend == "hip" && n.type == ElemType::U8)
    {
        const double sy = 1.0 / n.inv_y, sx = 1.0 / n.inv_x;
        const bool point = r == ref::ROUTE_LINEAR && !n.route.area_mode &&
                           std::max(n.src_h, n.src_w) < (1 << 23) && sy >= 1 && sx >= 1 &&
                           sy == std::floor(sy) && sx == std::floor(sx) &&
                           std::fmod(sy, 2.0) == 1 && std::fmod(sx, 2.0) == 1;
        const bool full2 = r == ref::ROUTE_AREA_FAST && n.route.ix == 2 && n.route.iy == 2 &&
                           n.src_h >= 2LL * n.dst_h && n.src_w >= 2LL * n.dst_w;
        if (point)
        {
            k.kernel = "_resize_cpu_nearest_kernel";
            k.signature = signature(ref::ROUTE_NEAREST, n.type);
            c = {{"BLOCK_H", cfg.block_h}, {"BLOCK_W", cfg.block_w}, {"cn", n.channels}};
        }
        else if (full2 && n.channels == 3 && n.dst_h * static_cast<long long>(n.dst_w) >= 65536 &&
                 requested.block_h == 0 && requested.block_w == 0 && requested.num_warps == 0)
        {
            k.kernel = "_hip_area2_channels";
            k.signature =
                "src:*u8, dst:*u8, dst_h:i32, W:i32, src_rs:i32, dst_rs:i32, src_bs:i32, dst_bs:i32";
            k.cfg.block_h = 1;
            k.cfg.block_w = 512;
            k.cfg.num_warps = 4;
            c = {{"BLOCK_E", 512}};
            ls.grid[0] = n.dst_h;
            ls.grid[1] = (static_cast<long long>(n.dst_w) * 3 + 511) / 512;
            ls.grid[2] = batch;
            return ls;
        }
        else if (full2 && (n.channels == 1 || n.channels == 4) &&
                 (n.channels == 4 || n.dst_w % 4 == 0) && cfg.block_w % 4 == 0 && src && dst &&
                 src->ptr % 4 == 0 && dst->ptr % 4 == 0 && src->step % 4 == 0 && dst->step % 4 == 0)
        {
            k.kernel = "_hip_area2_packed";
            k.signature =
                "src:*u8, dst:*u8, dst_h:i32, dst_w:i32, src_rs:i32, dst_rs:i32, src_bs:i32, dst_bs:i32";
            c = {{"BLOCK_H", cfg.block_h},
                 {"BLOCK_W", cfg.block_w},
                 {"C1", n.channels == 1 ? 1 : 0}};
        }
    }
    std::sort(c.begin(), c.end());
    // Grid axis 1 follows the kernel's tile unit: BLOCK_W output pixels for
    // area kernels, or BLOCK_W channel values for nearest and separable kernels.
    const long long W = static_cast<long long>(n.dst_w) * n.channels;
    const long long axis1 = (r == ref::ROUTE_AREA || r == ref::ROUTE_AREA_FAST) ? n.dst_w : W;
    ls.grid[0] = static_cast<unsigned>((n.dst_h + cfg.block_h - 1) / cfg.block_h);
    ls.grid[1] = static_cast<unsigned>((axis1 + cfg.block_w - 1) / cfg.block_w);
    ls.grid[2] = static_cast<unsigned>(batch);
    return ls;
}

} // namespace cv_triton_runtime
