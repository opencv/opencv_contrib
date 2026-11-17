// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright 2026, The FlagOS Contributors.
// Standalone CPU controls and test support do not use library-private headers.
#ifdef __OPENCV_BUILD
#include "../precomp.hpp"
#endif
#include "warpaffine.hpp"
#include "opencv2/core/utility.hpp"   // cv::useOptimized
#include "opencv2/imgproc.hpp"        // cv::INTER_AREA (the CPU rewrite this follows)
#include <algorithm>
#include <cmath>
#include <sstream>
#include <map>
#include <mutex>
#include <tuple>
#include <limits>
namespace cv_triton_warp
{

using cv_triton_runtime::Reason;
using cv_triton_runtime::Status;

namespace
{
Status prepared()
{
    Status s;
    s.code = cv_triton_runtime::Code::Prepared;
    return s;
}

Status unsupported(Reason r, std::string msg)
{
    Status s;
    s.code = cv_triton_runtime::Code::Unsupported;
    s.reason = r;
    s.message = std::move(msg);
    return s;
}

// Hardware FMA support does not identify the arithmetic compiled into imgproc.
// Calibrate the linked CPU vector path at rounding ties, with no CPU feature
// toggles and no device work. Padding keeps every sampled column out of the tail.
// Type/channel dispatch and useOptimized are part of the calibration identity.
bool reference_fused(int depth, int channels)
{
    static std::mutex mutex;
    static std::map<std::tuple<int, int, bool>, bool> cache;
    const auto key = std::make_tuple(depth, channels, cv::useOptimized());
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = cache.find(key);
    if (found != cache.end())
        return found->second;

    cv::Mat ids(37, 53, CV_MAKETYPE(CV_16U, channels));
    for (int y = 0; y < ids.rows; ++y)
        for (int x = 0; x < ids.cols; ++x)
            for (int c = 0; c < channels; ++c)
                ids.ptr<unsigned short>(y)[x * channels + c] =
                    static_cast<unsigned short>(1 + (y * ids.cols + x) * 17 + c * 73);
    cv::Mat source;
    ids.convertTo(source, CV_MAKETYPE(depth, channels));
    if (depth == CV_8U)
        for (int y = 0; y < source.rows; ++y)
            for (int x = 0; x < source.cols * channels; ++x)
                source.ptr<unsigned char>(y)[x] =
                    static_cast<unsigned char>(ids.ptr<unsigned short>(y)[x] % 251 + 1);

    // Fixed probes are independent of the user's matrix, seed or expected image.
    const double probes[][6] = {{1, 1.9, -20, .4, 1, -10}, {.9, .43589, -14, -.43589, .9, 10}};
    bool matches[2] = {true, true};
    for (const auto &matrix : probes)
    {
        double sampling[6];
        invert(matrix, sampling);
        float a[6];
        for (int i = 0; i < 6; ++i)
            a[i] = static_cast<float>(sampling[i]);
        cv::Mat reference;
        cv::warpAffine(source, reference, cv::Mat(2, 3, CV_64F, const_cast<double *>(matrix)),
                       cv::Size(256, 37), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
        cv::Mat src64, ref64;
        source.convertTo(src64, CV_MAKETYPE(CV_64F, channels));
        reference.convertTo(ref64, CV_MAKETYPE(CV_64F, channels));
        for (int fused = 0; fused < 2; ++fused)
            for (int y = 0; y < 37 && matches[fused]; ++y)
                for (int x = 0; x < 53 && matches[fused]; ++x)
                {
                    // Volatile intermediates prohibit compiler FMA contraction
                    // in the unfused alternative, regardless of module flags.
                    volatile float rowx = y * a[1], rowy = y * a[4];
                    const float rx = rowx + a[2], ry = rowy + a[5];
                    volatile float colx = x * a[0], coly = x * a[3];
                    const float sx = fused ? std::fma(float(x), a[0], rx) : colx + rx;
                    const float sy = fused ? std::fma(float(x), a[3], ry) : coly + ry;
                    const int ix = cvRound(sx), iy = cvRound(sy);
                    for (int c = 0; c < channels; ++c)
                    {
                        const double predicted = ix >= 0 && ix < 53 && iy >= 0 && iy < 37
                                                     ? src64.ptr<double>(iy)[ix * channels + c]
                                                     : 0;
                        if (predicted != ref64.ptr<double>(y)[x * channels + c])
                            matches[fused] = false;
                    }
                }
    }
    if (matches[0] == matches[1])
        CV_Error(cv::Error::StsUnsupportedFormat,
                 "cv::triton::warpAffine: linked CPU vector arithmetic could not be identified");
    cache.emplace(key, matches[1]);
    return matches[1];
}

bool is_border(int b)
{
    return b == 0 || b == 1 || b == 2 || b == 3 || b == 4 || b == 5;
    // CONSTANT/REPLICATE/REFLECT/WRAP/REFLECT_101/TRANSPARENT. TRANSPARENT preserves outside
    // nearest samples; LINEAR/AREA use the old destination for each outside tap.
}
} // namespace

void invert(const double M[6], double out[6])
{
    double D = M[0] * M[4] - M[1] * M[3];
    D = D != 0 ? 1.0 / D : 0.0;
    const double a11 = M[4] * D, a22 = M[0] * D;
    out[0] = a11;
    out[1] = M[1] * -D;
    out[3] = M[3] * -D;
    out[4] = a22;
    out[2] = -out[0] * M[2] - out[1] * M[5];
    out[5] = -out[3] * M[2] - out[4] * M[5];
}

// Matrix rules shared by pre-allocation admission and the final planner.
static Status sampling_matrix(const Request &req, double out[6])
{
    for (int i = 0; i < 6; ++i)
        if (!std::isfinite(req.M[i]))
            return unsupported(Reason::IndexRange, "matrix entries must be finite");
    if (req.inverse_map)
        std::copy(req.M, req.M + 6, out);
    else
        invert(req.M, out);
    for (int i = 0; i < 6; ++i)
        if (!std::isfinite(out[i]) ||
            std::abs(out[i]) > static_cast<double>(std::numeric_limits<float>::max()))
            return unsupported(Reason::IndexRange, "sampling matrix exceeds finite float32 range");
    return prepared();
}

Status validate_source(const Request &req)
{
    const cv_triton_runtime::DeviceView &s = req.src;

    // OpenCV's warpAffine rewrites AREA to LINEAR before selecting its kernel.
    const int interp = req.interpolation == cv::INTER_AREA ? cv::INTER_LINEAR : req.interpolation;
    if (interp != 0 && interp != 1 && interp != 2)
        return unsupported(Reason::Interpolation,
                           "cv::triton::warpAffine: interpolation " +
                               std::to_string(req.interpolation) +
                               " is not supported (supported modes are "
                               "INTER_NEAREST, INTER_LINEAR, INTER_CUBIC and INTER_AREA); "
                               "INTER_LANCZOS4 is refused rather than approximated");
    // BORDER_TRANSPARENT is served for NEAREST and LINEAR (including the AREA alias).
    // TransparentMatchesCpu checks the selected reference with initialized destinations,
    // every supported dtype/channel, ROI guards and inside/partial/outside geometries.
    // CUBIC + TRANSPARENT remains outside the implemented subset.
    if (req.border_mode == cv::BORDER_TRANSPARENT && req.interpolation == cv::INTER_CUBIC)
        return unsupported(
            Reason::Interpolation,
            "cv::triton::warpAffine: BORDER_TRANSPARENT with INTER_CUBIC is not "
            "served; TRANSPARENT works with INTER_NEAREST, INTER_LINEAR and INTER_AREA");
    if (!is_border(req.border_mode))
        return unsupported(
            Reason::Interpolation,
            "cv::triton::warpAffine: border mode " + std::to_string(req.border_mode) +
                " is not one of CONSTANT/REPLICATE/REFLECT/WRAP/REFLECT_101/TRANSPARENT");
    if (s.type != cv_triton_runtime::ElemType::U8 && s.type != cv_triton_runtime::ElemType::U16 &&
        s.type != cv_triton_runtime::ElemType::F32)
        return unsupported(Reason::DepthOrChannels,
                           "cv::triton::warpAffine: CV_8U, CV_16U and CV_32F only");
    if (s.channels != 1 && s.channels != 3 && s.channels != 4)
        return unsupported(Reason::DepthOrChannels, "channels must be 1, 3 or 4");
    for (int i = 0; i < 4; ++i)
    {
        if (!std::isfinite(req.border[i]))
            return unsupported(Reason::IndexRange, "border components must be finite");
        if (s.type == cv_triton_runtime::ElemType::F32 &&
            std::abs(req.border[i]) > std::numeric_limits<float>::max())
            return unsupported(Reason::IndexRange,
                               "float32 border components exceed finite float32 range");
    }
    double matrix[6];
    return sampling_matrix(req, matrix);
}

Status validate_output(const Request &req, int dst_h, int dst_w)
{
    const Status admitted = validate_source(req);
    if (!admitted.ok())
        return admitted;
    const Status output = cv_triton_runtime::validate_output_shape(dst_h, dst_w, req.src.channels);
    if (!output.ok())
        return output;
    double matrix[6];
    const Status prepared_matrix = sampling_matrix(req, matrix);
    if (!prepared_matrix.ok())
        return prepared_matrix;
    // Bound each float multiply/add term before calibration or allocation.
    // The conservative absolute-term bound covers both fused/split forms;
    // CUBIC clamps before integer conversion, other modes need an int32 margin.
    const double rounding = 1.0 + 8.0 * std::numeric_limits<float>::epsilon();
    for (int row = 0; row < 2; ++row)
    {
        const int i = row * 3;
        const double bound = std::abs(matrix[i]) * (dst_w - 1) +
                             std::abs(matrix[i + 1]) * (dst_h - 1) + std::abs(matrix[i + 2]);
        if (!std::isfinite(bound) || bound > std::numeric_limits<float>::max() / rounding)
            return unsupported(Reason::IndexRange,
                               "sampling coordinate expression exceeds finite float32 range");
        if (req.interpolation != cv::INTER_CUBIC && bound > 2147483392.0 / rounding)
            return unsupported(Reason::IndexRange, "sampling coordinates exceed safe int32 range");
    }
    return prepared();
}

Status normalize(const Request &req, Normalized &out)
{
    const cv_triton_runtime::DeviceView &s = req.src, &d = req.dst;
    const Status admitted = validate_output(req, d.rows, d.cols);
    if (!admitted.ok())
        return admitted;
    const int interp = req.interpolation == cv::INTER_AREA ? cv::INTER_LINEAR : req.interpolation;
    if (s.type != d.type || s.channels != d.channels)
        return unsupported(Reason::TypeMismatch,
                           "src and dst must have the same depth and channel count");
    if (!(s.ctx == d.ctx))
        return unsupported(Reason::ContextMismatch, "src and dst belong to different contexts");
    Status valid = cv_triton_runtime::validate_view(s, "src");
    if (!valid.ok())
        return valid;
    valid = cv_triton_runtime::validate_view(d, "dst");
    if (!valid.ok())
        return valid;
    if (cv_triton_runtime::views_overlap(s, d))
        return unsupported(Reason::InPlaceOrOverlap,
                           "src and dst overlap (in-place is not supported)");

    out.mode = interp == 0 ? Mode::Nearest : (interp == 1 ? Mode::Linear : Mode::Cubic);
    out.u8 = s.type == cv_triton_runtime::ElemType::U8;
    out.u16 = s.type == cv_triton_runtime::ElemType::U16;
    out.src_h = s.rows;
    out.src_w = s.cols;
    out.dst_h = d.rows;
    out.dst_w = d.cols;
    out.channels = s.channels;
    const Status matrix = sampling_matrix(req, out.A);
    if (!matrix.ok())
        return matrix;

    out.border_mode_code = req.border_mode;
    out.border_x = req.border_mode;
    out.border_y = req.border_mode;
    // A degenerate transparent axis must still detect missing taps and preserve
    // their old destination contribution; it cannot become a replicate axis.
    if (req.border_mode != cv::BORDER_CONSTANT && req.border_mode != cv::BORDER_TRANSPARENT)
    {
        if (out.src_w <= 1)
            out.border_x = 1;
        if (out.src_h <= 1)
            out.border_y = 1;
    }
    for (int i = 0; i < 4; ++i)
    {
        out.border[i] =
            (out.u8 || out.u16)
                ? std::max(0.0, std::min(out.u8 ? 255.0 : 65535.0, std::nearbyint(req.border[i])))
                : req.border[i];
    }

    // One tile of output pixels per program. The runtime may select a
    // backend-specific launch configuration.
    out.block_h = 1;
    out.block_w = 128;
    out.num_warps = 4;

    const int depth = out.u8 ? CV_8U : (out.u16 ? CV_16U : CV_32F);
    out.fused = reference_fused(depth, s.channels);
    return prepared();
}

std::string signature(const Normalized &n)
{
    // Must match the kernel's own parameter order:
    //   (src, dst, M00..M12, src_h, src_w, dst_h, dst_w, <constexprs>)
    // The six matrix entries are fp32 runtime arguments: making them constexpr would compile a
    // separate binary per affine matrix.
    // The four border components are RUNTIME fp32 arguments, never constexprs: as constexprs
    // they would be truncated to integers, which is wrong for CV_32F (a border of 0.5), and
    // they would also bake every border value into a separate compiled binary. fp32 rather
    // than the source dtype because a u8-typed scalar argument is not a legal ABI for this
    // kernel; the kernel converts to its own dtype, which is the same saturate/round the CPU
    // does.
    // The pointer types are concrete per dtype ("*u8" / "*fp32"), not a placeholder: the
    // compile helper maps each token to a Triton type and has no generic form.
    const char *t = n.u8 ? "u8" : (n.u16 ? "u16" : "fp32");
    std::ostringstream o;
    o << "src:*" << t << ", dst:*" << t
      << ", M00:fp32, M01:fp32, M02:fp32, M10:fp32, M11:fp32, M12:fp32, "
         "src_h:i32, src_w:i32, dst_h:i32, dst_w:i32, src_rs:i32, dst_rs:i32, "
         "bv0:fp32, bv1:fp32, bv2:fp32, bv3:fp32";
    return o.str();
}

std::string constexprs_text(const Normalized &n)
{
    std::ostringstream o;
    o << "BLOCK_H=" << n.block_h << ",BLOCK_W=" << n.block_w << ",BORDER=" << n.border_mode_code
      << ",BORDER_X=" << n.border_x << ",BORDER_Y=" << n.border_y << ",FUSED=" << (n.fused ? 1 : 0)
      << ",TRANSPARENT=" << (n.border_mode_code == cv::BORDER_TRANSPARENT ? 1 : 0);
    // U8/U16 exist only on the kernels that convert their loads (LINEAR, CUBIC); the NEAREST
    // kernel copies the element unchanged, so its dtype is carried by the pointer type alone
    // and declaring an unused constexpr would make the compile helper's key check fail.
    if (n.mode != Mode::Nearest)
        o << ",U8=" << (n.u8 ? 1 : 0) << ",U16=" << (n.u16 ? 1 : 0);
    o << ",cn=" << n.channels;
    return o.str();
}

void grid(const Normalized &n, unsigned out[3])
{
    const int bh = n.block_h > 0 ? n.block_h : 1;
    const int bw = n.block_w > 0 ? n.block_w : 128;
    out[0] = static_cast<unsigned>((n.dst_h + bh - 1) / bh);
    out[1] = static_cast<unsigned>((n.dst_w + bw - 1) / bw);
    out[2] = 1;
}

} // namespace cv_triton_warp
