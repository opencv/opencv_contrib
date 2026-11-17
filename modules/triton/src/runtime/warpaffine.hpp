// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Pure part of the warpAffine provider: validation and the explicit compile key. No device
// access, so it is testable without a GPU.
//
// Sampling follows the floating-point CPU cv::warpAffine paths in
// modules/imgproc/src/warp_kernels.simd.hpp. cv::cuda::warpAffine uses a different
// fixed-point sampling pipeline and is not a numerical reference for these kernels.
//
// Supported: INTER_NEAREST, INTER_LINEAR and INTER_CUBIC; INTER_AREA aliases LINEAR;
// CV_8U/CV_16U/CV_32F, 1/3/4 channels; five standard borders plus TRANSPARENT for
// NEAREST/LINEAR/AREA; WARP_INVERSE_MAP both ways. CUBIC+TRANSPARENT is refused, as is
// INTER_LANCZOS4, whose 8x8 interpolation is not implemented.
//
// CUBIC follows genericWarp -> bicubicVec. This path computes columns from
// absolute x without FMA; tests use the natural-width CPU destination as reference.
#ifndef OPENCV_TRITON_RUNTIME_WARPAFFINE_HPP
#define OPENCV_TRITON_RUNTIME_WARPAFFINE_HPP
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "opencv2/core/cvdef.h"   // CV_EXPORTS
#include "provider.hpp"
namespace cv_triton_warp
{

// Interpolation ids, matching cv::InterpolationFlags.
enum class Mode : int
{
    Nearest = 0,
    Linear = 1,
    Cubic = 2
};

struct Request
{
    cv_triton_runtime::DeviceView src, dst;
    // The 2x3 matrix as cv::warpAffine takes it (row-major, 6 doubles).
    double M[6] = {1, 0, 0, 0, 1, 0};
    int interpolation = 1; // cv::INTER_LINEAR; cv::INTER_AREA normalizes to Linear
    int border_mode = 0;   // cv::BORDER_CONSTANT
    // The border value is a Scalar: up to four components, one per channel. A scalar v means
    // Scalar(v, 0, 0, 0), a sequence means its components zero-padded to four. It stays a
    // double in the request and is converted to the output depth before launch.
    // Float borders retain fractional values, such as 0.5.
    double border[4] = {0, 0, 0, 0};
    bool inverse_map = false;
};

struct Normalized
{
    Mode mode = Mode::Linear;
    int src_h = 0, src_w = 0, dst_h = 0, dst_w = 0, channels = 0;
    bool u8 = true;
    bool u16 = false;
    // The sampling matrix: M itself when WARP_INVERSE_MAP is set, otherwise the closed-form
    // inverse cv::warpAffine computes in double.
    double A[6] = {1, 0, 0, 0, 1, 0};
    // Border codes per axis. A 1-pixel axis is sampled with REPLICATE whatever the requested
    // mode (cv::warpAffine's border_type_x/y), unless the mode is CONSTANT or TRANSPARENT.
    // Code 5 is TRANSPARENT; it selects a separate constexpr and requires initialized dst.
    int border_mode_code = 0, border_x = 0, border_y = 0;
    double border[4] = {0, 0, 0, 0};
    // Initial launch shape; the selected backend may replace it before building the compile key.
    int block_h = 1, block_w = 128, num_warps = 4;
    // Whether the reference CPU computes the column term with a real FMA. cv::v_fma is only
    // fused when the build has FMA3 (core/hal/intrin_sse.hpp uses _mm_add_ps(_mm_mul_ps(..))
    // otherwise), and the two land on different pixels at a 0.5 tie. normalize() calibrates
    // the actual linked CPU path per dtype/channels/useOptimized, not hardware capability.
    bool fused = true;
};

// Source/options admission before dst allocation; no device work or reference calibration.
cv_triton_runtime::Status validate_source(const Request &req);

// Pure output-dependent admission before allocation. Dimensions are rows then columns.
// Reuses source rules and rejects unrepresentable sampling expressions; no CPU calibration.
cv_triton_runtime::Status validate_output(const Request &req, int dst_h, int dst_w);

// Validation, then a cached linked-CPU arithmetic probe (no GPU). Fills every field of out.
cv_triton_runtime::Status normalize(const Request &req, Normalized &out);

// The closed-form 2x3 inverse cv::warpAffine uses (D = m0*m4 - m1*m3; D = D != 0 ? 1/D : 0).
void invert(const double M[6], double out[6]);

// The runtime-argument list of the kernel, in Triton's type strings. Constexpr parameters are
// not runtime arguments. Compared with what the JIT actually compiles by the compile helper's
// key check at compile time -- a mismatch is a refused compilation, not a wrong result -- and
// exercised end to end by test/test_warpaffine.cpp.
std::string signature(const Normalized &n);

// The explicit compile key's constexpr text (sorted by name).
std::string constexprs_text(const Normalized &n);

// Grid: one program per (BLOCK_H rows, BLOCK_W pixels) tile.
void grid(const Normalized &n, unsigned out[3]);

} // namespace cv_triton_warp
#endif // OPENCV_TRITON_RUNTIME_WARPAFFINE_HPP
