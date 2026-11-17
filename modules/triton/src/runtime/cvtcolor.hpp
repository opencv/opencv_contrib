// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Pure code/depth admission, request validation and explicit compile keys.
// No device access: CPU controls can exercise normalize without launching work.
//
// all() contains 84 numeric rows; enum aliases do not add rows. CV_8U serves
// every row. CV_16U serves 20: Bayer, colour/gray and YUV/YCrCb. CV_32F serves
// 20: channel permutations and gray copies, Op::Gray,
// Op::Yuv and Op::YCrCb. The two 2-channel Y extractions are 8U-only,
// as required by the CPU cvtColor wrapper. Integer conversions and copies use exact comparison;
// float arithmetic uses 8*2^-24*max(1,|ref|,max|input channels at this pixel|).
//
// Two-channel images are admitted only by explicit rows: 565/555 packed bytes,
// 422 decode and Y-channel extraction. The six 422 conversions require even
// width, while extraction allows odd width. 420 decode takes a packed 3H/2 by
// W image and produces H by W. normalize checks these geometry rules before
// any compile/launch and reuses the provider's shared DeviceView validation.
//
// Unsupported depths/codes (reverse HSV, HLS, forward linear Luv, Bayer VNG/EA,
// YUV encoding) are refused; no nearby conversion or CPU fallback is selected.
// test/test_cvtcolor.cpp compares the public GPU entry with the linked
// cv::cvtColor implementation. CPU dispatch, IPP/HAL and compiler options can
// affect the reference arithmetic. See the module documentation for limitations.
#ifndef OPENCV_TRITON_RUNTIME_CVTCOLOR_HPP
#define OPENCV_TRITON_RUNTIME_CVTCOLOR_HPP
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "opencv2/core/cvdef.h"   // CV_EXPORTS
#include "provider.hpp"
namespace cv_triton_color
{

// Operation ids. These are the OP constexprs of kernels/cvtcolor_cpu.py.
enum class Op : int
{
    Gray = 0,
    Permute = 1,
    Yuv = 2,
    YCrCb = 3,
    Hsv = 4,
    Packed = 5,
    Bayer = 6,
    Xyz = 7,
    Lab = 8,
    Luv = 9,
    Yuv422 = 10,
    Yuv420 = 11
};

// Arithmetic field meanings follow OpenCV 5.0.0 imgproc colour/demosaicing symbols.
struct CodeSpec
{
    const char *name; // the OpenCV constant name, e.g. "BGR2YUV"
    int code;         // the numeric cv::COLOR_* value
    Op op;
    int scn, dcn; // source / destination channel count
    int blue;     // blue index: 0 for a BGR-order source, 2 for an RGB-order one
    bool inverse; // direction: YUV/YCrCb inverse, and Packed decode (2ch -> colour)
    int hrange;   // Hsv only: 180 for HSV, 256 for HSV_FULL (h += h<0 ? hrange : 0)
    int perm[4];  // Permute only: source channel for each dst channel; -1 = opaque alpha
    // Packed only: green-bit count, the CPU's own discriminator between the two formats
    // (color.hpp greenBits(): 6 for every BGR565 code, 5 for every BGR555 code). It is a
    // field rather than something derived from the code value because the CPU decides by
    // passing 6 or 5 down to the invoker -- the code number is not consulted there.
    int gb = 0;
    // Bayer only: Start_with_green, as Bayer2RGB_/Bayer2Gray_ derive it from the code
    // (demosaicing.cpp). It is a field for the same reason `gb` is -- the
    // invoker is handed it already resolved and never sees the code number. Bayer's other phase
    // value, Blue, needs no new field: it is exactly -1 for BG/GB and +1 for RG/GR, which is the
    // `blue` above under a different meaning (0/2 is a channel index, +-1 a polarity), and no
    // op other than Bayer reads that field as a polarity.
    int swg = 0;
    // Xyz/Lab/Luv only: whether the sRGB transfer function applies. The "L" codes of this
    // family (LBGR2Lab, Lab2LBGR, Luv2LBGR, ...) mean LINEAR RGB, i.e. no transfer function at
    // all -- color.hpp is_sRGB() is true for exactly the non-L codes. For Lab this is the whole
    // difference (the CPU's linearGammaTab_b[i] is just i << 3); for Luv it decides which
    // reverse gamma applies, and the forward linear direction is refused entirely.
    bool srgb = true;
    // Xyz/Lab/Luv only: nine integer coefficients, derived by coeffs_for() from
    // the constructors in color_lab.cpp, including channel swaps and white-point
    // scaling. Zero for every other operation.
    int k[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    // Yuv422 only: which byte slot holds chroma U within the 4-byte macropixel -- 0 for UYVY,
    // 1 for YUY2, 2 for YVYU (color_yuv.simd.hpp). Keep this field
    // last to preserve the positional initializers of preceding fields.
    int ord = 0;
};

// The table. A code that is not here is refused by name; there is no nearest-neighbour
// substitution and no fallback to the CPU implementation.
// CV_EXPORTS for the same reason `all()` is: OpenCV builds with hidden visibility and the
// accuracy test checks the coefficient rows against coeffs_for.
CV_EXPORTS const CodeSpec *find(int code);
// CV_EXPORTS because OpenCV builds with hidden visibility and the accuracy test reads this
// table to prove the test's own coverage (test/test_cvtcolor.cpp, CoversEverySupportedCode).
CV_EXPORTS const std::vector<CodeSpec> &all();

struct Request
{
    cv_triton_runtime::DeviceView src, dst;
    int code = 0;
};

struct Normalized
{
    const CodeSpec *spec = nullptr;
    int rows = 0, cols = 0;
    long long n = 0; // output pixels = rows * cols
    // Both views have matching depth; normalize owns the family-specific admission.
    bool u16 = false;
    bool f32 = false;
    bool f32_cpu_separate4 = false;
};

// Private linked-CPU arithmetic identification; never receives user image data.
CV_EXPORTS bool float_cpu_separate4();

// Source admission used by every adapter before output allocation. A real source
// view (pointer, shape, byte step and optional allocation span) is required; no device access.
struct Output
{
    const CodeSpec *spec = nullptr;
    int rows = 0, cols = 0;
};
cv_triton_runtime::Status resolve_output(const cv_triton_runtime::DeviceView &src, int code,
                                         Output &out);

// Pure checks. On success `out.spec` points into the static table.
cv_triton_runtime::Status normalize(const Request &req, Normalized &out);

// The two int[256] tables RGB2HSV_b divides by, built exactly as the CPU builds them
// (color_hsv.simd.hpp: hsv_shift = 12, round-to-nearest on a double division).
struct HsvTables
{
    std::vector<int> sdiv, hdiv;
};
HsvTables hsv_tables();
// The same, for the _FULL codes: identical sdiv, hdiv built from (256 << 12) instead of
// (180 << 12), i.e. a hue range of 0..255.
HsvTables hsv_tables_full();

// Lookup tables reproduce color_lab.cpp's createLabTabs() and
// initLUTforLABLUVs16(), using softfloat/softdouble evaluation and cvRound().
// Replacing those operations with host libm functions can change rounded entries.
// The element types match kernel loads: cbrt requires int32 for values up to 37555;
// luv uses int64 because LvToVpl_b exceeds the signed 32-bit range.
struct LabLuvTables
{
    std::vector<int> gamma;      // 256:     sRGBGammaTab_b (or linearGammaTab_b, identical scale)
    std::vector<int> cbrt;       // 3072:    LabCbrtTab_b
    std::vector<int> yf;         // 512:     LabToYF_b, 2 per L
    std::vector<int> abxz;       // 36864:   abToXZ_b, already shifted to start at index 0
    std::vector<int> invgamma;   // 4096:    sRGBInvGammaTab_b
    std::vector<int> lu;         // 65536:   LuToUp_b
    std::vector<int> lv;         // 65536:   LvToVp_b
    std::vector<long long> lvpl; // 65536:   LvToVpl_b
    std::vector<short> luvLut;   // 33^3*3*8: RGB2LuvLUT_s16, built only for Op::Luv forward
};
// Built on first use and cached for the process: the Luv builder runs 33^3 softfloat
// evaluations plus a 33^3*8 fill, and it does not depend on the code.
const LabLuvTables &labluv_tables();

// The nine integer coefficients of a Xyz/Lab/Luv code, replicated from the CPU's constructors
// in color_lab.cpp (see the definition). Filled into CodeSpec::k when the table is built; zero
// for every other op. The test calls it directly to prove the table's rows and this function
// agree, so a code added to one and not the other cannot pass.
CV_EXPORTS void coeffs_for(int code, int out[9]);

// The offset, in ENTRIES, of each array inside the corresponding device buffer. These are the
// numbers the kernel hardcodes too (kernels/cvtcolor_cpu.py: T_GAMMA ... T_LV), and
// test_cvtcolor.cpp's numeric tests are what would fail first if the two drifted apart.
struct LabLuvLayout
{
    static const int gamma = 0;
    static const int cbrt = 256;
    static const int yf = 256 + 3072;
    static const int abxz = 256 + 3072 + 512;
    // abToXZ_b is LAB_BASE*9/4 = 36864 entries, not the 11264 the index RANGE suggests: the CPU
    // allocates for i = minABvalue .. minABvalue + LAB_BASE*9/4 (color_lab.cpp), and
    // the valid indices run to ~23000, so the array cannot be sized from the observed values.
    static const int invgamma = abxz + 36864;
    static const int lu = invgamma + 4096;
    static const int lv = lu + 65536;
    static const int t32_entries = lv + 65536;
};

// The runtime-argument list of the one jit function, in Triton's type strings. Constexpr
// parameters are not runtime arguments. The compile helper checks this against
// the signature of the compiled kernel.
std::string signature(const Normalized &n);

// The explicit compile key. Triton is given no implicit specialisation (every channel count
// and the operation are constexprs, every other argument is do_not_specialize), so this key
// is complete: equal keys run the same binary whatever the sizes or pointers.
std::string constexprs_text(const Normalized &n);
int block_size(const Normalized &n);
int num_warps(const Normalized &n);
void grid(const Normalized &n, int block, unsigned out[3]); // {rows, ceil(cols/BLOCK), 1}

} // namespace cv_triton_color
#endif // OPENCV_TRITON_RUNTIME_CVTCOLOR_HPP
