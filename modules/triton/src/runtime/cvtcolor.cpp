// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright 2026, The FlagOS Contributors.
// Pure part of the cvtColor provider: the code table, request validation and the explicit
// compile key. No device access, so it is testable without a GPU.
//
// Integer conversions follow the fixed-point formulas and lookup tables used by
// OpenCV's generic CPU implementation. Channel permutations are exact copies.
// Tests compare the public GPU entry with the linked cv::cvtColor implementation;
// alternative HAL, IPP or SIMD paths may use different rounding.
//
// Reverse HSV/HLS and forward linear Luv are not implemented. Their CPU floating
// point paths can differ between vector blocks and scalar tails; the conversion
// table below lists the supported alternatives explicitly.
// Standalone CPU controls and test support do not use library-private headers.
#ifdef __OPENCV_BUILD
#include "../precomp.hpp"
#endif
#include "cvtcolor.hpp"
#include <algorithm>
#include <cfloat>
#include <cfenv>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
#include <map>
#include <tuple>
#include "opencv2/imgproc.hpp"
#include <climits>
#include <cmath>
#include <cstring>
#include <mutex>
#include <sstream>
// The tables below are cvRound() of a softfloat/softdouble evaluation, and the CPU builds them
// the same way. Using the same primitives is the only way the two agree: a std::pow/lrint build
// disagrees on a handful of entries, and every pixel that reads one is then off by one. This is
// the same header modules/imgproc/src/color_lab.cpp itself includes.
#include "opencv2/core/softfloat.hpp"
// Arithmetic source references: OpenCV 5.0.0 imgproc colour/demosaicing symbols.
namespace cv_triton_color
{

using cv_triton_runtime::Code;
using cv_triton_runtime::Reason;
using cv_triton_runtime::Status;

namespace
{

// Leading columns: name, code, op, scn, dcn, blue, inverse, hrange, perm[].
// Trailing family-specific fields use CodeSpec defaults when omitted; labels on
// long rows are reading aids, not C++ designated initializers.
// perm[] selects source channels; -1 writes opaque alpha (255, 65535 or 1 by depth).
const CodeSpec TABLE[] = {
    // ---- channel permutes
    {"BGR2BGRA", 0, Op::Permute, 3, 4, 0, false, 180, {0, 1, 2, -1}},
    {"BGRA2BGR", 1, Op::Permute, 4, 3, 0, false, 180, {0, 1, 2, -1}},
    {"BGR2RGBA", 2, Op::Permute, 3, 4, 0, false, 180, {2, 1, 0, -1}},
    {"RGBA2BGR", 3, Op::Permute, 4, 3, 0, false, 180, {2, 1, 0, -1}},
    {"BGR2RGB", 4, Op::Permute, 3, 3, 0, false, 180, {2, 1, 0, -1}},
    {"BGRA2RGBA", 5, Op::Permute, 4, 4, 0, false, 180, {2, 1, 0, 3}},
    // ---- gray -> color (src has ONE channel; perm 0 reads it for all three outputs)
    //
    // COLOR_GRAY2RGB is an alias of COLOR_GRAY2BGR (both 8) and GRAY2RGBA of GRAY2BGRA (both 9):
    // the CPU writes the same gray value to all three colour channels, so the byte order does not
    // distinguish them. Two rows cover the four names.
    //
    // Op::Permute, not Op::Gray: the permute body already does "copy source channel I(k), or
    // write opaque alpha when I(k) is -1", and with scn=1 every I(k)=0 reads the gray value. Op::Gray
    // would load three channels and compute a luma sum, which is the opposite direction.
    //
    // Not served, and why: the CPU lets dstCn override the code (cvtColor(g,m,COLOR_GRAY2BGR,4)
    // is accepted and produces the 4-channel result), while this table pins dcn per row. Adding a
    // dstCn parameter here would be a public-API change; the supported form is the code's own
    // channel count.
    {"GRAY2BGR", 8, Op::Permute, 1, 3, 0, false, 180, {0, 0, 0, -1}},
    {"GRAY2RGBA", 9, Op::Permute, 1, 4, 0, false, 180, {0, 0, 0, -1}},
    // ---- to gray (blue index selects channel order; OpenCV accepts both)
    {"BGR2GRAY", 6, Op::Gray, 3, 1, 0, false, 180, {0, 0, 0, 0}},
    {"RGB2GRAY", 7, Op::Gray, 3, 1, 2, false, 180, {0, 0, 0, 0}},
    {"BGRA2GRAY", 10, Op::Gray, 4, 1, 0, false, 180, {0, 0, 0, 0}},
    {"RGBA2GRAY", 11, Op::Gray, 4, 1, 2, false, 180, {0, 0, 0, 0}},
    // YUV2GRAY_UYVY / _YUY2 (123 / 124). These are NOT a conversion: color.cpp calls
    // cvtColorYUV2Gray_ch, which is `extractChannel(_src, _dst, coi)` with coi = 1 for UYVY and
    // 0 for YUY2 (color_yuv.dispatch.cpp). So they are a plain channel copy, they accept
    // ODD widths (no parity assert anywhere on this path), and they are served by Permute with a
    // single source channel. COLOR_YUV2GRAY_YVYU is the same value 124 -- a synonym to YUY2,
    // and safe only because both extract Y.
    {"YUV2GRAY_UYVY", 123, Op::Permute, 2, 1, 0, true, 180, {1, 0, 0, 0}},
    {"YUV2GRAY_YUY2", 124, Op::Permute, 2, 1, 0, true, 180, {0, 0, 0, 0}},
    // ---- 4:2:0 semi-planar decode: NV12 / NV21 -> colour. DECODE ONLY.
    //
    // A 4:2:0 source is ONE CV_8UC1 of (3H/2) x W and the result is H x W, so this is the only
    // family the module serves whose SIZE changes; normalize() validates that rule rather than
    // mere equality. Rows [0,H) are Y, rows [H, 3H/2) are the interleaved chroma plane with one
    // (U,V) pair per 2x2 block. ord 0 = NV12 (U first), 1 = NV21 (V first).
    //
    // No encode direction: color -> NV12/I420/YV12 would have to produce a taller buffer, and
    // that is the same interface question in reverse. It stays refused.
    {"YUV2BGR_NV12",
     91,
     Op::Yuv420,
     1,
     3,
     0,
     true,
     180,
     {0, 0, 0, 0},
     0,
     0,
     true,
     {0, 0, 0, 0, 0, 0, 0, 0, 0},
     0},
    {"YUV2RGB_NV12",
     90,
     Op::Yuv420,
     1,
     3,
     2,
     true,
     180,
     {0, 0, 0, 0},
     0,
     0,
     true,
     {0, 0, 0, 0, 0, 0, 0, 0, 0},
     0},
    {"YUV2BGRA_NV12",
     95,
     Op::Yuv420,
     1,
     4,
     0,
     true,
     180,
     {0, 0, 0, 0},
     0,
     0,
     true,
     {0, 0, 0, 0, 0, 0, 0, 0, 0},
     0},
    {"YUV2RGBA_NV12",
     94,
     Op::Yuv420,
     1,
     4,
     2,
     true,
     180,
     {0, 0, 0, 0},
     0,
     0,
     true,
     {0, 0, 0, 0, 0, 0, 0, 0, 0},
     0},
    {"YUV2BGR_NV21",
     93,
     Op::Yuv420,
     1,
     3,
     0,
     true,
     180,
     {0, 0, 0, 0},
     0,
     0,
     true,
     {0, 0, 0, 0, 0, 0, 0, 0, 0},
     1},
    {"YUV2RGB_NV21",
     92,
     Op::Yuv420,
     1,
     3,
     2,
     true,
     180,
     {0, 0, 0, 0},
     0,
     0,
     true,
     {0, 0, 0, 0, 0, 0, 0, 0, 0},
     1},
    {"YUV2BGRA_NV21",
     97,
     Op::Yuv420,
     1,
     4,
     0,
     true,
     180,
     {0, 0, 0, 0},
     0,
     0,
     true,
     {0, 0, 0, 0, 0, 0, 0, 0, 0},
     1},
    {"YUV2RGBA_NV21",
     96,
     Op::Yuv420,
     1,
     4,
     2,
     true,
     180,
     {0, 0, 0, 0},
     0,
     0,
     true,
     {0, 0, 0, 0, 0, 0, 0, 0, 0},
     1},

    // ---- YCrCb
    {"BGR2YCrCb", 36, Op::YCrCb, 3, 3, 0, false, 180, {0, 0, 0, 0}},
    {"RGB2YCrCb", 37, Op::YCrCb, 3, 3, 2, false, 180, {0, 0, 0, 0}},
    {"YCrCb2BGR", 38, Op::YCrCb, 3, 3, 0, true, 180, {0, 0, 0, 0}},
    {"YCrCb2RGB", 39, Op::YCrCb, 3, 3, 2, true, 180, {0, 0, 0, 0}},
    // ---- HSV, integer path, hue range 0..180
    {"BGR2HSV", 40, Op::Hsv, 3, 3, 0, false, 180, {0, 0, 0, 0}},
    {"RGB2HSV", 41, Op::Hsv, 3, 3, 2, false, 180, {0, 0, 0, 0}},
    // ---- HSV, full hue range 0..255: same code, a different div table
    {"BGR2HSV_FULL", 66, Op::Hsv, 3, 3, 0, false, 256, {0, 0, 0, 0}},
    {"RGB2HSV_FULL", 67, Op::Hsv, 3, 3, 2, false, 256, {0, 0, 0, 0}},
    // ---- YUV
    // ---- 4:2:2 interleaved: UYVY / YUY2 / YVYU -> colour (decode only)
    //
    // The source is CV_8UC2 and the destination keeps the same rows/cols (the CPU's dstSz = sz),
    // so these fit the current entry. Only `width % 2 == 0` is required: one 4-byte macropixel
    // spans two output pixels and carries one chroma pair. `ord` picks the byte order
    // (0 UYVY, 1 YUY2, 2 YVYU).
    //
    // YUV2GRAY_UYVY/YUY2 (123/124) are NOT here: the CPU implements those as a plain
    // extractChannel (color_yuv.dispatch.cpp), which accepts ODD widths and is a channel
    // copy, not a conversion -- so they are served by the Permute rows below.
    {"YUV2RGB_UYVY",
     107,
     Op::Yuv422,
     2,
     3,
     2,
     true,
     180,
     {0, 0, 0, 0},
     0,
     0,
     true,
     {0, 0, 0, 0, 0, 0, 0, 0, 0},
     0},
    {"YUV2BGR_UYVY",
     108,
     Op::Yuv422,
     2,
     3,
     0,
     true,
     180,
     {0, 0, 0, 0},
     0,
     0,
     true,
     {0, 0, 0, 0, 0, 0, 0, 0, 0},
     0},
    {"YUV2RGB_YUY2",
     115,
     Op::Yuv422,
     2,
     3,
     2,
     true,
     180,
     {0, 0, 0, 0},
     0,
     0,
     true,
     {0, 0, 0, 0, 0, 0, 0, 0, 0},
     1},
    {"YUV2BGR_YUY2",
     116,
     Op::Yuv422,
     2,
     3,
     0,
     true,
     180,
     {0, 0, 0, 0},
     0,
     0,
     true,
     {0, 0, 0, 0, 0, 0, 0, 0, 0},
     1},
    {"YUV2RGB_YVYU",
     117,
     Op::Yuv422,
     2,
     3,
     2,
     true,
     180,
     {0, 0, 0, 0},
     0,
     0,
     true,
     {0, 0, 0, 0, 0, 0, 0, 0, 0},
     2},
    {"YUV2BGR_YVYU",
     118,
     Op::Yuv422,
     2,
     3,
     0,
     true,
     180,
     {0, 0, 0, 0},
     0,
     0,
     true,
     {0, 0, 0, 0, 0, 0, 0, 0, 0},
     2},
    {"BGR2YUV", 82, Op::Yuv, 3, 3, 0, false, 180, {0, 0, 0, 0}},
    {"RGB2YUV", 83, Op::Yuv, 3, 3, 2, false, 180, {0, 0, 0, 0}},
    {"YUV2BGR", 84, Op::Yuv, 3, 3, 0, true, 180, {0, 0, 0, 0}},
    {"YUV2RGB", 85, Op::Yuv, 3, 3, 2, true, 180, {0, 0, 0, 0}},
    // ---- packed 5x5 RGB (codes 12..31)
    //
    // THE LAYOUT FACT THAT DECIDES EVERYTHING HERE: OpenCV 5.x represents a 5x5-packed image as
    // CV_8UC2, not CV_16UC1 -- one 16-bit word per pixel, little-endian inside the two bytes.
    // The CPU's own check says so: color_rgb.dispatch.cpp cvtColor5x52BGR is
    // CvtHelper<Set<2>, Set<3,4>, Set<CV_8U>>, and passing a 16UC1 Mat is rejected with
    // Bad-number-of-channels ('scn' is 1). So scn=2 below is not "two channels of colour"; it is
    // two BYTES of one pixel, and the kernel assembles the word from them.
    //
    // Both directions are Op::Packed and are told apart by `inverse` (true = decode, 2ch ->
    // colour), exactly as YUV uses it. The format is `gb`, the green-bit count the CPU itself
    // passes down (color.hpp greenBits()): 6 for 565, 5 for 555.
    //
    // Bit fields (color_rgb.simd.hpp unpack,  pack, verified over all 65536
    // words):   565 b=(v<<3)&0xF8 g=(v>>3)&0xFC r=(v>>8)&0xF8, alpha 255
    //           555 b=(v<<3)&0xF8 g=(v>>2)&0xF8 r=(v>>7)&0xF8, bit15 = 1-bit alpha
    // Note the 555 green is NOT bit-replicated: full scale is 248, not 255.
    //
    // scn is pinned per row although the CPU accepts 3 OR 4 for the packing direction
    // (CvtHelper<Set<3,4>,...>): the table has one source channel count per code, the same way
    // BGR2BGRA is scn=3 and BGRA2BGRA does not exist. The 4-channel rows are the ones whose
    // code name says A, and they are the only place alpha reaches the packed word.
    {"BGR2BGR565", 12, Op::Packed, 3, 2, 0, false, 180, {0, 0, 0, 0}, 6},
    {"RGB2BGR565", 13, Op::Packed, 3, 2, 2, false, 180, {0, 0, 0, 0}, 6},
    {"BGRA2BGR565", 16, Op::Packed, 4, 2, 0, false, 180, {0, 0, 0, 0}, 6},
    {"RGBA2BGR565", 17, Op::Packed, 4, 2, 2, false, 180, {0, 0, 0, 0}, 6},
    {"GRAY2BGR565", 20, Op::Packed, 1, 2, 0, false, 180, {0, 0, 0, 0}, 6},
    {"BGR5652BGR", 14, Op::Packed, 2, 3, 0, true, 180, {0, 0, 0, 0}, 6},
    {"BGR5652RGB", 15, Op::Packed, 2, 3, 2, true, 180, {0, 0, 0, 0}, 6},
    {"BGR5652BGRA", 18, Op::Packed, 2, 4, 0, true, 180, {0, 0, 0, 0}, 6},
    {"BGR5652RGBA", 19, Op::Packed, 2, 4, 2, true, 180, {0, 0, 0, 0}, 6},
    {"BGR5652GRAY", 21, Op::Packed, 2, 1, 0, true, 180, {0, 0, 0, 0}, 6},
    {"BGR2BGR555", 22, Op::Packed, 3, 2, 0, false, 180, {0, 0, 0, 0}, 5},
    {"RGB2BGR555", 23, Op::Packed, 3, 2, 2, false, 180, {0, 0, 0, 0}, 5},
    {"BGRA2BGR555", 26, Op::Packed, 4, 2, 0, false, 180, {0, 0, 0, 0}, 5},
    {"RGBA2BGR555", 27, Op::Packed, 4, 2, 2, false, 180, {0, 0, 0, 0}, 5},
    {"GRAY2BGR555", 30, Op::Packed, 1, 2, 0, false, 180, {0, 0, 0, 0}, 5},
    {"BGR5552BGR", 24, Op::Packed, 2, 3, 0, true, 180, {0, 0, 0, 0}, 5},
    {"BGR5552RGB", 25, Op::Packed, 2, 3, 2, true, 180, {0, 0, 0, 0}, 5},
    {"BGR5552BGRA", 28, Op::Packed, 2, 4, 0, true, 180, {0, 0, 0, 0}, 5},
    {"BGR5552RGBA", 29, Op::Packed, 2, 4, 2, true, 180, {0, 0, 0, 0}, 5},
    {"BGR5552GRAY", 31, Op::Packed, 2, 1, 0, true, 180, {0, 0, 0, 0}, 5},
    // ---- Bayer bilinear demosaic (codes 46-49, 86, 139), 8U and 16U
    //
    // The four BGR codes are ONE kernel with the 2x2 pattern phase shifted. The CPU derives that
    // phase from the code (demosaicing.cpp) and never consults it again, so the phase
    // is a constexpr here too: `blue` carries Bayer2RGB_'s Blue (-1 for BG/GB, +1 for RG/GR) and
    // `swg` its Start_with_green (1 for GB/GR, 0 for BG/RG).
    //
    // Bayer enum names and documented pattern names use different conventions. The fields
    // below follow the CPU invoker arguments rather than deriving phase from the enum name.
    //
    // BayerBG2BGR at dcn=4 is a distinct CPU call path (_dst created with 4 channels and the
    // invoker's `dcn == 4` arms) and so is a distinct row, not a rerun of the 3-channel one.
    // Bayer2GRAY is a separate kernel with its own coefficients, dcn pinned to 1.
    //
    // VNG (62) and EA (135) are intentionally absent -- see the header.
    {"BayerBG2BGR", 46, Op::Bayer, 1, 3, -1, false, 180, {0, 0, 0, 0}, 0, 0},
    {"BayerGB2BGR", 47, Op::Bayer, 1, 3, -1, false, 180, {0, 0, 0, 0}, 0, 1},
    {"BayerRG2BGR", 48, Op::Bayer, 1, 3, 1, false, 180, {0, 0, 0, 0}, 0, 0},
    {"BayerGR2BGR", 49, Op::Bayer, 1, 3, 1, false, 180, {0, 0, 0, 0}, 0, 1},
    {"BayerBG2GRAY", 86, Op::Bayer, 1, 1, -1, false, 180, {0, 0, 0, 0}, 0, 0},
    {"BayerBG2BGRA", 139, Op::Bayer, 1, 4, -1, false, 180, {0, 0, 0, 0}, 0, 0},
    // ---- XYZ and the Lab/Luv families (codes 32-35, 44/45, 56/57, 58/59, 74-81)
    //
    // All CV_8U, all integer, and the sRGB-vs-linear split is the `srgb` flag: the CPU's is_sRGB
    // is true for exactly the non-L codes (color.hpp). The Luv forward sRGB direction is
    // the one deliberate approximation in this group -- a trilinear LUT, not the CIE formula;
    // see kernels/cvtcolor_cpu.py.
    //
    // The coefficients are not in these rows: they are computed by coeffs_for() below from the
    // code, because the CPU computes them from softdouble in the constructor and nine hand-
    // written ints per row would be neither checkable nor correctable.
    //
    // Forward linear Luv (COLOR_LBGR2Luv and COLOR_LRGB2Luv) is not implemented.
    // Its CPU vector and scalar paths can produce different rounded results.
    {"BGR2XYZ",
     32,
     Op::Xyz,
     3,
     3,
     0,
     false,
     180,
     {0, 0, 0, 0},
     /*gb=*/0,
     /*swg=*/0,
     /*srgb=*/true,
     /*k=*/{0, 0, 0, 0, 0, 0, 0, 0, 0},
     /*ord=*/0},
    {"RGB2XYZ",
     33,
     Op::Xyz,
     3,
     3,
     2,
     false,
     180,
     {0, 0, 0, 0},
     /*gb=*/0,
     /*swg=*/0,
     /*srgb=*/true,
     /*k=*/{0, 0, 0, 0, 0, 0, 0, 0, 0},
     /*ord=*/0},
    {"XYZ2BGR",
     34,
     Op::Xyz,
     3,
     3,
     0,
     true,
     180,
     {0, 0, 0, 0},
     /*gb=*/0,
     /*swg=*/0,
     /*srgb=*/true,
     /*k=*/{0, 0, 0, 0, 0, 0, 0, 0, 0},
     /*ord=*/0},
    {"XYZ2RGB",
     35,
     Op::Xyz,
     3,
     3,
     2,
     true,
     180,
     {0, 0, 0, 0},
     /*gb=*/0,
     /*swg=*/0,
     /*srgb=*/true,
     /*k=*/{0, 0, 0, 0, 0, 0, 0, 0, 0},
     /*ord=*/0},
    {"BGR2Lab",
     44,
     Op::Lab,
     3,
     3,
     0,
     false,
     180,
     {0, 0, 0, 0},
     /*gb=*/0,
     /*swg=*/0,
     /*srgb=*/true,
     /*k=*/{0, 0, 0, 0, 0, 0, 0, 0, 0},
     /*ord=*/0},
    {"RGB2Lab",
     45,
     Op::Lab,
     3,
     3,
     2,
     false,
     180,
     {0, 0, 0, 0},
     /*gb=*/0,
     /*swg=*/0,
     /*srgb=*/true,
     /*k=*/{0, 0, 0, 0, 0, 0, 0, 0, 0},
     /*ord=*/0},
    {"Lab2BGR",
     56,
     Op::Lab,
     3,
     3,
     0,
     true,
     180,
     {0, 0, 0, 0},
     /*gb=*/0,
     /*swg=*/0,
     /*srgb=*/true,
     /*k=*/{0, 0, 0, 0, 0, 0, 0, 0, 0},
     /*ord=*/0},
    {"Lab2RGB",
     57,
     Op::Lab,
     3,
     3,
     2,
     true,
     180,
     {0, 0, 0, 0},
     /*gb=*/0,
     /*swg=*/0,
     /*srgb=*/true,
     /*k=*/{0, 0, 0, 0, 0, 0, 0, 0, 0},
     /*ord=*/0},
    {"BGR2Luv",
     50,
     Op::Luv,
     3,
     3,
     0,
     false,
     180,
     {0, 0, 0, 0},
     /*gb=*/0,
     /*swg=*/0,
     /*srgb=*/true,
     /*k=*/{0, 0, 0, 0, 0, 0, 0, 0, 0},
     /*ord=*/0},
    {"RGB2Luv",
     51,
     Op::Luv,
     3,
     3,
     2,
     false,
     180,
     {0, 0, 0, 0},
     /*gb=*/0,
     /*swg=*/0,
     /*srgb=*/true,
     /*k=*/{0, 0, 0, 0, 0, 0, 0, 0, 0},
     /*ord=*/0},
    {"Luv2BGR",
     58,
     Op::Luv,
     3,
     3,
     0,
     true,
     180,
     {0, 0, 0, 0},
     /*gb=*/0,
     /*swg=*/0,
     /*srgb=*/true,
     /*k=*/{0, 0, 0, 0, 0, 0, 0, 0, 0},
     /*ord=*/0},
    {"Luv2RGB",
     59,
     Op::Luv,
     3,
     3,
     2,
     true,
     180,
     {0, 0, 0, 0},
     /*gb=*/0,
     /*swg=*/0,
     /*srgb=*/true,
     /*k=*/{0, 0, 0, 0, 0, 0, 0, 0, 0},
     /*ord=*/0},
    // The linear variants: srgb = false. LBGR2Lab/LRGB2Lab differ from the sRGB codes ONLY in
    // the transfer function (linearGammaTab_b[i] == i << 3), which the kernel folds into a shift.
    {"LBGR2Lab",
     74,
     Op::Lab,
     3,
     3,
     0,
     false,
     180,
     {0, 0, 0, 0},
     /*gb=*/0,
     /*swg=*/0,
     /*srgb=*/false,
     /*k=*/{0, 0, 0, 0, 0, 0, 0, 0, 0},
     /*ord=*/0},
    {"LRGB2Lab",
     75,
     Op::Lab,
     3,
     3,
     2,
     false,
     180,
     {0, 0, 0, 0},
     /*gb=*/0,
     /*swg=*/0,
     /*srgb=*/false,
     /*k=*/{0, 0, 0, 0, 0, 0, 0, 0, 0},
     /*ord=*/0},
    {"Lab2LBGR",
     78,
     Op::Lab,
     3,
     3,
     0,
     true,
     180,
     {0, 0, 0, 0},
     /*gb=*/0,
     /*swg=*/0,
     /*srgb=*/false,
     /*k=*/{0, 0, 0, 0, 0, 0, 0, 0, 0},
     /*ord=*/0},
    {"Lab2LRGB",
     79,
     Op::Lab,
     3,
     3,
     2,
     true,
     180,
     {0, 0, 0, 0},
     /*gb=*/0,
     /*swg=*/0,
     /*srgb=*/false,
     /*k=*/{0, 0, 0, 0, 0, 0, 0, 0, 0},
     /*ord=*/0},
    {"Luv2LBGR",
     80,
     Op::Luv,
     3,
     3,
     0,
     true,
     180,
     {0, 0, 0, 0},
     /*gb=*/0,
     /*swg=*/0,
     /*srgb=*/false,
     /*k=*/{0, 0, 0, 0, 0, 0, 0, 0, 0},
     /*ord=*/0},
    {"Luv2LRGB",
     81,
     Op::Luv,
     3,
     3,
     2,
     true,
     180,
     {0, 0, 0, 0},
     /*gb=*/0,
     /*swg=*/0,
     /*srgb=*/false,
     /*k=*/{0, 0, 0, 0, 0, 0, 0, 0, 0},
     /*ord=*/0},
};

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

} // namespace

namespace
{

// The rows above are literals; the Xyz/Lab/Luv coefficients are not, because the CPU computes
// them from softdouble in its constructors (coeffs_for). They are filled ONCE, here, so a row's
// `k` is available to every reader -- constexprs_text, the launch, and the test -- without any
// of them having to know how the numbers are made.
std::vector<CodeSpec> buildTable()
{
    std::vector<CodeSpec> v(TABLE, TABLE + sizeof(TABLE) / sizeof(TABLE[0]));
    for (CodeSpec &c : v)
        if (c.op == Op::Xyz || c.op == Op::Lab || c.op == Op::Luv)
            coeffs_for(c.code, c.k);
    return v;
}

const std::vector<CodeSpec> &table()
{
    static const std::vector<CodeSpec> v = buildTable();
    return v;
}

} // namespace

const CodeSpec *find(int code)
{
    for (const CodeSpec &c : table())
        if (c.code == code)
            return &c;
    return nullptr;
}

const std::vector<CodeSpec> &all()
{
    return table();
}

namespace
{
// Volatile stores enforce one binary32 rounding per operation even in FMA builds.
float separate_mul(float a, float b)
{
    volatile float v = a * b;
    return v;
}
float separate_add(float a, float b)
{
    volatile float v = a + b;
    return v;
}
float separate_sub(float a, float b)
{
    volatile float v = a - b;
    return v;
}

cv::Mat separate4_model(const cv::Mat &src, const CodeSpec &c)
{
    cv::Mat dst(src.rows, src.cols, CV_MAKETYPE(CV_32F, c.dcn));
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x)
        {
            const float *s = src.ptr<float>(y) + x * c.scn;
            float *d = dst.ptr<float>(y) + x * c.dcn;
            if (c.op == Op::Gray || !c.inverse)
            {
                const float a = separate_mul(s[0], c.blue == 0 ? .114f : .299f);
                const float b = separate_mul(s[1], .587f);
                const float e = separate_mul(s[2], c.blue == 0 ? .299f : .114f);
                const float v = c.op != Op::Gray && x < (src.cols / 4) * 4
                                    ? separate_add(a, separate_add(b, e))
                                    : separate_add(separate_add(a, b), e);
                d[0] = v;
                if (c.op != Op::Gray)
                {
                    const bool uv = c.op == Op::Yuv;
                    const float cr = separate_add(
                        separate_mul(separate_sub(s[c.blue ^ 2], v), uv ? .877f : .713f), .5f);
                    const float cb = separate_add(
                        separate_mul(separate_sub(s[c.blue], v), uv ? .492f : .564f), .5f);
                    d[uv ? 2 : 1] = cr;
                    d[uv ? 1 : 2] = cb;
                }
            }
            else
            {
                const bool uv = c.op == Op::Yuv;
                const float cr = separate_sub(s[uv ? 2 : 1], .5f);
                const float cb = separate_sub(s[uv ? 1 : 2], .5f);
                d[c.blue] = separate_add(s[0], separate_mul(cb, uv ? 2.032f : 1.773f));
                d[1] = separate_add(separate_add(s[0], separate_mul(cb, uv ? -.395f : -.344f)),
                                    separate_mul(cr, uv ? -.581f : -.714f));
                d[c.blue ^ 2] = separate_add(s[0], separate_mul(cr, uv ? 1.140f : 1.403f));
                if (c.dcn == 4)
                    d[3] = 1.f;
            }
        }
    return dst;
}

using FloatControls = std::tuple<bool, bool, bool, int, unsigned>;
FloatControls float_controls()
{
    unsigned fp = 0;
#if defined(__SSE__)
    fp = _mm_getcsr() & 0xe040u;
#endif
    return std::make_tuple(cv::useOptimized(), cv::ipp::useIPP(), cv::ipp::useIPP_NotExact(),
                           std::fegetround(), fp);
}

bool identify_separate4()
{
    // A fixture match identifies only the tested finite normal domain. It is not a
    // universal guarantee for other HALs, nonfinite inputs, overflow or subnormals.
    int codes = 0;
    for (const CodeSpec &c : all())
    {
        if (c.op != Op::Gray && c.op != Op::Yuv && c.op != Op::YCrCb)
            continue;
        ++codes;
        for (int w : {1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 65})
            for (bool roi : {false, true})
            {
                cv::Mat parent(5, w + 7, CV_MAKETYPE(CV_32F, c.scn));
                cv::RNG rng(0x932500 + c.code * 31 + w);
                rng.fill(parent, cv::RNG::UNIFORM, -2., 2.);
                cv::Mat src = parent(cv::Rect(2, 1, w, 3));
                if (!roi)
                    src = src.clone();
                for (int y = 0; y < src.rows; ++y)
                    for (int x = 0; x < w; ++x)
                        for (int k = 0; k < c.scn; ++k)
                            src.ptr<float>(y)[x * c.scn + k] *= (x + y) % 3 ? 32767.5f : 1.f;
                const cv::Mat expected = separate4_model(src, c);
                cv::Mat actual;
                cv::cvtColor(src, actual, c.code);
                if (actual.size() != expected.size() || actual.type() != expected.type())
                    return false;
                for (int y = 0; y < actual.rows; ++y)
                    for (int x = 0; x < actual.cols * actual.channels(); ++x)
                    {
                        const float a = actual.ptr<float>(y)[x], b = expected.ptr<float>(y)[x];
                        if (!std::isfinite(a) || !std::isfinite(b) || a != b)
                            return false;
                    }
            }
    }
    return codes == 12;
}
} // namespace

bool float_cpu_separate4()
{
    // IPP controls can be thread-local: cache by their values, not by a process
    // default. Never mutate caller controls. Do not publish an unstable probe.
    static std::mutex mutex;
    static std::map<FloatControls, bool> cache;
    const FloatControls before = float_controls();
    if (std::get<3>(before) != FE_TONEAREST || std::get<4>(before) != 0)
        return false;
    std::lock_guard<std::mutex> lock(mutex);
    if (float_controls() != before)
        return false;
    const auto found = cache.find(before);
    if (found != cache.end())
        return found->second && float_controls() == before;
    bool matches = false;
    try
    {
        matches = identify_separate4();
    }
    catch (const cv::Exception &)
    {
        return false;
    }
    if (float_controls() != before)
        return false;
    cache.emplace(before, matches);
    return matches;
}

Status resolve_output(const cv_triton_runtime::DeviceView &s, int code, Output &out)
{
    const CodeSpec *spec = find(code);
    if (!spec)
        return unsupported(Reason::Interpolation,
                           "cvtColor code " + std::to_string(code) +
                               " is not served by this module: see doc/cvtcolor.markdown for the "
                               "supported set and depth-specific numeric criteria");
    // Depth. CV_8U is the rule for the table; CV_16U is additionally served by the families whose
    // CPU path is the SAME integer formula at both depths, i.e. the same coefficient tables with
    // only the container widened:
    //
    //   Op::Bayer    demosaicing.cpp hands both depths to the same template, and the
    //                16U SIMD interpolator is a stub (the plain scalar loop).
    //   Op::Gray     RGB2Gray<ushort> uses BY15/GY15/RY15 and gray_shift=15 like the uchar
    //                specialisation (color_rgb.simd.hpp).
    //   Op::Yuv      RGB2YCrCb_i<ushort> / YCrCb2RGB_i<ushort>, yuv_shift = 14
    //   Op::YCrCb    (color_yuv.simd.hpp). Only the chroma bias
    //                (ColorChannel<T>::half() << shift) and the clamp (255 -> 65535) differ.
    //   Op::Permute  ONLY the gray->colour rows (scn == 1): Gray2RGB is a pure copy with
    //                alpha = ColorChannel<T>::max() (color_rgb.simd.hpp).
    //
    // Everything else stays 8U, and the colour permutes do NOT get 16U as a side effect: the
    // scn == 1 condition is what keeps BGR2BGRA and the YUV2GRAY_* extractChannel rows out.
    //
    // src and dst must agree: the CPU creates dst at src's depth (`CV_MAKE_TYPE(depth, dcn)`)
    // and has no mixed-depth path.
    const bool u8 = s.type == cv_triton_runtime::ElemType::U8;
    const bool u16 = s.type == cv_triton_runtime::ElemType::U16;
    const bool bayer = spec->op == Op::Bayer;
    const bool serves16U = bayer || spec->op == Op::Gray || spec->op == Op::Yuv ||
                           spec->op == Op::YCrCb || (spec->op == Op::Permute && spec->scn == 1);
    const bool f32 = s.type == cv_triton_runtime::ElemType::F32;
    // CV_32F copies preserve source bits. Arithmetic conversions allow
    // |got-ref| <= 8*2^-24*max(1,|ref|,max|input channels at this pixel|), accounting
    // for differences in CPU fused arithmetic and vector/scalar evaluation order.
    // YUV2GRAY_* are Op::Permute, but the CPU cvtColor wrapper admits CV_8U
    // only before calling extractChannel. Do not invent a 32F cvtColor extension.
    const bool f32_exact_copy = f32 && ((spec->op == Op::Permute && spec->scn != 2) ||
                                        (spec->op == Op::Gray && spec->inverse));
    const bool f32_arithmetic =
        f32 && (spec->op == Op::Gray || spec->op == Op::Yuv || spec->op == Op::YCrCb);
    if (!u8 && !(serves16U && u16) && !f32_exact_copy && !f32_arithmetic)
        return unsupported(
            Reason::DepthOrChannels,
            std::string(spec->name) +
                ": unsupported depth or mismatched src/dst depths for this conversion");
    if (s.channels != spec->scn)
        return unsupported(Reason::DepthOrChannels,
                           std::string(spec->name) + " needs a " + std::to_string(spec->scn) +
                               "-channel source, got " + std::to_string(s.channels));
    if (spec->op == Op::Yuv422 && s.cols % 2 != 0)
        return unsupported(Reason::EmptyOrSize,
                           std::string(spec->name) + ": 4:2:2 conversion needs an even width");
    // The expected destination size depends on the code. Most conversions keep it; a 4:2:0 decode
    // (NV12/NV21) takes ONE CV_8UC1 of (3H/2) x W and produces H x W, because the CPU's
    // CvtHelper FROM_YUV asserts `sz.width % 2 == 0 && sz.height % 3 == 0` and sets
    // `dstSz = Size(sz.width, sz.height * 2 / 3)` (color.simd_helpers.hpp). Checking
    // equality for those codes would reject every legal call, and -- worse -- a relaxed check
    // alone would accept a same-height destination and read the chroma plane out of range, so
    // the size is validated against the rule rather than merely not-equality.
    if (spec->op == Op::Yuv420)
    {
        if (s.cols % 2 != 0 || s.rows % 3 != 0)
            return unsupported(Reason::EmptyOrSize, std::string(spec->name) +
                                                        ": the 4:2:0 source must have an even "
                                                        "width and a height divisible by 3 (got " +
                                                        std::to_string(s.cols) + "x" +
                                                        std::to_string(s.rows) + ")");
    }
    const Status source = cv_triton_runtime::validate_view(s, "src");
    if (!source.ok())
        return source;
    const int rows = spec->op == Op::Yuv420 ? (s.rows / 3) * 2 : s.rows;
    const Status output = cv_triton_runtime::validate_output_shape(rows, s.cols, spec->dcn);
    if (!output.ok())
        return output;
    if (static_cast<long long>(rows) * s.cols > 0x7fffffffLL)
        return unsupported(Reason::IndexRange, "pixel count exceeds int32 indexing");
    // Admission precedes output allocation in every adapter. uv_rs is i32,
    // while the pointer arithmetic itself is 64-bit.
    if (spec->op == Op::Yuv420 &&
        static_cast<unsigned long long>(rows) * (s.step / cv_triton_runtime::elem_size(s.type)) >
            0x7fffffffULL)
        return unsupported(Reason::IndexRange, "4:2:0 chroma-plane offset exceeds int32 indexing");
    out.spec = spec;
    out.cols = s.cols;
    out.rows = rows;
    return prepared();
}

Status normalize(const Request &req, Normalized &out)
{
    const cv_triton_runtime::DeviceView &s = req.src, &d = req.dst;
    Output expected;
    const Status resolved = resolve_output(s, req.code, expected);
    if (!resolved.ok())
        return resolved;
    const CodeSpec *spec = expected.spec;
    if (s.type != d.type)
        return unsupported(
            Reason::DepthOrChannels,
            std::string(spec->name) +
                ": unsupported depth or mismatched src/dst depths for this conversion");
    if (d.channels != spec->dcn)
        return unsupported(Reason::DepthOrChannels,
                           std::string(spec->name) + " writes " + std::to_string(spec->dcn) +
                               " channels, dst has " + std::to_string(d.channels));
    if (d.rows != expected.rows || d.cols != expected.cols)
        return unsupported(Reason::EmptyOrSize,
                           std::string(spec->name) + " writes " + std::to_string(expected.cols) +
                               "x" + std::to_string(expected.rows) + ", dst is " +
                               std::to_string(d.cols) + "x" + std::to_string(d.rows));
    if (!(s.ctx == d.ctx))
        return unsupported(Reason::ContextMismatch, "src and dst belong to different contexts");
    const Status destination = cv_triton_runtime::validate_view(d, "dst");
    if (!destination.ok())
        return destination;
    if (cv_triton_runtime::views_overlap(s, d))
        return unsupported(Reason::InPlaceOrOverlap,
                           "src and dst overlap (in-place is not supported)");
    // The pixel count and the launch geometry are the DESTINATION's: grid() emits one program per
    // `n.rows` row, so leaving rows at the source height launches 1.5x the programs for a 4:2:0
    // decode and writes rows past the end of dst.
    const long long n = static_cast<long long>(d.rows) * d.cols;
    if (n > 0x7fffffffLL)
        return unsupported(Reason::IndexRange, "pixel count exceeds int32 indexing");
    out.spec = spec;
    out.rows = d.rows;
    out.cols = d.cols;
    out.n = n;
    out.u16 = s.type == cv_triton_runtime::ElemType::U16;
    out.f32 = s.type == cv_triton_runtime::ElemType::F32;
    return prepared();
}

HsvTables hsv_tables()
{
    // color_hsv.simd.hpp: hsv_shift = 12; sdiv = (255<<12)/i, hdiv = (range<<12)/(6*i),
    // both saturate_cast<int> of a double division, i = 1..255, entry 0 is 0.
    HsvTables t;
    t.sdiv.resize(256, 0);
    t.hdiv.resize(256, 0);
    for (int i = 1; i < 256; ++i)
    {
        t.sdiv[i] = static_cast<int>(std::lrint(1044480.0 / i));        // (255 << 12) / i
        t.hdiv[i] = static_cast<int>(std::lrint(737280.0 / (6.0 * i))); // (180 << 12) / (6 i)
    }
    return t;
}

HsvTables hsv_tables_full()
{
    // hdiv for the 0..255 hue range: (256 << 12) / (6 i).
    HsvTables t;
    t.sdiv.resize(256, 0);
    t.hdiv.resize(256, 0);
    for (int i = 1; i < 256; ++i)
    {
        t.sdiv[i] = static_cast<int>(std::lrint(1044480.0 / i));
        t.hdiv[i] = static_cast<int>(std::lrint(1048576.0 / (6.0 * i))); // (256 << 12) / (6 i)
    }
    return t;
}

// ---------------------------------------------------------------------------------------------
// XYZ / Lab / Luv constants, replicated from modules/imgproc/src/color_lab.cpp.
// Every value below is quoted from that file rather than re-derived, because the CPU's numbers
// are the specification here, not the textbook ones.
namespace
{

const int kLabShift = 12;  // == xyz_shift (color.simd_helpers.hpp), color_lab.cpp
const int kGammaShift = 3; // color_lab.cpp
const int kLabShift2 = 15; // lab_shift + gamma_shift (color_lab.cpp)
const int kInvGammaShift = 12;
const int kInvGammaTabSize = 1 << kInvGammaShift;
const int kLabCbrtTabSizeB = 256 * 3 / 2 * (1 << kGammaShift); // 3072
const int kLabBaseShift = 14;
const int kLabBase = 1 << kLabBaseShift;
const int kLutBase = 1 << 14;
const int kLabLutDim = 33; // (1 << 5) + 1
const int kMinAB = -8145;

using cv::softdouble;
using cv::softfloat;

// color_lab.cpp (applyGamma / applyInvGamma inside a softdouble evaluation).
softfloat applyGamma(softfloat x)
{
    static const softdouble threshold = softdouble(809) / softdouble(20000);
    static const softdouble lowScale = softdouble(323) / softdouble(25);
    static const softdouble power = softdouble(12) / softdouble(5);
    static const softdouble xshift = softdouble(11) / softdouble(200);
    const softdouble xd = x;
    return xd <= threshold ? xd / lowScale
                           : pow((xd + xshift) / (softdouble::one() + xshift), power);
}
softfloat applyInvGamma(softfloat x)
{
    static const softdouble threshold = softdouble(7827) / softdouble(2500000);
    static const softdouble lowScale = softdouble(323) / softdouble(25);
    static const softdouble power = softdouble(12) / softdouble(5);
    static const softdouble xshift = softdouble(11) / softdouble(200);
    const softdouble xd = x;
    return xd <= threshold
               ? xd * lowScale
               : pow(xd, softdouble::one() / power) * (softdouble::one() + xshift) - xshift;
}

// D65 and the sRGB matrices as raw bit patterns (color_lab.cpp). Raw and not
// decimal literals on purpose: the decimal comments in the source are rounded renderings and
// parsing them back gives different doubles.
const softdouble D65[3] = {softdouble::fromRaw(0x3fee6a22b3892ee8), softdouble::one(),
                           softdouble::fromRaw(0x3ff16b8950763a19)};
const softdouble kSRGB2XYZ_D65[9] = {
    softdouble::fromRaw(0x3fda65a14488c60d), softdouble::fromRaw(0x3fd6e297396d0918),
    softdouble::fromRaw(0x3fc71819d2391d58), softdouble::fromRaw(0x3fcb38cda6e75ff6),
    softdouble::fromRaw(0x3fe6e297396d0918), softdouble::fromRaw(0x3fb279aae6c8f755),
    softdouble::fromRaw(0x3f93cc4ac6cdaf4b), softdouble::fromRaw(0x3fbe836eb4e98138),
    softdouble::fromRaw(0x3fee68427418d691)};
const softdouble kXYZ2sRGB_D65[9] = {
    softdouble::fromRaw(0x4009ec804102ff8f), softdouble::fromRaw(0xbff8982a9930be0e),
    softdouble::fromRaw(0xbfdfe7ff583a53b9), softdouble::fromRaw(0xbfef042528ae74f3),
    softdouble::fromRaw(0x3ffe040f23897204), softdouble::fromRaw(0x3fa546d3f9e7b80b),
    softdouble::fromRaw(0x3fac7de5082cf52c), softdouble::fromRaw(0xbfca1e14bdfd2631),
    softdouble::fromRaw(0x3ff0eabef06b3786)};

// XYZ's integer matrices are these literals directly: the 8-bit path never scales the floats
// (color_lab.cpp).
const int kSRGB2XYZ_i[9] = {1689, 1465, 739, 871, 2929, 296, 79, 488, 3892};
const int kXYZ2sRGB_i[9] = {13273, -6296, -2042, -3970, 7684, 170, 228, -836, 4331};

// The white-point chroma initLUTforLUV receives (color_lab.cpp). Note `dd` is a
// softfloat, not a softdouble, because `dd` was declared softfloat and std::max therefore
// returns softfloat -- so the double-valued reciprocal is ROUNDED TO FLOAT before it is used.
// That is why the value is 2.571912... and not the textbook 1/(0.9505+15+3*1.0888) = 0.0526.
void whiteChroma(softfloat &un, softfloat &vn)
{
    softfloat dd = D65[0] + D65[1] * softdouble(15) + D65[2] * softdouble(3);
    dd = softfloat::one() / std::max(dd, softfloat::eps());
    un = dd * softfloat(13 * 4) * D65[0];
    vn = dd * softfloat(13 * 9) * D65[1];
}

} // namespace

void coeffs_for(int code, int out[9])
{
    // The CPU builds these in the FUNCTOR CONSTRUCTORS from softdouble at run time -- they are
    // not literals in the source (except XYZ's), so replicating the constructor is the only way
    // to get them.
    //
    // ONE OUTPUT LAYOUT FOR ALL CODE KINDS, and the kernel depends on it:
    //   forward:  out[3r + k] multiplies src[k]                (r = which output channel)
    //   reverse:  result[r] = out[3r + k] * xyz[k], stored at dst[r]
    // i.e. ROW r IS DESTINATION SLOT r, in slot order 0,1,2 = whatever the CPU writes there
    // (B,G,R for a BGR destination). Getting this wrong is not subtle numerically but IS subtle
    // to read: a transposed or row-permuted colour matrix still looks like a colour matrix.
    std::fill(out, out + 9, 0);
    const bool forward = (code == 44 || code == 45 || code == 74 || code == 75 || // BGR/RGB -> Lab
                          code == 50 || code == 51);                              // BGR/RGB -> Luv
    const bool xyz = (code == 32 || code == 33);
    const bool xyzInv = (code == 34 || code == 35);
    const bool luv =
        (code == 50 || code == 51 || code == 58 || code == 59 || code == 80 || code == 81);
    // blueIdx = swapBlue(code) ? 2 : 0. swapBlue is false for the BGR/XYZ codes and true for
    // their RGB twins (color.hpp), which is what makes the two directions of the SAME code
    // family differ.
    const bool rgbCode = (code == 33 || code == 35 || code == 45 || code == 75 || code == 51 ||
                          code == 57 || code == 59 || code == 79 || code == 81);
    const int blueIdx = rgbCode ? 2 : 0;
    static const softdouble lshift(1 << kLabShift);

    if (xyz)
    {
        // RGB2XYZ_i<uchar> (color_lab.cpp): the int table with swapBlueCoeffsCols for
        // blueIdx == 0. Column 0 of the CPU's array is then the coefficient of `b` = src[0], so
        // the CPU's order already IS our slot order and no extra work is needed beyond the swap.
        for (int r = 0; r < 3; ++r)
        {
            const int *row = kSRGB2XYZ_i + 3 * r;
            out[3 * r + 0] = (blueIdx == 0) ? row[2] : row[0];
            out[3 * r + 1] = row[1];
            out[3 * r + 2] = (blueIdx == 0) ? row[0] : row[2];
        }
        return;
    }
    if (xyzInv)
    {
        // XYZ2RGB_i<uchar> (color_lab.cpp) applies swapBlueCoeffsRows when blueIdx == 0 --
        // ROWS, not columns, and blueIdx 0 is the BGR-output code. The CPU writes dst[0] from
        // flat row 0, so a row swap moves which destination channel each matrix row feeds.
        for (int r = 0; r < 3; ++r)
        {
            const int srcRow = (blueIdx == 0) ? 2 - r : r;
            for (int c = 0; c < 3; ++c)
                out[3 * r + c] = kXYZ2sRGB_i[3 * srcRow + c];
        }
        return;
    }

    if (forward)
    {
        // RGB2Lab_b's constructor (color_lab.cpp). c[] is indexed by the LOGICAL
        // channel R, G, B and the constructor writes slot (blueIdx^2) for c[0], 1 for c[1],
        // blueIdx for c[2]. Our slot k reads src[k], which is the CPU's `b` for k = 0 whatever
        // blueIdx is (the CPU reads src[0] as b, src[1] as g, src[2] as r, and only the
        // coefficient placement changes). So slot 0 must carry the b coefficient: c[2] when
        // blueIdx == 0, c[0] when blueIdx == 2.
        for (int i = 0; i < 3; ++i)
        {
            softdouble c[3];
            for (int j = 0; j < 3; ++j)
                c[j] = kSRGB2XYZ_D65[i * 3 + j];
            const softdouble w = D65[i];
            out[3 * i + 0] = cvRound(lshift * c[blueIdx == 0 ? 2 : 0] / w);
            out[3 * i + 1] = cvRound(lshift * c[1] / w);
            out[3 * i + 2] = cvRound(lshift * c[blueIdx == 0 ? 0 : 2] / w);
        }
        return;
    }

    // Lab2RGBinteger / Luv2RGBinteger (color_lab.cpp).
    //
    // THE INDEX ORDER IS TRANSPOSED HERE: `c[j] = XYZ2sRGB_D65[i + j*3]`, against the forward
    // direction's `c[j] = M[i*3 + j]`. The reverse transform requires this transpose.
    //
    // c[j] is the j-th element of a COLUMN of the output matrix (it multiplies x, y, z in order
    // after the transpose), so it is written DOWN slot j. The CPU places c[j] at flat index
    // (i + slotj*3) with slotj = blueIdx, 1, blueIdx^2 for j = 0,1,2 -- i.e. for blueIdx 0 the
    // column order is (0,1,2) and for blueIdx 2 it is reversed to (2,1,0). Since the CPU writes
    // dst[r] from flat row r, blueIdx 2 exchanges which destination channel the outer columns
    // feed, which is the `row` reversal below.
    for (int i = 0; i < 3; ++i)
    {
        softdouble c[3];
        for (int j = 0; j < 3; ++j)
            c[j] = kXYZ2sRGB_D65[i + j * 3];
        // Lab scales by the white point; Luv does not (its white point is fixed at D65 and was
        // already folded into un/vn when the tables were built).
        const softdouble w = luv ? softdouble::one() : D65[i];
        for (int j = 0; j < 3; ++j)
        {
            const int dstRow = (blueIdx == 2) ? j : 2 - j;
            out[dstRow * 3 + i] = cvRound(lshift * c[j] * w);
        }
    }
}

namespace
{

LabLuvTables buildLabLuvTables()
{
    LabLuvTables t;
    t.gamma.resize(256);
    for (int i = 0; i < 256; ++i)
    {
        const softfloat x = softfloat(i) / softfloat(255);
        t.gamma[i] = cvRound(softfloat(255 * (1 << kGammaShift)) * applyGamma(x));
    }
    t.cbrt.resize(kLabCbrtTabSizeB);
    {
        // cbTabScale = 1/(255*8): the argument is i/2040 and the result is scaled by 2^15.
        static const softfloat cbTabScale(softfloat::one() /
                                          (softfloat(255) * softfloat(1 << kGammaShift)));
        static const softfloat lthresh = softfloat(216) / softfloat(24389);
        static const softfloat lscale = softfloat(841) / softfloat(108);
        static const softfloat lbias = softfloat(16) / softfloat(116);
        static const softfloat lshift2(1 << kLabShift2);
        for (int i = 0; i < kLabCbrtTabSizeB; ++i)
        {
            const softfloat x = cbTabScale * softfloat(i);
            t.cbrt[i] = cvRound(lshift2 * (x < lthresh ? mulAdd(x, lscale, lbias) : cbrt(x)));
        }
    }
    t.yf.resize(256 * 2);
    for (int i = 0; i < 256; ++i)
    {
        int y, ify;
        if (i <= 20)
        { // 8 * 255 / 100 == 20.4: the CIE piecewise breakpoint on the 0..255 scale
            y = cvRound(softfloat(i * kLutBase * 20 * 9) / softfloat(17 * 29 * 29 * 29));
            ify = cvRound(softfloat((int)kLutBase) * (softfloat(16) / softfloat(116) +
                                                      softfloat(i * 5) / softfloat(3 * 17 * 29)));
        }
        else
        {
            const softfloat fy = softfloat(i * 100 * kLutBase) / softfloat(255 * 116) +
                                 softfloat(16 * kLutBase) / softfloat(116);
            ify = cvRound(fy);
            y = cvRound(fy * fy * fy / softfloat(kLutBase * kLutBase));
        }
        t.yf[i * 2] = y;
        t.yf[i * 2 + 1] = ify;
    }
    t.abxz.resize(kLabBase * 9 / 4);
    for (int i = kMinAB; i < kLabBase * 9 / 4 + kMinAB; ++i)
    {
        int v;
        // The two branches are f^-1 and t^3, meeting at 6/29 * BASE = 3389.73.
        if (i <= 3390)
            v = i * 108 / 841 - kLutBase * 16 / 116 * 108 / 841;
        else
            v = i * i / kLutBase * i / kLutBase;
        t.abxz[i - kMinAB] = v;
    }
    t.invgamma.resize(kInvGammaTabSize);
    for (int i = 0; i < kInvGammaTabSize; ++i)
    {
        static const softfloat invScale = softfloat::one() / softfloat(kInvGammaTabSize);
        t.invgamma[i] = cvRound(softfloat(255) * applyInvGamma(invScale * softfloat(i)));
    }
    softfloat un, vn;
    whiteChroma(un, vn);

    // initLUTforLUV (color_lab.cpp). LvToVpl_b uses `long long`
    // because intermediate values exceed the signed 32-bit range.
    t.lu.resize(256 * 256);
    t.lv.resize(256 * 256);
    t.lvpl.resize(256 * 256);
    {
        static const softfloat uLow(-134), uHigh(220);
        static const softfloat vLow(-140), vHigh(122);
        const softfloat uRange = uHigh - uLow, vRange = vHigh - vLow;
        const softfloat oneof4 = softfloat::one() / softfloat(4);
        for (int LL = 0; LL < 256; ++LL)
        {
            const softfloat L = softfloat(LL * 100) / softfloat(255);
            for (int uu = 0; uu < 256; ++uu)
            {
                const softfloat u = softfloat(uu) * uRange / softfloat(255) + uLow;
                const softfloat up = softfloat(9) * (u + L * un);
                t.lu[LL * 256 + uu] = cvRound(up * softfloat(kLutBase / 1024));
            }
            for (int vv = 0; vv < 256; ++vv)
            {
                const softfloat v = softfloat(vv) * vRange / softfloat(255) + vLow;
                softfloat vp = oneof4 / (v + L * vn);
                if (vp > oneof4)
                    vp = oneof4;
                if (vp < -oneof4)
                    vp = -oneof4;
                const int ivp = cvRound(vp * softfloat(kLutBase * 1024));
                t.lv[LL * 256 + vv] = ivp;
                // The CPU computes `int vpl = ivp*LL` in int32 and only then widens. Max |ivp*LL|
                // is 247463936 (verified), so int32 does not overflow there and the widening is
                // lossless; the multiply is done in long long here, which is the same value.
                const long long vpl = static_cast<long long>(ivp) * LL;
                t.lvpl[LL * 256 + vv] = (12 * 13 * 100 * (kLutBase / 1024)) * vpl;
            }
        }
    }

    // RGB2LuvLUT_s16 (color_lab.cpp): the 33^3 grid, in softfloat, then the corner
    // duplication fill_one applies so that the last node has a valid "up" neighbour.
    {
        const int n3 = kLabLutDim * kLabLutDim * kLabLutDim;
        t.luvLut.assign(static_cast<std::size_t>(n3) * 3 * 8, 0);
        std::vector<short> prev(static_cast<std::size_t>(n3) * 3, 0);

        static const softfloat lthresh = softfloat(216) / softfloat(24389);
        static const softfloat lscale = softfloat(841) / softfloat(108);
        static const softfloat lbias = softfloat(16) / softfloat(116);
        static const softfloat uLow(-134), uHigh(220);
        static const softfloat vLow(-140), vHigh(122);
        const softfloat uRange = uHigh - uLow, vRange = vHigh - vLow;
        static const softfloat lld(kLabLutDim - 1), f116(116), f16(16), f100(100);
        static const softfloat f128(128), f256(256), lbase((int)kLabBase);
        static const softfloat f9of4 = softfloat(9) / softfloat(4);
        static const softfloat f15(15), f3(3);

        // The Luv grid's coefficients are NOT white-point scaled: building the LUT divides by
        // the white point only for Lab (scaleWhite), and the Luv block uses `coeffs`, which is
        // sRGB2XYZ_D65 with R and B swapped (color_lab.cpp).
        softfloat c[9];
        for (int i = 0; i < 3; ++i)
        {
            softfloat s0 = kSRGB2XYZ_D65[i * 3 + 0];
            softfloat s1 = kSRGB2XYZ_D65[i * 3 + 1];
            softfloat s2 = kSRGB2XYZ_D65[i * 3 + 2];
            c[i * 3 + 2] = s0;
            c[i * 3 + 1] = s1;
            c[i * 3 + 0] = s2;
        }

        softfloat gammaTab[kLabLutDim];
        for (int n = 0; n < kLabLutDim; ++n)
            gammaTab[n] = applyGamma(softfloat(n) / lld);

        for (int p = 0; p < kLabLutDim; ++p)
            for (int q = 0; q < kLabLutDim; ++q)
                for (int r = 0; r < kLabLutDim; ++r)
                {
                    const int idx = p * 3 + q * kLabLutDim * 3 + r * kLabLutDim * kLabLutDim * 3;
                    const softfloat R = gammaTab[p], G = gammaTab[q], B = gammaTab[r];
                    const softfloat X = R * c[0] + G * c[1] + B * c[2];
                    const softfloat Y = R * c[3] + G * c[4] + B * c[5];
                    const softfloat Z = R * c[6] + G * c[7] + B * c[8];
                    // Luv's L uses the CIE branch on Y directly (not squared, unlike Lab's L).
                    softfloat L = Y < lthresh ? mulAdd(Y, lscale, lbias) : cbrt(Y);
                    L = L * f116 - f16;
                    const softfloat d =
                        softfloat(4 * 13) / std::max(X + f15 * Y + f3 * Z, softfloat(FLT_EPSILON));
                    const softfloat u = L * (X * d - un);
                    const softfloat v = L * (f9of4 * Y * d - vn);
                    prev[idx] = (short)cvRound(lbase * L / f100);
                    prev[idx + 1] = (short)cvRound(lbase * (u - uLow) / uRange);
                    prev[idx + 2] = (short)cvRound(lbase * (v - vLow) / vRange);
                }
        // fill_one: corner (_p,_q,_r) of node (p,q,r) takes the value of node
        // (min(p+_p, 32), min(q+_q, 32), min(r+_r, 32)) -- the duplication that gives the last
        // node an "up" neighbour -- and lands at 4*_p + 2*_q + _r inside the node's 24 int16,
        // with the channels at +0, +8, +16.
        for (int p = 0; p < kLabLutDim; ++p)
            for (int q = 0; q < kLabLutDim; ++q)
                for (int r = 0; r < kLabLutDim; ++r)
                    for (int pp = 0; pp < 2; ++pp)
                        for (int qq = 0; qq < 2; ++qq)
                            for (int rr = 0; rr < 2; ++rr)
                            {
                                const long long idxold =
                                    static_cast<long long>(std::min(p + pp, kLabLutDim - 1)) * 3 +
                                    static_cast<long long>(std::min(q + qq, kLabLutDim - 1)) *
                                        kLabLutDim * 3 +
                                    static_cast<long long>(std::min(r + rr, kLabLutDim - 1)) *
                                        kLabLutDim * kLabLutDim * 3;
                                const long long idxnew =
                                    static_cast<long long>(p) * 3 * 8 +
                                    static_cast<long long>(q) * kLabLutDim * 3 * 8 +
                                    static_cast<long long>(r) * kLabLutDim * kLabLutDim * 3 * 8 +
                                    4 * pp + 2 * qq + rr;
                                t.luvLut[idxnew] = prev[idxold];
                                t.luvLut[idxnew + 8] = prev[idxold + 1];
                                t.luvLut[idxnew + 16] = prev[idxold + 2];
                            }
    }
    return t;
}

} // namespace

const LabLuvTables &labluv_tables()
{
    // One process-wide build. The CPU builds these once too (initLabTabs' static local), and the
    // values do not depend on the code, the image or the device.
    static const LabLuvTables t = buildLabLuvTables();
    return t;
}

std::string signature(const Normalized &n)
{
    std::ostringstream o;
    // Order matters: this must be the kernel's own parameter order
    // (_cvtcolor_cpu_kernel(src, dst, cols, src_rs, dst_rs, sdiv, hdiv, ...)), and the compile
    // helper rejects a signature that disagrees with it.
    //
    // cols and the two row strides are runtime arguments rather than constexprs on purpose:
    // making them constexpr would compile a separate binary for every image width. They are
    // i32, and the two strides are in ELEMENTS of the pointer type, which is the addressing
    // unit the kernel indexes in: `row*rs + col*SCN` is an element offset from a typed pointer.
    // For CV_8U that is the same number as DeviceView::step; for CV_16U the launch divides the
    // byte step by 2. Getting this wrong is silent for 8U and a wrong row every time for 16U,
    // so the kernel states it as `ES` and divides by the same constant.
    // The pointer type must follow the element type, not just the "is it 16-bit" question: a 32F
    // buffer read through a *u8 pointer would interpret float storage as byte samples.
    const char *t = n.u16 ? "u16" : (n.f32 ? "fp32" : "u8");
    // uv_rs is the chroma plane base in ELEMENTS from `src`, i.e. H * src_rs (the Y plane's
    // height times the row pitch). It is a runtime argument like the strides, and it is only read
    // by Op::Yuv420; every other op ignores it, so one signature serves all codes.
    o << "src:*" << t << ", dst:*" << t
      << ", cols:i32, src_rs:i32, dst_rs:i32, sdiv:*i32, hdiv:*i32"
      << ", uv_rs:i32, rows:i32, t32:*i32, t16:*i16, t64:*i64";
    return o.str();
}

std::string constexprs_text(const Normalized &n)
{
    const CodeSpec &s = *n.spec;
    std::ostringstream o;
    o << "BLOCK=" << block_size(n) << ",BLUE=" << s.blue << ",DCN=" << s.dcn << ",GB=" << s.gb
      << ",I0=" << s.perm[0] << ",I1=" << s.perm[1] << ",I2=" << s.perm[2] << ",I3=" << s.perm[3]
      << ",HR=" << s.hrange << ",INV=" << (s.inverse ? 1 : 0) << ",OP=" << static_cast<int>(s.op)
      << ",SCN=" << s.scn << ",SWG=" << s.swg << ",U16=" << (n.u16 ? 1 : 0)
      << ",F32=" << (n.f32 ? 1 : 0) << ",F32_CPU_SEPARATE4=" << (n.f32_cpu_separate4 ? 1 : 0)
      << ",ORD=" << s.ord;
    // The nine coefficients are constexprs for the same reason the channel counts are: they
    // come from the CODE, so C++ computes them exactly and they belong in the key. The kernel
    // uses them only for Xyz/Lab/Luv, but the key carries them for every op -- an op that does
    // not read them has zeros there anyway (CodeSpec's default), so this costs nothing and keeps
    // one constexpr name set for all codes, which is what the compile helper requires.
    o << ",SRGB=" << (s.srgb ? 1 : 0);
    for (int i = 0; i < 9; ++i)
        o << ",E" << i << "=" << s.k[i];
    return o.str();
}

int block_size(const Normalized &n)
{
    // Output pixels per program along the row. A tuning constant: the grid is (rows,
    // ceil(cols/BLOCK)), so this is the only shape the kernel depends on and it stays fixed
    // to keep one binary per (SCN, DCN, OP, ...) rather than one per image size. 256 covers a
    // full row of most images in one program without spilling registers in the 3-channel
    // branches.
    (void)n;
    return 256;
}

int num_warps(const Normalized &n)
{
    (void)n;
    return 4;
}

void grid(const Normalized &n, int block, unsigned out[3])
{
    // One program per (row, column block). Rows go on x because y is limited to 65535 on
    // every supported SM generation and y would be the unbounded axis for a tall image.
    const long long b = block > 0 ? block : 256;
    out[0] = static_cast<unsigned>(n.rows);
    out[1] = static_cast<unsigned>((n.cols + b - 1) / b);
    out[2] = 1;
}

} // namespace cv_triton_color
