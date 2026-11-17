# Generated HIP cvtcolor kernels; regenerate with tools/make_kernels_cpu_hip.py.
# source_sha256: 3836188c45d23f89dfbe917020ed76d1e496c9202a72af754396179000d75592
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
#
# GPU colour conversions reproducing selected OpenCV CPU arithmetic.
# The cpu prefix denotes numerical compatibility; this file runs on the GPU.
# Source references below name OpenCV 5.0.0 imgproc files/symbols; row strides are
# typed elements. The host CodeSpec supplies operation and coefficient constexprs.
#
# Integer conversions preserve OpenCV CPU arithmetic. Constants and operation order
# follow modules/imgproc/src/color*.simd_helpers.hpp, color_*.simd.hpp and color_lab.cpp.
# Floating-point conversions use the operation groupings described at each branch.
#
# DELIBERATELY ABSENT: HSV2BGR and the HLS pair. Those are float32 in OpenCV, and the same
# build rounds differently in its SIMD body than in its scalar tail (SIMD truncates the
# final *255, the tail rounds). The answer for one pixel therefore depends on where it sits
# in the row. A fixed CPU build/input is deterministic, but these conversions are outside
# this module's selected portable numerical target and are refused explicitly.
#
# Specialisation policy is the same as the resize kernels: channel counts and the operation
# are constexprs, every other runtime argument is do_not_specialize, so the compile key is
# something C++ can compute exactly.

import triton
import triton.language as tl
from triton.language.extra import libdevice


# HIP/AMDGCN replacements for missing libdevice primitives: add_rn, mul_rn, sub_rn.
# Explicit vector arithmetic preserves separate add/multiply/subtract operations.

@triton.jit
def _triton_add_rn(a, b):
    return tl.inline_asm_elementwise("v_add_f32 $0, $1, $2", "=v,v,v", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _triton_mul_rn(a, b):
    return tl.inline_asm_elementwise("v_mul_f32 $0, $1, $2", "=v,v,v", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _triton_sub_rn(a, b):
    return tl.inline_asm_elementwise("v_sub_f32 $0, $1, $2", "=v,v,v", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


# Operation ids. tl.constexpr, not plain ints: Triton rejects a global that is not a
# constexpr inside a kernel, and each id becoming its own binary is what lets the branches
# below fold away instead of costing a runtime compare per pixel.
OP_GRAY = tl.constexpr(0)      # BGR/RGB (3 or 4 ch) -> gray (1 or 4 ch)
OP_PERMUTE = tl.constexpr(1)   # a pure channel gather/insert (BGR<->RGB, add/drop alpha)
OP_YUV = tl.constexpr(2)       # BGR/RGB <-> YUV
OP_YCRCB = tl.constexpr(3)     # BGR/RGB <-> YCrCb
OP_HSV = tl.constexpr(4)       # BGR/RGB -> HSV (integer)
OP_PACKED = tl.constexpr(5)    # BGR/RGB(A)/GRAY <-> BGR565/BGR555 (CV_8UC2 pixels)
OP_BAYER = tl.constexpr(6)     # Bayer BG/GB/RG/GR -> BGR/BGRA/GRAY (8U and 16U)
OP_XYZ = tl.constexpr(7)       # BGR/RGB <-> XYZ (integer, xyz_shift = 12)
OP_LAB = tl.constexpr(8)       # BGR/RGB <-> Lab (integer, lab_shift = 12)
OP_LUV = tl.constexpr(9)       # BGR/RGB <-> Luv (integer; sRGB forward is a 33^3 LUT)
OP_YUV422 = tl.constexpr(10)   # 4:2:2 interleaved UYVY/YUY2/YVYU <-> colour (CV_8UC2 pixels)
OP_YUV420 = tl.constexpr(11)   # NV12/NV21 semi-planar 4:2:0 -> colour (one CV_8UC1, 3H/2 rows)

# Every constant a kernel body touches must be a tl.constexpr: Triton rejects a plain
# module global inside @triton.jit. (Annotating with `: tl.constexpr` is NOT the same
# thing and does not work.) Values are OpenCV's own.

# color.simd_helpers.hpp
GRAY_SHIFT = tl.constexpr(15)
BY15 = tl.constexpr(3735)                # == B2YF * 32768 + 0.5
GY15 = tl.constexpr(19235)
RY15 = tl.constexpr(9798)
YUV_SHIFT = tl.constexpr(14)
R2Y = tl.constexpr(4899)                 # == R2YF * 16384
G2Y = tl.constexpr(9617)
B2Y = tl.constexpr(1868)
HSV_SHIFT = tl.constexpr(12)

# color_yuv.simd.hpp: forward chroma and inverse coefficients.
YCRI = tl.constexpr(11682)
YCBI = tl.constexpr(9241)
R2VI = tl.constexpr(14369)
B2UI = tl.constexpr(8061)
V2RI = tl.constexpr(18678)
V2GI = tl.constexpr(-9519)
U2GI = tl.constexpr(-6472)
U2BI = tl.constexpr(33292)
CR2RI = tl.constexpr(22987)
CR2GI = tl.constexpr(-11698)
CB2GI = tl.constexpr(-5636)
CB2BI = tl.constexpr(29049)
CHROMA_DELTA = tl.constexpr(128 << 14)

# color_yuv.simd.hpp -- the 4:2:2 / 4:2:0 path's own 20-bit BT.601 constants. These are
# NOT the yuv_shift=14 fixed-point ones above: the interleaved formats use a wider shift and
# different rounding, so sharing the constants would be silently wrong.
BT_CY = tl.constexpr(1220542)
BT_CUB = tl.constexpr(2116026)
BT_CUG = tl.constexpr(-409993)
BT_CVG = tl.constexpr(-852492)
BT_CVR = tl.constexpr(1673527)
BT_SHIFT = tl.constexpr(20)
BT_HALF = tl.constexpr(1 << 19)

# demosaicing.cpp -- Bayer2Gray_. Note G2Y is the same 9617 as the luma one above but
# is a separate constant in the CPU (a different function, its own SHIFT); kept separate here
# so that a change to one cannot silently move the other.
BAYER_R2Y = tl.constexpr(4899)
BAYER_B2Y = tl.constexpr(1868)
BAYER_G2Y = tl.constexpr(9617)
BAYER_SHIFT = tl.constexpr(14)

# --------------------------------------------------------------------------------------------
# XYZ / Lab / Luv (color_lab.cpp). Not "one entry per code": the coefficient tables are built
# by the CPU constructors from softdouble, so their nine ints are computed in C++ and arrive
# here as constexprs (C0..C8). The LUTs -- gamma, cbrt, LabToYF, abToXZ, the two Luv tables and
# the 33^3 Luv LUT -- are device data like sdiv/hdiv, because their entries are cvRound() of a
# softfloat evaluation and cannot be recomputed on the device without reimplementing softfloat.
#
# Shifts, from color_lab.cpp: lab_shift is xyz_shift (12), gamma_shift 3, so
# lab_shift2 15; the reverse path's `shift` is lab_shift + (14 - inv_gamma_shift) = 14 because
# it starts from a BASE=2^14 X/Y/Z. LAB_BASE is 2^14 and LUV_BASE=1024 is the Luv tables' own
# scale (initLUTforLUV: up is stored as up*(LUT_BASE/1024), i.e. 16x, and vp as vp*LUT_BASE*1024).
XYZ_SHIFT = tl.constexpr(12)
LAB_SHIFT = tl.constexpr(12)
LAB_SHIFT2 = tl.constexpr(15)      # lab_shift + gamma_shift
BASE = tl.constexpr(16384)         # LAB_BASE == LUT_BASE == 1 << 14
REV_SHIFT = tl.constexpr(14)       # lab_shift + (base_shift - inv_gamma_shift)
INV_GAMMA_MAX = tl.constexpr(4095) # INV_GAMMA_TAB_SIZE - 1
LSCALE = tl.constexpr(296)         # (116*255+50)/100
LSHIFT = tl.constexpr(-1336934)    # -((16*255*(1<<lab_shift2)+50)/100)
AB_DELTA = tl.constexpr(128 << 15) # 128*(1 << lab_shift2)
MIN_AB = tl.constexpr(-8145)       # minABvalue, abToXZ_b's index origin
# Lab2RGBinteger::process fixed-point approximations of a/500 and b/200 (color_lab.cpp),
# plus the two biases the CPU subtracts. 128*BASE/500 = 4194 and 128*BASE/200 = 10485, both
# integer divisions in the CPU, and the b path adds 1 AFTER subtracting -- that +1 is in the
# source and matters, so it is folded into the bias here.
A_MUL = tl.constexpr(53687)
B_MUL = tl.constexpr(41943)
ADIV_BIAS = tl.constexpr(4194)
BDIV_BIAS = tl.constexpr(10484)
# Luv 8-bit chroma ranges (color_lab.cpp) as the LUT node evaluation uses them.
LUV_BASE = tl.constexpr(1024)
# RGB2Luvinterpolate's grid (color_lab.cpp): 2^5+1 nodes per axis, baseDiv
# LAB_BASE/256 = 64, so a uchar channel maps to node >> 3 and fraction (2*ch) & 15.
LUT_DIM = tl.constexpr(33)
LUT_SHIFT = tl.constexpr(3)        # channel -> node index
BASE_DIV = tl.constexpr(64)        # LAB_BASE/256: the trilinear result is descaled by this
TRI_SHIFT = tl.constexpr(12)       # trilinear_shift*3 = 4*3, the weight total 16^3
LUV_LUT_STRIDE = tl.constexpr(3 * 8 * LUT_DIM)     # int16 per node along q
LUV_LUT_STRIDE2 = tl.constexpr(3 * 8 * LUT_DIM * LUT_DIM)   # ... along r

# Table layout, in ENTRIES of the pointer's own element type. The host packs these in this
# order (cvtcolor.cpp, cvt_tables()); the two must agree or every pixel is wrong, so the first
# thing that would fail is the numeric test.
#
#   t16 (int16)  [0, 862488)              RGB2LuvLUT_s16: the 33^3 trilinear Luv grid
#   t32 (int32)  [0, 256)                 sRGBGammaTab_b / linearGammaTab_b
#                [256, 3328)              LabCbrtTab_b (widths: the CPU's ushort is widened)
#                [3328, 3840)             LabToYF_b (2 per L)
#                [3840, 15104)            abToXZ_b, pre-shifted by -minABvalue
#                [15104, 19200)           sRGBInvGammaTab_b
#                [19200, 84736)           Luv LuToUp_b   (256*256)
#                [84736, 150272)          Luv LvToVp_b   (256*256)
#   t64 (int64)  [0, 65536)               Luv LvToVpl_b
#
# LabCbrtTab_b is int32 and not int16 on purpose: its entries reach 37555, which does not fit a
# signed 16-bit load, and the CPU's own array is `ushort` for that reason. Widening it host-side
# costs 8 KiB and removes an unsigned-interpretation trap from the kernel.
T_GAMMA = tl.constexpr(0)
T_CBRT = tl.constexpr(256)
T_YF = tl.constexpr(256 + 3072)
T_ABXZ = tl.constexpr(256 + 3072 + 512)
# abToXZ_b is LAB_BASE*9/4 = 36864 entries -- NOT the ~11264 the index range suggests, because
# the CPU allocates for the whole i = minABvalue .. minABvalue + LAB_BASE*9/4 span
# (color_lab.cpp). Sizing it from the observed indices misaligns every table after it.
T_INVGAMMA = tl.constexpr(256 + 3072 + 512 + 36864)
T_LU = tl.constexpr(T_INVGAMMA + 4096)
T_LV = tl.constexpr(T_LU + 65536)

DEFAULT_CONFIG = {"BLOCK": 256, "num_warps": 4}


# Compile-time groups (fields of CodeSpec in src/runtime/cvtcolor.hpp):
# SCN/DCN: source/destination channels; OP/BLUE: family and channel order/Bayer polarity.
# I0..I3: output channel sources (-1 inserts opaque alpha); INV: conversion direction.
# HR: integer hue range (180/256); GB: packed green bits (5/6).
# SWG: Bayer start-with-green phase; ORD: packed/semi-planar chroma byte order.
# E0..E8: row-major 3x3 coefficients, already channel-swapped by the host.
# U16/F32 select depth; F32_CPU_SEPARATE4 selects the identified CPU arithmetic profile.
# Runtime cols/rows count output pixels; uv_rs is the chroma-plane element offset,
# not a row stride. sdiv/hdiv and t32/t16/t64 are typed lookup-table pointers.
@triton.jit
def _cvtcolor_cpu_kernel(src, dst, cols, src_rs, dst_rs, sdiv, hdiv, uv_rs, rows,
                         t32, t16, t64,
                         SCN: tl.constexpr, DCN: tl.constexpr,
                         OP: tl.constexpr, BLUE: tl.constexpr,
                         I0: tl.constexpr, I1: tl.constexpr, I2: tl.constexpr,
                         I3: tl.constexpr, INV: tl.constexpr, HR: tl.constexpr,
                         GB: tl.constexpr, U16: tl.constexpr, F32: tl.constexpr,
                         F32_CPU_SEPARATE4: tl.constexpr,
                         SWG: tl.constexpr,
                         ORD: tl.constexpr,
                         E0: tl.constexpr, E1: tl.constexpr, E2: tl.constexpr,
                         E3: tl.constexpr, E4: tl.constexpr, E5: tl.constexpr,
                         E6: tl.constexpr, E7: tl.constexpr, E8: tl.constexpr,
                         SRGB: tl.constexpr,
                         BLOCK: tl.constexpr):
    """BLOCK output pixels of ONE row per program.

    Two dimensions rather than a flat pass over pixels: a GpuMat's row pitch is not
    cols*channels (OpenCV pads the allocation -- a 4x3 CV_8UC4 GpuMat reports step=512),
    so a flat index must divide by cols to find the row. Addressing row and column
    separately costs nothing and removes the division, and it is the same shape the
    resize kernels use.

    `I0..I3` are the destination channel sources for OP_PERMUTE only (-1 = opaque alpha for the output depth).
    `BLUE` is the blue index (0 for BGR order, 2 for RGB). `INV` selects the inverse
    direction for OP_YUV / OP_YCRCB, and the decode direction for OP_PACKED. `HR` is the
    hue range: 180 for BGR2HSV, 256 for BGR2HSV_FULL -- the CPU adds hr to a negative hue,
    so the two differ by this constant and not only by the hdiv table. `GB` is the packed
    format's green-bit count (6 for 565, 5 for 555) and is unused by every other op.

    `E0..E8` are the 3x3 coefficient table of OP_XYZ / OP_LAB / OP_LUV, already in the order
    the CPU's constructor produces -- including the blueIdx swap, which the CPU applies to the
    matrix itself and which is therefore a property of the CODE, not something this kernel
    re-derives. `SRGB` is false for the linear ("L") codes, which differ only in whether the
    transfer function is applied: linearRGB means tab[i] = 8*i, i.e. no gamma table at all.

    Three table pointers, not one, because their element types differ and a pointer's type is
    what Triton uses for the load: `t32` is int32 (abToXZ_b, the two 256*256 Luv tables),
    `t16` is int16 (the 33^3 Luv LUT), and `t64` is int64 (LvToVpl_b, which reaches
    6.2e13 at L=255). LabCbrtTab_b uses `t32` because its values exceed signed int16;
    see cvtcolor.cpp.
    Unused pointers are still passed a live allocation; per-code constexprs mean the dead ones
    are never dereferenced."""
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = col < cols
    sp = row * src_rs + col * SCN
    dp = row * dst_rs + col * DCN

    if OP == OP_GRAY:
        # OpenCV 5.0.0 color_rgb.simd.hpp, RGB2Gray<uchar/ushort>: the same
        # BY15/GY15/RY15 coefficients sum to 2^15 at both depths. CV_DESCALE adds
        # the rounding bias before shifting. Even 65535*32768 + 16384 fits int32.
        # CPU SIMD fixmul corrections compensate for its int16 packing; this GPU
        # path retains int32 intermediates until the store and needs no such correction.
        if F32:
            # CV_32F. The CPU does NOT have one expression here: its fused body (W >= 16) is
            # fma(r, cr, fma(g, cg, b*cb)) and its fused tail (W < 16) is
            # fma(r, cr, fma(b, cb, g*cg)) -- different groupings, 1 ULP apart -- and its non-FMA
            # build is the unfused (b*cb + g*cg) + r*cr. There is consequently no bit-exact
            # target across all CPU configurations. The unidentified-profile fallback uses
            # the body expression; identified separate4 uses source-order operations below.
            # Tests bound the absolute error by
            # 8 * 2^-24 * max(1, |reference|, max|input|).
            b = tl.load(src + sp + BLUE, mask=m, other=0).to(tl.float32)
            g = tl.load(src + sp + 1, mask=m, other=0).to(tl.float32)
            r = tl.load(src + sp + (BLUE ^ 2), mask=m, other=0).to(tl.float32)
            # No coefficient swap between BGR and RGB order: `b` was loaded through
            # `sp + BLUE`, so it holds the BLUE channel in both orders and `r` holds red. Swapping
            # the literals here (as the CPU's own blueIdx does when it swaps pointers instead of
            # indices) would apply the blue/red permutation twice.
            if F32_CPU_SEPARATE4:
                # Source-channel order matters even without fusion (RGB vs BGR).
                a = tl.load(src + sp, mask=m, other=0).to(tl.float32)
                e = tl.load(src + sp + 2, mask=m, other=0).to(tl.float32)
                if BLUE == 0:
                    p0, p2 = _triton_mul_rn(a, 0.114), _triton_mul_rn(e, 0.299)
                else:
                    p0, p2 = _triton_mul_rn(a, 0.299), _triton_mul_rn(e, 0.114)
                y = _triton_add_rn(_triton_add_rn(p0, _triton_mul_rn(g, 0.587)), p2)
            else:
                y = libdevice.fma(r, 0.299, libdevice.fma(g, 0.587, b * 0.114))
            tl.store(dst + dp, y, mask=m)
            if DCN == 4:
                tl.store(dst + dp + 3, tl.full((BLOCK,), 1.0, tl.float32), mask=m)
        else:
            b = tl.load(src + sp + BLUE, mask=m, other=0).to(tl.int32)
            g = tl.load(src + sp + 1, mask=m, other=0).to(tl.int32)
            r = tl.load(src + sp + (BLUE ^ 2), mask=m, other=0).to(tl.int32)
            y = (b * BY15 + g * GY15 + r * RY15 + (1 << (GRAY_SHIFT - 1))) >> GRAY_SHIFT
            tl.store(dst + dp, y.to(dst.dtype.element_ty), mask=m)
        if DCN == 4:
            # Alpha<T> is 255 at 8U and 65535 at 16U (color.simd_helpers.hpp; the 16U
            # invoker writes ColorChannel<ushort>::max() unchanged).
            tl.store(dst + dp + 3,
                     tl.full((BLOCK,), 65535 if U16 else 255, tl.int32).to(dst.dtype.element_ty),
                     mask=m)
    elif OP == OP_PERMUTE:
        # The inserted channel is ColorChannel<T>::max(), which is depth-dependent: 255 at 8U,
        # 65535 at 16U, 1.0f at 32F. Writing the 8U literal at another depth is a silent
        # truncation (a 32F alpha of 255.0f would pass an 8-bit "looks right" check and fail the
        # CPU comparison), so it is derived from the pointer type rather than hard-coded.
        if src.dtype.element_ty == tl.float32:
            alpha = tl.full((BLOCK,), 1.0, tl.float32)
        elif U16:
            alpha = tl.full((BLOCK,), 65535, tl.int32).to(tl.uint16)
        else:
            alpha = tl.full((BLOCK,), 255, tl.int32).to(tl.uint8)
        if I0 >= 0:
            tl.store(dst + dp + 0, tl.load(src + sp + I0, mask=m, other=0), mask=m)
        else:
            tl.store(dst + dp + 0, alpha, mask=m)
        if DCN > 1:
            if I1 >= 0:
                tl.store(dst + dp + 1, tl.load(src + sp + I1, mask=m, other=0), mask=m)
            else:
                tl.store(dst + dp + 1, alpha, mask=m)
        if DCN > 2:
            if I2 >= 0:
                tl.store(dst + dp + 2, tl.load(src + sp + I2, mask=m, other=0), mask=m)
            else:
                tl.store(dst + dp + 2, alpha, mask=m)
        if DCN > 3:
            if I3 >= 0:
                tl.store(dst + dp + 3, tl.load(src + sp + I3, mask=m, other=0), mask=m)
            else:
                tl.store(dst + dp + 3, alpha, mask=m)
    elif OP == OP_YUV or OP == OP_YCRCB:
        # Depth. The CPU's 16U functors use the SAME 14-bit table as the 8U ones -- R2Y/G2Y/B2Y,
        # YCRI/YCBI, R2VI/B2UI and the four inverse coefficients are shared, only the container
        # widens (RGB2YCrCb_i<ushort>, color_yuv.simd.hpp; YCrCb2RGB_i<ushort>).
        # Two things DO change with depth:
        #   * the chroma bias: ColorChannel<T>::half() << shift, i.e. 128<<14 at 8U and
        #     32768<<14 at 16U (color_yuv.simd.hpp `sdelta`, `delta`), and
        #   * the clamp: 255 at 8U, 65535 at 16U. It is reachable at 16U -- the forward chroma
        #     numerator (r-Y)*C3 + sdelta peaks near 1.3e9 before the shift -- while at 8U the
        #     same expression cannot leave the byte range.
        if not INV and F32:
            # CV_32F forward. Same coefficient literals as the CPU (color_yuv.simd.hpp),
            # same fused grouping as its W>=16 body. Compute y first and derive chroma from
            # the floating-point y, rather than the rounded value used by the integer path.
            b = tl.load(src + sp + BLUE, mask=m, other=0).to(tl.float32)
            g = tl.load(src + sp + 1, mask=m, other=0).to(tl.float32)
            r = tl.load(src + sp + (BLUE ^ 2), mask=m, other=0).to(tl.float32)
            if F32_CPU_SEPARATE4:
                a = tl.load(src + sp, mask=m, other=0).to(tl.float32)
                e = tl.load(src + sp + 2, mask=m, other=0).to(tl.float32)
                if BLUE == 0:
                    p0, p2 = _triton_mul_rn(a, 0.114), _triton_mul_rn(e, 0.299)
                else:
                    p0, p2 = _triton_mul_rn(a, 0.299), _triton_mul_rn(e, 0.114)
                p1 = _triton_mul_rn(g, 0.587)
                body = _triton_add_rn(p0, _triton_add_rn(p1, p2))
                tail = _triton_add_rn(_triton_add_rn(p0, p1), p2)
                y = tl.where(col < (cols // 4) * 4, body, tail)
                if OP == OP_YUV:
                    c0 = _triton_add_rn(_triton_mul_rn(_triton_sub_rn(r, y), 0.877), 0.5)
                    c1 = _triton_add_rn(_triton_mul_rn(_triton_sub_rn(b, y), 0.492), 0.5)
                else:
                    c0 = _triton_add_rn(_triton_mul_rn(_triton_sub_rn(r, y), 0.713), 0.5)
                    c1 = _triton_add_rn(_triton_mul_rn(_triton_sub_rn(b, y), 0.564), 0.5)
            else:
                y = libdevice.fma(b, 0.114, libdevice.fma(g, 0.587, r * 0.299))
                if OP == OP_YUV:
                    c0 = libdevice.fma(r - y, 0.877, 0.5)
                    c1 = libdevice.fma(b - y, 0.492, 0.5)
                else:
                    c0 = libdevice.fma(r - y, 0.713, 0.5)
                    c1 = libdevice.fma(b - y, 0.564, 0.5)
            tl.store(dst + dp, y, mask=m)
            if OP == OP_YUV:
                tl.store(dst + dp + 1, c1, mask=m)   # [Y, U, V]
                tl.store(dst + dp + 2, c0, mask=m)
            else:
                tl.store(dst + dp + 1, c0, mask=m)   # [Y, Cr, Cb]
                tl.store(dst + dp + 2, c1, mask=m)
        elif INV and F32:
            # CV_32F inverse. y is added outside the chroma term, exactly as the CPU does
            # (color_yuv.simd.hpp); 0.5 is ColorChannel<float>::half() exactly.
            y = tl.load(src + sp + 0, mask=m, other=0).to(tl.float32)
            if F32_CPU_SEPARATE4:
                if OP == OP_YUV:
                    cb = _triton_sub_rn(tl.load(src + sp + 1, mask=m, other=0).to(tl.float32), 0.5)
                    cr = _triton_sub_rn(tl.load(src + sp + 2, mask=m, other=0).to(tl.float32), 0.5)
                    bb = _triton_add_rn(y, _triton_mul_rn(cb, 2.032))
                    gg = _triton_add_rn(_triton_add_rn(y, _triton_mul_rn(cb, -0.395)), _triton_mul_rn(cr, -0.581))
                    rr = _triton_add_rn(y, _triton_mul_rn(cr, 1.140))
                else:
                    cr = _triton_sub_rn(tl.load(src + sp + 1, mask=m, other=0).to(tl.float32), 0.5)
                    cb = _triton_sub_rn(tl.load(src + sp + 2, mask=m, other=0).to(tl.float32), 0.5)
                    bb = _triton_add_rn(y, _triton_mul_rn(cb, 1.773))
                    gg = _triton_add_rn(_triton_add_rn(y, _triton_mul_rn(cb, -0.344)), _triton_mul_rn(cr, -0.714))
                    rr = _triton_add_rn(y, _triton_mul_rn(cr, 1.403))
            else:
                if OP == OP_YUV:
                    cb = tl.load(src + sp + 1, mask=m, other=0).to(tl.float32) - 0.5
                    cr = tl.load(src + sp + 2, mask=m, other=0).to(tl.float32) - 0.5
                    bb = libdevice.fma(cb, 2.032, y)
                    gg = libdevice.fma(cr, -0.581, libdevice.fma(cb, -0.395, y))
                    rr = libdevice.fma(cr, 1.140, y)
                else:
                    cr = tl.load(src + sp + 1, mask=m, other=0).to(tl.float32) - 0.5
                    cb = tl.load(src + sp + 2, mask=m, other=0).to(tl.float32) - 0.5
                    bb = libdevice.fma(cb, 1.773, y)
                    gg = libdevice.fma(cr, -0.714, libdevice.fma(cb, -0.344, y))
                    rr = libdevice.fma(cr, 1.403, y)
            tl.store(dst + dp + BLUE, bb, mask=m)
            tl.store(dst + dp + 1, gg, mask=m)
            tl.store(dst + dp + (BLUE ^ 2), rr, mask=m)
            if DCN == 4:
                tl.store(dst + dp + 3, tl.full((BLOCK,), 1.0, tl.float32), mask=m)
        elif not INV:
            # Forward. Y uses the luma row; the two chroma channels are computed from the
            # ALREADY SHIFTED integer Y, which is what the CPU does and is why this matches.
            b = tl.load(src + sp + BLUE, mask=m, other=0).to(tl.int32)
            g = tl.load(src + sp + 1, mask=m, other=0).to(tl.int32)
            r = tl.load(src + sp + (BLUE ^ 2), mask=m, other=0).to(tl.int32)
            # int32 holds the luma at 16U: 65535*16384 = 1073725440, + 8192.
            y = (b * B2Y + g * G2Y + r * R2Y + (1 << (YUV_SHIFT - 1))) >> YUV_SHIFT
            if OP == OP_YUV:
                c0 = (r - y) * R2VI
                c1 = (b - y) * B2UI
            else:
                c0 = (r - y) * YCRI
                c1 = (b - y) * YCBI
            # Worst case at 16U: (65535 - 0)*14369 + 536870912 = 1478489... < INT32_MAX, so the
            # int32 arithmetic is exact and the clamp below is the only saturation.
            cdelta = (32768 if U16 else 128) << YUV_SHIFT
            cmax = 65535 if U16 else 255
            c0 = tl.minimum(tl.maximum((c0 + cdelta + (1 << (YUV_SHIFT - 1))) >> YUV_SHIFT, 0), cmax)
            c1 = tl.minimum(tl.maximum((c1 + cdelta + (1 << (YUV_SHIFT - 1))) >> YUV_SHIFT, 0), cmax)
            tl.store(dst + dp, y.to(dst.dtype.element_ty), mask=m)
            # c0 is the Cr slot (V for YUV, Cr for YCrCb) and c1 the Cb slot (U, Cb).
            # The CPU writes dst[1 + yuvOrder] = Cr and dst[2 - yuvOrder] = Cb, where
            # yuvOrder is 1 for YUV and 0 for YCrCb -- so YUV is [Y, U, V] and YCrCb is
            # [Y, Cr, Cb]. The slot order must therefore depend on the conversion.
            if OP == OP_YUV:
                tl.store(dst + dp + 1, c1.to(dst.dtype.element_ty), mask=m)
                tl.store(dst + dp + 2, c0.to(dst.dtype.element_ty), mask=m)
            else:
                tl.store(dst + dp + 1, c0.to(dst.dtype.element_ty), mask=m)
                tl.store(dst + dp + 2, c1.to(dst.dtype.element_ty), mask=m)
        else:
            # Inverse: only the chroma term is rounded and shifted; Y is added afterwards,
            # outside the rounding. Cr/Cb sit in different slots for YUV and YCrCb.
            # The bias subtracted here is the same ColorChannel<T>::half() the forward path adds
            # (color_yuv.simd.hpp), and the final saturate_cast<ushort> is the
            # clamp to [0, 65535].
            y = tl.load(src + sp + 0, mask=m, other=0).to(tl.int32)
            half = 32768 if U16 else 128
            if OP == OP_YUV:
                u = tl.load(src + sp + 1, mask=m, other=0).to(tl.int32) - half
                v = tl.load(src + sp + 2, mask=m, other=0).to(tl.int32) - half
                bb = u * U2BI
                gg = u * U2GI + v * V2GI
                rr = v * V2RI
            else:
                cr = tl.load(src + sp + 1, mask=m, other=0).to(tl.int32) - half
                cb = tl.load(src + sp + 2, mask=m, other=0).to(tl.int32) - half
                bb = cb * CB2BI
                gg = cb * CB2GI + cr * CR2GI
                rr = cr * CR2RI
            # int32 holds every term at 16U: |cred - 32768| <= 32767, and the largest coefficient
            # is 33292, so |bb| <= 1.09e9; gg sums two such products (|gg| <= 5.7e8).
            h = 1 << (YUV_SHIFT - 1)
            cmax = 65535 if U16 else 255
            b = y + ((bb + h) >> YUV_SHIFT)
            g = y + ((gg + h) >> YUV_SHIFT)
            r = y + ((rr + h) >> YUV_SHIFT)
            b = tl.minimum(tl.maximum(b, 0), cmax)
            g = tl.minimum(tl.maximum(g, 0), cmax)
            r = tl.minimum(tl.maximum(r, 0), cmax)
            tl.store(dst + dp + BLUE, b.to(dst.dtype.element_ty), mask=m)
            tl.store(dst + dp + 1, g.to(dst.dtype.element_ty), mask=m)
            tl.store(dst + dp + (BLUE ^ 2), r.to(dst.dtype.element_ty), mask=m)
            if DCN == 4:
                tl.store(dst + dp + 3,
                         tl.full((BLOCK,), 65535 if U16 else 255, tl.int32).to(dst.dtype.element_ty),
                         mask=m)
    elif OP == OP_PACKED:
        # A packed 5x5 pixel is CV_8UC2: one 16-bit word per pixel, little-endian inside the two
        # bytes. So both directions work on BYTES: decode reads two and assembles the word, pack
        # writes the low byte then the high byte. Reading them as a u16 would need a 2-byte
        # load, which Triton cannot express through the byte-addressed pointer this kernel
        # shares with every other op -- and the two byte stores are what the CPU's
        # `((ushort*)dst)[0] = d` does on every supported target anyway.
        #
        # GB selects the format and is the CPU's own discriminator (color_rgb.simd.hpp branches
        # on `gb == 6`). The 555 green is NOT bit-replicated, so its full scale is 248, not 255.
        if INV:
            # color_rgb.simd.hpp. b = (v << 3) & 0xF8 is written as a masked shift
            # because Triton has no 16-bit intermediate here and (v << 3) on the int32 word
            # keeps bits above 7, which the mask removes.
            lo = tl.load(src + sp, mask=m, other=0).to(tl.int32)
            hi = tl.load(src + sp + 1, mask=m, other=0).to(tl.int32)
            v = lo | (hi << 8)
            b = (v << 3) & 0xF8
            g = ((v >> 3) & 0xFC) if GB == 6 else ((v >> 2) & 0xF8)
            r = ((v >> 8) & 0xF8) if GB == 6 else ((v >> 7) & 0xF8)
            if DCN == 1:
                # Same BT.601 triple as BGR2GRAY, applied to the EXPANDED fields; the CPU runs
                # RGB5x52Gray (color_rgb.simd.hpp), using the same coefficients.
                y = (b * BY15 + g * GY15 + r * RY15 + (1 << (GRAY_SHIFT - 1))) >> GRAY_SHIFT
                tl.store(dst + dp, y.to(tl.uint8), mask=m)
            else:
                tl.store(dst + dp + BLUE, b.to(tl.uint8), mask=m)
                tl.store(dst + dp + 1, g.to(tl.uint8), mask=m)
                tl.store(dst + dp + (BLUE ^ 2), r.to(tl.uint8), mask=m)
                if DCN == 4:
                    # 565 has no alpha bit, so alpha is 255. 555 carries a 1-bit alpha that the
                    # CPU scales to 0 or 255 -- it is a flag, not a 5-bit field.
                    if GB == 6:
                        a = tl.full((BLOCK,), 255, tl.int32)
                    else:
                        a = ((v >> 15) & 1) * 255
                    tl.store(dst + dp + 3, a.to(tl.uint8), mask=m)
        else:
            # color_rgb.simd.hpp. GRAY is its own arm because Gray2RGB5x5 is a different
            # function in the CPU (bits replicated into all three fields), not a colour packed
            # with equal channels.
            if SCN == 1:
                t = tl.load(src + sp, mask=m, other=0).to(tl.int32)
                t3 = t >> 3
                if GB == 6:
                    d = t3 | ((t & 0xFC) << 3) | (t3 << 11)
                else:
                    d = t3 | (t3 << 5) | (t3 << 10)
            else:
                b = tl.load(src + sp + BLUE, mask=m, other=0).to(tl.int32)
                g = tl.load(src + sp + 1, mask=m, other=0).to(tl.int32)
                r = tl.load(src + sp + (BLUE ^ 2), mask=m, other=0).to(tl.int32)
                # `& ~3` / `& ~7` are the CPU's `g & ~3`, `r & ~7`: the low bits of the field
                # would otherwise collide with the neighbouring channel's bits. The value is
                # TRUNCATED (>>), never rounded -- 255 packs to 31/63, and the round trip
                # 565 -> BGR -> 565 is the identity over all 65536 words.
                if GB == 6:
                    d = (b >> 3) | ((g & 0xFC) << 3) | ((r & 0xF8) << 8)
                else:
                    # bit 15 is the 1-bit alpha: set iff the source alpha is non-zero. A
                    # 3-channel source has no alpha and therefore never sets it.
                    a = tl.load(src + sp + 3, mask=m, other=0).to(tl.int32) if SCN == 4 else 0
                    d = (b >> 3) | ((g & 0xF8) << 2) | ((r & 0xF8) << 7)
                    d = d | ((a != 0).to(tl.int32) << 15)
            tl.store(dst + dp, (d & 0xFF).to(tl.uint8), mask=m)
            tl.store(dst + dp + 1, ((d >> 8) & 0xFF).to(tl.uint8), mask=m)
    elif OP == OP_YUV422:
        # 4:2:2 interleaved (UYVY / YUY2 / YVYU). A 4-byte macropixel covers TWO output pixels and
        # carries one shared chroma pair. Byte layout inside the macropixel
        # (color_yuv.simd.hpp, uidx = 1 - yIdx + uIdx*2, vidx = (2+uidx)%4), ORD picks:
        #   UYVY: U Y0 V Y1     YUY2: Y0 U Y1 V     YVYU: Y0 V Y1 U
        # The source is CV_8UC2 and src_rs is in BYTES for an 8U element, so the byte offset of
        # output column c is row*src_rs + 4*(c//2), and c is Y0 when even, Y1 when odd.
        # Size does not change (the CPU's dstSz = sz) and only width % 2 == 0 is required.
        mac = col >> 1
        base = row * src_rs + mac * 4
        m422 = col < cols
        i0 = tl.load(src + base + 0, mask=m422, other=0).to(tl.int32)
        i1 = tl.load(src + base + 1, mask=m422, other=0).to(tl.int32)
        i2 = tl.load(src + base + 2, mask=m422, other=0).to(tl.int32)
        i3 = tl.load(src + base + 3, mask=m422, other=0).to(tl.int32)
        if ORD == 0:                      # UYVY
            U, Y0, V, Y1 = i0, i1, i2, i3
        elif ORD == 1:                    # YUY2
            Y0, U, Y1, V = i0, i1, i2, i3
        else:                             # YVYU
            Y0, V, Y1, U = i0, i1, i2, i3
        uu = U - 128
        vv = V - 128
        ruv = BT_HALF + BT_CVR * vv
        guv = BT_HALF + BT_CVG * vv + BT_CUG * uu
        buv = BT_HALF + BT_CUB * uu
        # `max(0, Y-16)` clamps rather than wraps: Y below 16 is treated as 16.
        # Y0/Y1 are runtime loads (int32); force both to int32 before the select so the ternary
        # has one type -- Triton rejects `tl.where(cond, int32, constexpr-int)`.
        yysel = tl.where((col & 1) != 0, Y1.to(tl.int32), Y0.to(tl.int32))
        yy = tl.maximum(yysel - 16, 0) * BT_CY
        if DCN == 1:
            tl.store(dst + row * dst_rs + col,
                     tl.minimum(tl.maximum(yy >> BT_SHIFT, 0), 255).to(tl.uint8), mask=m422)
        else:
            # Channel order follows BLUE, exactly as the other op branches do: the CPU routes
            # every YUV2* code through cvtColorYUV2BGR with swapBlue(code), so the *_RGB_* codes
            # write R,G,B and the *_BGR_* ones write B,G,R.
            bl = tl.minimum(tl.maximum((yy + buv) >> BT_SHIFT, 0), 255).to(tl.uint8)
            gr = tl.minimum(tl.maximum((yy + guv) >> BT_SHIFT, 0), 255).to(tl.uint8)
            rd = tl.minimum(tl.maximum((yy + ruv) >> BT_SHIFT, 0), 255).to(tl.uint8)
            if BLUE == 0:
                c0, c2 = bl, rd
            else:
                c0, c2 = rd, bl
            tl.store(dst + row * dst_rs + col * DCN + 0, c0, mask=m422)
            tl.store(dst + row * dst_rs + col * DCN + 1, gr, mask=m422)
            tl.store(dst + row * dst_rs + col * DCN + 2, c2, mask=m422)
            if DCN == 4:
                tl.store(dst + row * dst_rs + col * DCN + 3,
                         tl.full((BLOCK,), 255, tl.int32).to(tl.uint8), mask=m422)

    elif OP == OP_YUV420:
        # NV12 / NV21 semi-planar 4:2:0 -> colour. The source is ONE CV_8UC1 of (3H/2) x W:
        # rows [0,H) are Y, rows [H, 3H/2) are the interleaved chroma plane, one (U,V) pair per
        # 2x2 block (color.simd_helpers.hpp: `sz.width % 2 == 0 && sz.height % 3 == 0`,
        # `dstSz = Size(sz.width, sz.height * 2 / 3)`). `rows` here is the DESTINATION height H,
        # so the chroma plane starts at element uv_rs (== H * src element pitch) and the pair for
        # output pixel (r,c) is at row (r>>1), byte column (c & ~1 as bytes within the row).
        # ORD decides which of the pair is U: NV12 has U first (0), NV21 V first (1).
        # Arithmetic is the same 20-bit BT.601 path as 4:2:2 (color_yuv.simd.hpp).
        mac = (col >> 1) * 2
        base = uv_rs + (row >> 1) * src_rs + mac
        m420 = col < cols
        c0 = tl.load(src + base + 0, mask=m420, other=0).to(tl.int32)
        c1 = tl.load(src + base + 1, mask=m420, other=0).to(tl.int32)
        if ORD == 0:                      # NV12: U then V
            U, V = c0, c1
        else:                             # NV21: V then U
            V, U = c0, c1
        Y = tl.load(src + row * src_rs + col, mask=m420, other=0).to(tl.int32)
        uu = U - 128
        vv = V - 128
        ruv = BT_HALF + BT_CVR * vv
        guv = BT_HALF + BT_CVG * vv + BT_CUG * uu
        buv = BT_HALF + BT_CUB * uu
        yy = tl.maximum(Y - 16, 0) * BT_CY
        bl = tl.minimum(tl.maximum((yy + buv) >> BT_SHIFT, 0), 255).to(tl.uint8)
        gr = tl.minimum(tl.maximum((yy + guv) >> BT_SHIFT, 0), 255).to(tl.uint8)
        rd = tl.minimum(tl.maximum((yy + ruv) >> BT_SHIFT, 0), 255).to(tl.uint8)
        if BLUE == 0:
            a0, a2 = bl, rd
        else:
            a0, a2 = rd, bl
        tl.store(dst + row * dst_rs + col * DCN + 0, a0, mask=m420)
        if DCN > 1:
            tl.store(dst + row * dst_rs + col * DCN + 1, gr, mask=m420)
        if DCN > 2:
            tl.store(dst + row * dst_rs + col * DCN + 2, a2, mask=m420)
        if DCN == 4:
            tl.store(dst + row * dst_rs + col * DCN + 3,
                     tl.full((BLOCK,), 255, tl.int32).to(tl.uint8), mask=m420)

    elif OP == OP_BAYER:
        # Bilinear Bayer demosaic, transcribed from Bayer2RGB_Invoker::operator()
        # (modules/imgproc/src/demosaicing.cpp) and Bayer2Gray_Invoker.
        #
        # The CPU invoker starts at
        #     dst + (range.start + 1) * dst_step + dcn + 1
        # ELEMENTS -- one element past the first interior pixel -- and every write is expressed
        # relative to that: dst[-1], dst[0], dst[1] for the pixel at column 1, then
        # dst[2..4] for column 2, and so on; at the end of the row dst[0] has walked to the
        # column one PAST the last interior pixel, which is why dst[-4] lands on the
        # row's first pixel.
        #
        # Interior pixel (r,c) is centred on source (r,c) and its neighbours are the eight
        # surrounding source samples. The two edge columns copy the adjacent interior column,
        # and the two edge rows copy the adjacent interior row.
        #
        # The row phase comes from the SOURCE row index r-1: an odd one flips both
        # Blue and Start_with_green. SWG and BLUE are constexprs because the CPU derives them
        # from the code and never revisits them.
        #
        # Depth: the CPU runs the same algorithm for uchar and ushort; the 16U path
        # is the SIMD stub, i.e. the plain scalar loop. U16 only changes the element type the
        # pointer is dereferenced as, so the arithmetic is identical in int32 and the store
        # narrows once, exactly like the CPU's `dst[..] = (T)value`.
        # src_rs/dst_rs arrive in ELEMENTS (the launch divides the byte step by the element
        # size, cuda_runtime.cpp). That is the addressing unit of the typed pointer, so the same
        # arithmetic below serves both depths -- the only depth-dependent thing is the store's
        # narrowing and the alpha constant.
        srs = src_rs
        drs = dst_rs
        # Explicit runtime height keeps the module ABI identical on CUDA and HIP.
        # On HIP tl.num_programs introduces hidden block-count parameters, which
        # the module intentionally refuses rather than guessing their values.
        H = rows
        W = cols

        # Interior pixels exclude the outermost rows and columns, so r and c are clamped into
        # [1, H-2] x [1, W-2] and the edge pixels are handled by RE-COMPUTING the interior
        # pixel they copy. That is what turns the CPU's pointer walk into a pure per-pixel map
        # and keeps every load inside the image. H <= 2 / W <= 2 have no interior at all: the
        # clamps degenerate and the result is zeroed below.
        rr = tl.minimum(tl.maximum(row, 1), H - 2)
        cc = tl.minimum(tl.maximum(col, 1), W - 2)

        # Interior pixel (r,c) is centred on SOURCE (r,c): the pair loop reads bayer[0] from the
        # row above and bayer[bayer_step+1] as the centre, so the taps sit at r-1, r, r+1
        # (demosaicing.cpp). The row PHASE, on the other hand, is taken from r-1 -- the
        # invoker's range.start -- which is why an odd-row flip is written below as (rr-1).
        base = rr * srs + cc
        # With no interior the clamp degenerates to a negative index (W=1 gives cc=-1, H=1
        # gives rr=-1), so the loads are masked off as well as the stores. The result is
        # overwritten by the zero below anyway; masking keeps the address inside the buffer.
        interior = (W > 2) & (H > 2)
        lm = m & interior
        up = tl.load(src + base - srs, mask=lm, other=0).to(tl.int32)
        ul = tl.load(src + base - srs - 1, mask=lm, other=0).to(tl.int32)
        ur = tl.load(src + base - srs + 1, mask=lm, other=0).to(tl.int32)
        ml = tl.load(src + base - 1, mask=lm, other=0).to(tl.int32)
        mm = tl.load(src + base, mask=lm, other=0).to(tl.int32)
        mr = tl.load(src + base + 1, mask=lm, other=0).to(tl.int32)
        dn = tl.load(src + base + srs, mask=lm, other=0).to(tl.int32)
        dl = tl.load(src + base + srs - 1, mask=lm, other=0).to(tl.int32)
        dr = tl.load(src + base + srs + 1, mask=lm, other=0).to(tl.int32)

        # Row phase, from the SOURCE row index r-1. An odd one negates Blue and flips
        # Start_with_green. This is a per-row
        # runtime select, not a constexpr, because it depends on the row.
        odd = ((rr - 1) & 1) != 0
        blue = tl.where(odd, -BLUE, BLUE)
        # green iff the column parity matches the row's phase: column 1 is green exactly when
        # start_with_green is set, and the phase toggles on odd rows.
        green = (((cc - (SWG ^ ((rr - 1) & 1))) & 1) == 0)

        # Vertical / horizontal neighbour pairs (green site) and the 2x2 diagonal / orthogonal
        # sums (red/blue site), each with its rounding term.
        tv = (up + dn + 1) >> 1                       # (S[r-1][c] + S[r+1][c] + 1) / 2
        th = (ml + mr + 1) >> 1                       # (S[r][c-1] + S[r][c+1] + 1) / 2
        td = ((ul + ur) + (dl + dr) + 2) >> 2         # 4 diagonals
        to = ((up + ml) + (mr + dn) + 2) >> 2         # 4 orthogonals

        dp = row * drs + col * DCN
        if DCN == 1:
            # Bayer2GRAY uses its own coefficients at both depths.
            # CV_DESCALE(x, n) is `(x + (1 << (n-1))) >> n`.
            # The two sites use different shifts: SHIFT+1 (2-neighbour) and SHIFT+2
            # (4-neighbour). R2Y/B2Y start as (B2Y,R2Y) for BG/GB and swapped for RG/GR
            # then swap again on odd rows.
            bc0: tl.constexpr = BAYER_B2Y if BLUE < 0 else BAYER_R2Y
            rc0: tl.constexpr = BAYER_R2Y if BLUE < 0 else BAYER_B2Y
            bc = tl.where(odd, rc0, bc0)
            rc = tl.where(odd, bc0, rc0)
            # Signedness of the accumulator. The CPU's t0/t1/t2 are `unsigned`, and
            # at 16U the 4-neighbour total can exceed both 2^31 and -- for an all-65535 image --
            # 2^32: it reaches ~4.29e9. Triton computes in int32, so the sum is UB-free but
            # wraps at 2^31 rather than 2^32. That does NOT change the result: only the low 16
            # bits survive the caller's narrowing to ushort, and both `>> 16` on a wrapped int32
            # and on a wrapped uint32 land on the same low 16 bits once narrowed.
            y4 = ((ul + ur + dl + dr) * rc + (up + ml + mr + dn) * BAYER_G2Y +
                  mm * (4 * bc) + (1 << (BAYER_SHIFT + 1))) >> (BAYER_SHIFT + 2)
            y2 = ((up + dn) * rc + (ml + mr) * bc + mm * (2 * BAYER_G2Y) +
                  (1 << BAYER_SHIFT)) >> (BAYER_SHIFT + 1)
            y = tl.where(green, y2, y4)
            # No interior at all: every pixel of the row is zeroed.
            y = tl.where((W <= 2) | (H <= 2), 0, y)
            tl.store(dst + dp, y.to(dst.dtype.element_ty), mask=m)
        else:
            # t0 = diagonals, t1 = orthogonals, and the centre sample is the site's own colour.
            # The CPU writes dst[-blue], dst[0], dst[blue] with blue = +-1, so with
            # 0-based BGR slots the centre lands at 1 and the other two at 0 and 2, mirrored by
            # the sign of blue. blue > 0 is the RED site: dst[0] = diagonal average, dst[2] =
            # the centre. blue < 0 is the BLUE site and mirrors it.
            # Green site: the CPU writes `dst[-blue] = t0 (vertical), dst[0] = centre,
            # dst[blue] = t1 (horizontal)` (including the pair loop's u0/u1 form),
            # so blue > 0 puts the VERTICAL pair on channel 0 and the horizontal on channel 2.
            c0 = tl.where(green, tl.where(blue > 0, tv, th), tl.where(blue > 0, td, mm))
            c2 = tl.where(green, tl.where(blue > 0, th, tv), tl.where(blue > 0, mm, td))
            c1 = tl.where(green, mm, to)
            zero = (W <= 2) | (H <= 2)
            c0 = tl.where(zero, 0, c0)
            c1 = tl.where(zero, 0, c1)
            c2 = tl.where(zero, 0, c2)
            # The edge columns copy the adjacent interior pixel and the edge rows
            # copy the adjacent interior row; the clamps on rr/cc already forced
            # those pixels onto their neighbour, so the values above ARE the copies. Only the
            # no-interior case zeroes them. DCN == 4 adds alpha = numeric_limits<T>::max()
            # (Alpha<T>) -- 255 for uchar, 65535 for ushort. Note the alpha is NOT
            # zeroed by the W<=2 branch: that branch's four-channel form writes
            # `dst[-5..-3] = 0` and then `dst[-2] = alpha`, so a narrow image keeps
            # its alpha and only loses its colour. It IS zeroed when H <= 2, where the whole
            # edge row is memset over all dcn slots.
            alpha = tl.full((BLOCK,), 65535 if U16 else 255, tl.int32)
            alpha = tl.where(H <= 2, 0, alpha)
            tl.store(dst + dp + 0, c0.to(dst.dtype.element_ty), mask=m)
            tl.store(dst + dp + 1, c1.to(dst.dtype.element_ty), mask=m)
            tl.store(dst + dp + 2, c2.to(dst.dtype.element_ty), mask=m)
            if DCN == 4:
                tl.store(dst + dp + 3, alpha.to(dst.dtype.element_ty), mask=m)
    elif OP == OP_XYZ:
        # color_lab.cpp (RGB2XYZ_i<uchar>) and (XYZ2RGB_i<uchar>).
        #
        # The 8-bit path never touches the float matrices: it uses sRGB2XYZ_D65_i /
        # XYZ2sRGB_D65_i, already scaled by 2^shift, and CV_DESCALE. CV_DESCALE rounds
        # half-up. A floating-point implementation using rint()/cvRound would round
        # exact ties differently.
        #
        # THE COEFFICIENT SLOTS ARE POSITIONAL, NOT LOGICAL. cvtcolor.cpp emits them in the
        # canonical order (slot 0 = the channel at src[0], slot 1 = src[1], slot 2 = src[2]),
        # having already resolved the CPU's blueIdx permutation into that order. So this reads
        # src[0..2] and nothing else: adding a BLUE swap here would apply the permutation a
        # second time.
        if not INV:
            c0 = tl.load(src + sp + 0, mask=m, other=0).to(tl.int32)
            c1 = tl.load(src + sp + 1, mask=m, other=0).to(tl.int32)
            c2 = tl.load(src + sp + 2, mask=m, other=0).to(tl.int32)
            X = (c0 * E0 + c1 * E1 + c2 * E2 + (1 << (XYZ_SHIFT - 1))) >> XYZ_SHIFT
            Y = (c0 * E3 + c1 * E4 + c2 * E5 + (1 << (XYZ_SHIFT - 1))) >> XYZ_SHIFT
            Z = (c0 * E6 + c1 * E7 + c2 * E8 + (1 << (XYZ_SHIFT - 1))) >> XYZ_SHIFT
            tl.store(dst + dp, tl.minimum(tl.maximum(X, 0), 255).to(tl.uint8), mask=m)
            tl.store(dst + dp + 1, tl.minimum(tl.maximum(Y, 0), 255).to(tl.uint8), mask=m)
            tl.store(dst + dp + 2, tl.minimum(tl.maximum(Z, 0), 255).to(tl.uint8), mask=m)
        else:
            x = tl.load(src + sp + 0, mask=m, other=0).to(tl.int32)
            y = tl.load(src + sp + 1, mask=m, other=0).to(tl.int32)
            z = tl.load(src + sp + 2, mask=m, other=0).to(tl.int32)
            # Negative coefficients: CV_DESCALE's add-then-arithmetic-shift floors a negative
            # numerator. Triton's arithmetic right shift on int32 preserves that rounding.
            B = (x * E0 + y * E1 + z * E2 + (1 << (XYZ_SHIFT - 1))) >> XYZ_SHIFT
            G = (x * E3 + y * E4 + z * E5 + (1 << (XYZ_SHIFT - 1))) >> XYZ_SHIFT
            R = (x * E6 + y * E7 + z * E8 + (1 << (XYZ_SHIFT - 1))) >> XYZ_SHIFT
            tl.store(dst + dp, tl.minimum(tl.maximum(B, 0), 255).to(tl.uint8), mask=m)
            tl.store(dst + dp + 1, tl.minimum(tl.maximum(G, 0), 255).to(tl.uint8), mask=m)
            tl.store(dst + dp + 2, tl.minimum(tl.maximum(R, 0), 255).to(tl.uint8), mask=m)
    elif OP == OP_LAB:
        # color_lab.cpp (RGB2Lab_b) and (Lab2RGBinteger).
        #
        # Table layout in `t16` (all int32 after the host widens them -- LabCbrtTab_b reaches
        # 37555 in the CPU's own ushort array, so a signed 16-bit load would be wrong):
        #   [0, 256)        sRGBGammaTab_b   (sRGB codes only) or linearGammaTab_b
        #   [256, 3328)     LabCbrtTab_b, 3072 entries
        #   [3328, 3840)    LabToYF_b, 2 per L
        #   [3840, 15104)   abToXZ_b, indexed by (ifxz - minABvalue)
        #   [15104, 19200)  sRGBInvGammaTab_b / the linear inverse is a formula, not a table
        # The offsets are constexprs computed in C++ (cvtcolor.cpp) and are part of the key.
        if not INV:
            # Forward. Only the transfer function differs between the sRGB and "L" codes:
            # linearGammaTab_b[i] == i << gamma_shift, which is the CPU's single difference
            # for LBGR2Lab/LRGB2Lab; everything downstream is the same code.
            #
            # Positional slots again (see OP_XYZ). The transfer function is applied to the RAW
            # channel byte, so the selection is on src[0..2], not on a logical R/G/B.
            q0 = tl.load(src + sp + 0, mask=m, other=0).to(tl.int32)
            q1 = tl.load(src + sp + 1, mask=m, other=0).to(tl.int32)
            q2 = tl.load(src + sp + 2, mask=m, other=0).to(tl.int32)
            if SRGB:
                R = tl.load(t32 + T_GAMMA + q0, mask=m, other=0)
                G = tl.load(t32 + T_GAMMA + q1, mask=m, other=0)
                B = tl.load(t32 + T_GAMMA + q2, mask=m, other=0)
            else:
                R = q0 << 3
                G = q1 << 3
                B = q2 << 3
            h = 1 << (LAB_SHIFT - 1)
            iX = (R * E0 + G * E1 + B * E2 + h) >> LAB_SHIFT
            iY = (R * E3 + G * E4 + B * E5 + h) >> LAB_SHIFT
            iZ = (R * E6 + G * E7 + B * E8 + h) >> LAB_SHIFT
            fX = tl.load(t32 + T_CBRT + iX, mask=m, other=0)
            fY = tl.load(t32 + T_CBRT + iY, mask=m, other=0)
            fZ = tl.load(t32 + T_CBRT + iZ, mask=m, other=0)
            h2 = 1 << (LAB_SHIFT2 - 1)
            L = (LSCALE * fY + LSHIFT + h2) >> LAB_SHIFT2
            a = (500 * (fX - fY) + AB_DELTA + h2) >> LAB_SHIFT2
            bb = (200 * (fY - fZ) + AB_DELTA + h2) >> LAB_SHIFT2
            tl.store(dst + dp, tl.minimum(tl.maximum(L, 0), 255).to(tl.uint8), mask=m)
            tl.store(dst + dp + 1, tl.minimum(tl.maximum(a, 0), 255).to(tl.uint8), mask=m)
            tl.store(dst + dp + 2, tl.minimum(tl.maximum(bb, 0), 255).to(tl.uint8), mask=m)
        else:
            LL = tl.load(src + sp + 0, mask=m, other=0).to(tl.int32)
            aa = tl.load(src + sp + 1, mask=m, other=0).to(tl.int32)
            bb = tl.load(src + sp + 2, mask=m, other=0).to(tl.int32)
            y = tl.load(t32 + T_YF + LL * 2, mask=m, other=0)
            ify = tl.load(t32 + T_YF + LL * 2 + 1, mask=m, other=0)
            # adiv = ((5*aa*53687 + (1<<7)) >> 13) - 128*BASE/500 ; bdiv is the b analogue with
            # its own shift and its own trailing +1 (color_lab.cpp).
            adiv = ((5 * aa * A_MUL + (1 << 7)) >> 13) - ADIV_BIAS
            bdiv = ((bb * B_MUL + (1 << 4)) >> 9) - BDIV_BIAS
            # The CPU indexes abToXZ_b[v - minABvalue]; minABvalue is -8145, so the offset is
            # added here and the table is uploaded already shifted to start at 0.
            x = tl.load(t32 + T_ABXZ + ify + adiv - MIN_AB, mask=m, other=0)
            z = tl.load(t32 + T_ABXZ + ify - bdiv - MIN_AB, mask=m, other=0)
            # ROWS ARE DESTINATION SLOTS. The CPU's constructor fills coeffs[i+blueIdx*3],
            # coeffs[i+3], coeffs[i+(blueIdx^2)*3] -- for blueIdx 0 that is slots 0, 3, 6, i.e.
            # row-major, and the CPU then writes dst[0]=bo (row 0), dst[1]=go (row 1),
            # dst[2]=ro (row 2), so coefficients are used in row-major order.
            d0 = (E0 * x + E1 * y + E2 * z + (1 << (REV_SHIFT - 1))) >> REV_SHIFT
            d1 = (E3 * x + E4 * y + E5 * z + (1 << (REV_SHIFT - 1))) >> REV_SHIFT
            d2 = (E6 * x + E7 * y + E8 * z + (1 << (REV_SHIFT - 1))) >> REV_SHIFT
            d0 = tl.minimum(tl.maximum(d0, 0), INV_GAMMA_MAX)
            d1 = tl.minimum(tl.maximum(d1, 0), INV_GAMMA_MAX)
            d2 = tl.minimum(tl.maximum(d2, 0), INV_GAMMA_MAX)
            if SRGB:
                d0 = tl.load(t32 + T_INVGAMMA + d0, mask=m, other=0)
                d1 = tl.load(t32 + T_INVGAMMA + d1, mask=m, other=0)
                d2 = tl.load(t32 + T_INVGAMMA + d2, mask=m, other=0)
            else:
                # linearInvGammaTab_b is not a table: the CPU writes
                # `rgb = (rgb*255) >> inv_gamma_shift`, i.e. ((v << 8) - v) >> 12.
                d0 = ((d0 << 8) - d0) >> 12
                d1 = ((d1 << 8) - d1) >> 12
                d2 = ((d2 << 8) - d2) >> 12
            tl.store(dst + dp, d0.to(tl.uint8), mask=m)
            tl.store(dst + dp + 1, d1.to(tl.uint8), mask=m)
            tl.store(dst + dp + 2, d2.to(tl.uint8), mask=m)
    elif OP == OP_LUV:
        # color_lab.cpp (RGB2Luv_b -> RGB2Luvinterpolate) and Luv2RGBinteger.
        #
        # THE sRGB FORWARD DIRECTION IS A TRILINEAR LUT, NOT THE CIE FORMULA. RGB2Luv_b sets
        # useInterpolation when srgb is set, so the 8-bit path is RGB2Luvinterpolate over
        # RGB2LuvLUT_s16 -- a 33^3 grid. Computing exact CIE Luv would be WRONG here: the LUT
        # approximates that formula and has different rounding from the floating-point path.
        # Reproduce the LUT, including corner duplication. Only the forward direction uses the
        # 33-node grid; Luv2BGR uses integer arithmetic like Lab2BGR.
        #
        # Corner layout: `t16` holds the LUT as int16 (its values are bounded by 16384, unlike
        # LabCbrtTab_b). Each node carries 8 corners x 3 channels = 24 int16, corners at stride
        # 1 in _p,_q,_r order and channels at stride 8 -- the CPU's own
        # `baseLUT[i], baseLUT[i+8], baseLUT[i+16]` (color_lab.cpp).
        # The 8 weights (trilinearLUT) pair weight i with corner
        # (_p = i>>2, _q = (i>>1)&1, _r = i&1) and use the fraction where the bit is 1, so
        # weight i multiplies the node one step UP in each of those axes.
        if not INV:
            # The LUT uses its own int16 buffer at T_LUV_LUT. The sRGB LUT is
            # the only 8-bit forward Luv path: RGB2Luv_b's useInterpolation is set exactly when
            # srgb is true and no custom coefficients were supplied (color_lab.cpp),
            # so there is no linear-Luv forward arm here at all -- those two codes are refused
            # by the public argument validator.
            r = tl.load(src + sp + BLUE, mask=m, other=0).to(tl.int32)
            g = tl.load(src + sp + 1, mask=m, other=0).to(tl.int32)
            b = tl.load(src + sp + (BLUE ^ 2), mask=m, other=0).to(tl.int32)
            # cx = ch * (LAB_BASE/256) = ch * 64; node tx = cx >> 9 = ch >> 3 and the fraction
            # (cx >> 5) & 15 = (2*ch) & 15. Verified against the CPU's own shifts for all 256.
            tr = r >> LUT_SHIFT
            tg = g >> LUT_SHIFT
            tb = b >> LUT_SHIFT
            fr = (r * 2) & 15
            fg = (g * 2) & 15
            fb = (b * 2) & 15
            # The CPU's fill_one duplicates the last node into the corner one past the grid
            # (min(p+_p, LAB_LUT_DIM-1)), so tx+1 is clamped at 32 rather than wrapping.
            nr = tl.minimum(tr + 1, 32)
            ng = tl.minimum(tg + 1, 32)
            nb = tl.minimum(tb + 1, 32)
            # Node offsets. Step along each axis is 3*8 = 24 int16 per node, and the axes are
            # (p,q,r) = (R,G,B) in the CPU's own build order, so the R axis is the slowest.
            #
            # Trilinear corner weights follow the CPU table order. The table
            # builder puts pp = 16-p on w[0] and p on w[4]; trilinearInterpolate
            # then pairs w with the corner at 4*_p+2*_q+_r. So the weight carried
            # by the corner one node UP in an axis is the FRACTION in that axis, and the weight
            # on the low corner is its complement -- the standard trilinear form. Any other
            # reading produces a plausible-looking image that is wrong everywhere.
            px = tr * 24 + tg * LUV_LUT_STRIDE + tb * LUV_LUT_STRIDE2
            wr0 = 16 - fr
            wr1 = fr
            wg0 = 16 - fg
            wg1 = fg
            wb0 = 16 - fb
            wb1 = fb
            # Steps to the corner one node up on each axis, with the last-node duplication
            # (fill_one's min(p+_p, 32)) applied: at node 32 the "up" corner IS node 32, so
            # the step is 0 there and the weight lands on the same value twice -- which is what
            # the CPU does, and why the clamp is applied to the step rather than the node.
            s_r = (nr - tr) * 24
            s_g = (ng - tg) * LUV_LUT_STRIDE
            s_b = (nb - tb) * LUV_LUT_STRIDE2
            # Channel stride inside a node is 8: L at +0, u at +8, v at +16.
            accL = tl.zeros((BLOCK,), tl.int32)
            accU = tl.zeros((BLOCK,), tl.int32)
            accV = tl.zeros((BLOCK,), tl.int32)
            zero = tl.zeros((BLOCK,), tl.int32)
            for i in tl.static_range(8):
                # i is a Python int at trace time (static_range), so these are Python ints 0/1
                # and the ternaries below are resolved while tracing -- no dynamic select
                # reaches the device. The explicit `zero` keeps both arms the same SHAPE: a
                # scalar 0 mixed with a BLOCK-vector in a ternary is a Triton compile error
                # ("Ternary expression with dynamic condition has inconsistent types").
                pr = (i >> 2) & 1
                qg = (i >> 1) & 1
                rb = i & 1
                off = px + (s_r if pr else zero) + (s_g if qg else zero) + (s_b if rb else zero)
                wgt = (wr1 if pr else wr0) * (wg1 if qg else wg0) * (wb1 if rb else wb0)
                accL += tl.load(t16 + off, mask=m, other=0) * wgt
                accU += tl.load(t16 + off + 8, mask=m, other=0) * wgt
                accV += tl.load(t16 + off + 16, mask=m, other=0) * wgt
            # CV_DESCALE(..., trilinear_shift*3 = 12); the weights sum to 16^3 = 4096, so the
            # descaled value is again on the LUT's 2^14 scale. The CPU then TRUNCATES (v/baseDiv
            # with baseDiv = 64, and v >= 0 by construction), which is `>> 6`, not a rounding.
            L = (accL + (1 << (TRI_SHIFT - 1))) >> TRI_SHIFT
            U = (accU + (1 << (TRI_SHIFT - 1))) >> TRI_SHIFT
            V = (accV + (1 << (TRI_SHIFT - 1))) >> TRI_SHIFT
            tl.store(dst + dp, (L >> 6).to(tl.uint8), mask=m)
            tl.store(dst + dp + 1, (U >> 6).to(tl.uint8), mask=m)
            tl.store(dst + dp + 2, (V >> 6).to(tl.uint8), mask=m)
        else:
            LL = tl.load(src + sp + 0, mask=m, other=0).to(tl.int32)
            uu = tl.load(src + sp + 1, mask=m, other=0).to(tl.int32)
            vv = tl.load(src + sp + 2, mask=m, other=0).to(tl.int32)
            y = tl.load(t32 + T_YF + LL * 2, mask=m, other=0)
            up = tl.load(t32 + T_LU + LL * 256 + uu, mask=m, other=0)
            vp = tl.load(t32 + T_LV + LL * 256 + vv, mask=m, other=0)
            vpl = tl.load(t64 + LL * 256 + vv, mask=m, other=0)
            y64 = y.to(tl.int64)
            xv = up.to(tl.int64) * vp.to(tl.int64)
            # C++ integer division and Triton's signed // truncate toward zero.
            # A right shift would floor negative values instead.
            x = (xv // BASE).to(tl.int32)
            x = ((y64 * x.to(tl.int64)) // BASE).to(tl.int32)
            zp = (vpl - xv * 85) // BASE          # 255/3 == 85 in the CPU (integer division)
            zq = zp - 5 * 255 * BASE
            zm = ((y64 * zq) // BASE).to(tl.int32)
            z = zm // 256 + zm // 65536
            x = tl.minimum(tl.maximum(x, 0), 2 * BASE)
            z = tl.minimum(tl.maximum(z, 0), 2 * BASE)
            # Destination-slot rows, same as OP_LAB's reverse.
            d0 = (E0 * x + E1 * y + E2 * z + (1 << (REV_SHIFT - 1))) >> REV_SHIFT
            d1 = (E3 * x + E4 * y + E5 * z + (1 << (REV_SHIFT - 1))) >> REV_SHIFT
            d2 = (E6 * x + E7 * y + E8 * z + (1 << (REV_SHIFT - 1))) >> REV_SHIFT
            d0 = tl.minimum(tl.maximum(d0, 0), INV_GAMMA_MAX)
            d1 = tl.minimum(tl.maximum(d1, 0), INV_GAMMA_MAX)
            d2 = tl.minimum(tl.maximum(d2, 0), INV_GAMMA_MAX)
            if SRGB:
                d0 = tl.load(t32 + T_INVGAMMA + d0, mask=m, other=0)
                d1 = tl.load(t32 + T_INVGAMMA + d1, mask=m, other=0)
                d2 = tl.load(t32 + T_INVGAMMA + d2, mask=m, other=0)
            else:
                d0 = ((d0 << 8) - d0) >> 12
                d1 = ((d1 << 8) - d1) >> 12
                d2 = ((d2 << 8) - d2) >> 12
            tl.store(dst + dp, d0.to(tl.uint8), mask=m)
            tl.store(dst + dp + 1, d1.to(tl.uint8), mask=m)
            tl.store(dst + dp + 2, d2.to(tl.uint8), mask=m)
    else:
        # OP_HSV, forward only: the integer path of RGB2HSV_b.
        b = tl.load(src + sp + BLUE, mask=m, other=0).to(tl.int32)
        g = tl.load(src + sp + 1, mask=m, other=0).to(tl.int32)
        r = tl.load(src + sp + (BLUE ^ 2), mask=m, other=0).to(tl.int32)
        v = tl.maximum(tl.maximum(b, g), r)
        mn = tl.minimum(tl.minimum(b, g), r)
        d = v - mn
        s = (d * tl.load(sdiv + v, mask=m, other=0) + (1 << (HSV_SHIFT - 1))) >> HSV_SHIFT
        # Channel priority r > g > b, as the CPU's vr/vg masks encode.
        hraw = tl.where(v == r, g - b, tl.where(v == g, b - r + 2 * d, r - g + 4 * d))
        h = (hraw * tl.load(hdiv + d, mask=m, other=0) + (1 << (HSV_SHIFT - 1))) >> HSV_SHIFT
        h = tl.where(h < 0, h + HR, h)
        h = tl.minimum(tl.maximum(h, 0), 255)
        tl.store(dst + dp, h.to(tl.uint8), mask=m)
        tl.store(dst + dp + 1, s.to(tl.uint8), mask=m)
        tl.store(dst + dp + 2, v.to(tl.uint8), mask=m)


# HIP-specific aligned colour kernels.
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
#
# HIP GPU fast path: convert four aligned uint8 BGR pixels to grayscale per lane.
# The generator appends this fragment to cvtcolor_cpu_hip.py, inheriting its imports.
# Host predicates check width/alignment; row strides count uint8 elements (bytes).


@triton.jit(do_not_specialize=['src', 'dst', 'cols', 'src_rs', 'dst_rs'])
def _cvtcolor_hip_packed(src, dst, cols, src_rs, dst_rs,
                        BLOCK: tl.constexpr):
    """Four aligned uint8 BGR pixels per lane, with independent row pitches."""
    row = tl.program_id(0).to(tl.int64)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = x < cols // 4
    s = (src + row * src_rs).to(tl.pointer_type(tl.uint32))
    d = (dst + row * dst_rs).to(tl.pointer_type(tl.uint32))
    a = tl.load(s + 3 * x, mask, 0)
    b = tl.load(s + 3 * x + 1, mask, 0)
    c = tl.load(s + 3 * x + 2, mask, 0)
    # COLOR_BGR2GRAY, CPU gray_shift=15.
    y0 = ((a & 255) * 3735 + ((a >> 8) & 255) * 19235 + ((a >> 16) & 255) * 9798 + 16384) >> 15
    y1 = ((a >> 24) * 3735 + (b & 255) * 19235 + ((b >> 8) & 255) * 9798 + 16384) >> 15
    y2 = (((b >> 16) & 255) * 3735 + (b >> 24) * 19235 + (c & 255) * 9798 + 16384) >> 15
    y3 = (((c >> 8) & 255) * 3735 + ((c >> 16) & 255) * 19235 + (c >> 24) * 9798 + 16384) >> 15
    tl.store(d + x, y0 | (y1 << 8) | (y2 << 16) | (y3 << 24), mask)
