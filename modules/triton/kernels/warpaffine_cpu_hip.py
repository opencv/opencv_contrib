# Generated HIP warpaffine kernels; regenerate with tools/make_kernels_cpu_hip.py.
# source_sha256: c20d55a8d91a4170705780573662cb11c0d00a06c8279a7a4153f6d0b769711c
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright 2026, The FlagOS Contributors.
#
# GPU affine sampling with the selected OpenCV CPU coordinate and rounding rules.
# The cpu prefix describes the arithmetic, not where execution takes place.
# M maps destination pixels to source coordinates after host-side inversion if needed.
# Row strides count typed elements; border indices are source pixel indices.

import triton
import triton.language as tl
from triton.language.extra import libdevice


# HIP/AMDGCN replacements for missing libdevice primitives: add_rn, mul_rn, sub_rn, rint.
# Explicit vector arithmetic preserves separate add/multiply/subtract operations.
# v_rndne_f32 rounds to the nearest even integer.

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


@triton.jit
def _triton_rint(a):
    return tl.inline_asm_elementwise("v_rndne_f32 $0, $1", "=v,v", [a], dtype=tl.float32, is_pure=True, pack=1)



@triton.jit
def _pmod(p, n):
    """p mod n, always in [0, n). Triton's `//` and `%` truncate toward zero like
    C, so a negative index would otherwise come back negative; every border rule
    below is periodic and needs the mathematical modulo."""
    return ((p % n) + n) % n


@triton.jit
def _bm(p, n, BORDER: tl.constexpr):
    """borderInterpolate for one index, in closed form (borderInterpolate_fast).

    REPLICATE clips; WRAP is periodic with period n; REFLECT has period 2n and
    REFLECT_101 period 2(n-1) (the border pixel is not repeated). All three are
    full modulo forms rather than loops because the index range here is
    unbounded (an affine map can point anywhere).
    """
    if BORDER == 1:                                   # BORDER_REPLICATE
        return tl.minimum(tl.maximum(p, 0), n - 1)
    if BORDER == 3:                                   # BORDER_WRAP
        return _pmod(p, n)
    if BORDER == 2:                                   # BORDER_REFLECT, period 2n
        m = _pmod(p, 2 * n)
        return tl.where(m < n, m, 2 * n - m - 1)
    period = tl.maximum(2 * (n - 1), 1)               # BORDER_REFLECT_101
    m = _pmod(p, period)
    return tl.where(m < n, m, period - m)


@triton.jit
def _fma(a, b, c, FUSED: tl.constexpr):
    """cv::v_fma: fused on an FMA3 build, multiply-then-add otherwise.

    core/hal/intrin_sse.hpp expands v_fma to _mm_add_ps(_mm_mul_ps(a,b), c) when CV_FMA3 is 0.
    OpenCV's warp kernels use v_fma for the coordinate term AND for the interpolation
    accumulation, so both must follow the same switch; reproducing only the coordinate one
    leaves the interpolation off by an ulp, which reaches the integer store.
    """
    if FUSED:
        return libdevice.fma(a, b, c)
    return _triton_add_rn(_triton_mul_rn(a, b), c)


@triton.jit
def _mapped_coord(xf, M00, M10, row_x, row_y, FUSED: tl.constexpr):
    """The column term of CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2.

    `v_fma(M0, dst_x, M_x)` on the CPU is only a fused operation when the build has FMA3:
    core/hal/intrin_sse.hpp implements v_fma as _mm_add_ps(_mm_mul_ps(a,b), c) when CV_FMA3 is
    0. An FMA3 build therefore rounds once and a baseline-only build rounds twice, and at a
    0.5 tie in the interpolation the two land on different source pixels.

    The C++ planner identifies the linked CPU arithmetic with fixed numerical probes
    and passes it as FUSED. A different rounding at a sampling tie can select an
    adjacent source pixel.
    """
    if FUSED:
        return (libdevice.fma(xf, M00.to(tl.float32), row_x),
                libdevice.fma(xf, M10.to(tl.float32), row_y))
    return (_triton_add_rn(_triton_mul_rn(xf, M00.to(tl.float32)), row_x),
            _triton_add_rn(_triton_mul_rn(xf, M10.to(tl.float32)), row_y))


@triton.jit
def _warp_nearest(src, dst, M00, M01, M02, M10, M11, M12, src_h, src_w, dst_h, dst_w,
                  src_rs, dst_rs, bv0, bv1, bv2, bv3,
                  cn: tl.constexpr, BORDER: tl.constexpr, BORDER_X: tl.constexpr,
                  BORDER_Y: tl.constexpr, FUSED: tl.constexpr, TRANSPARENT: tl.constexpr,
                  BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    ys = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    xs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]
    m = (ys < dst_h) & (xs < dst_w)

    # static_cast<float>(y*M1 + M2): OpenCV's M[] is FLOAT, so the row term is
    # float arithmetic rounded twice (multiply, add). Computing it in double and
    # narrowing once moves the result by an ulp on ties, which a 0.5 tie turns
    # into a whole output level.
    yf = ys.to(tl.float32)
    # mul_rn/add_rn rather than `a*b + c`: the compiler would contract that into
    # one fma, and a single rounding here moves the row term by an ulp, which a
    # 0.5 tie in the interpolation turns into a whole output level.
    row_x = _triton_add_rn(_triton_mul_rn(yf, M01.to(tl.float32)), M02.to(tl.float32))
    row_y = _triton_add_rn(_triton_mul_rn(yf, M11.to(tl.float32)), M12.to(tl.float32))
    xf = xs.to(tl.float32)
    sx, sy = _mapped_coord(xf, M00, M10, row_x, row_y, FUSED)

    ix = _triton_rint(sx).to(tl.int32)             # cvRound: half to even
    iy = _triton_rint(sy).to(tl.int32)

    if BORDER == 0 or TRANSPARENT:                   # BORDER_CONSTANT, or TRANSPARENT's range test
        inside = (ix >= 0) & (ix < src_w) & (iy >= 0) & (iy < src_h)
        cx = tl.minimum(tl.maximum(ix, 0), src_w - 1)
        cy = tl.minimum(tl.maximum(iy, 0), src_h - 1)
    else:
        inside = m
        # border_type_x/y: a 1-pixel axis is replicated whatever the mode, because
        # the reflection periods collapse there (imgwarp.cpp warpAffine)
        cx = _bm(ix, src_w, BORDER_X)
        cy = _bm(iy, src_h, BORDER_Y)

    # Row pitch in ELEMENTS, not src_w*cn: a GpuMat's step is padded for alignment, so
    # recomputing the pitch from the width reads the wrong row on every row but the first.
    base = cy.to(tl.int64) * src_rs + cx.to(tl.int64) * cn
    # BORDER_TRANSPARENT (NEAREST): write only where the rounded source point is inside, using
    # the CPU's own unsigned range test -- `(unsigned)cvRound(sx) < W && (unsigned)cvRound(sy) < H`.
    # The destination is never read, so "skip" and "write the old value" are the same thing for
    # the pixels that are skipped; the store mask is the whole rule.
    store_mask = m & inside if TRANSPARENT else m
    for c in tl.static_range(cn):
        v = tl.load(src + base + c, mask=m & inside, other=0)
        if BORDER == 0:
            # The border value is a Scalar carried per channel, NOT one value repeated:
            # cv2.warpAffine(..., borderValue=(10,20,30)) fills (10,20,30), and a scalar
            # fills (v,0,0). Collapsing it to one number is wrong for every channel but 0.
            if c == 0:
                bv = bv0
            elif c == 1:
                bv = bv1
            elif c == 2:
                bv = bv2
            else:
                bv = bv3
            # bv is an fp32 runtime argument already rounded and clamped on the host into
            # the destination dtype (saturate_cast), so a uint8/uint16 store just narrows it.
            v = tl.where(m & ~inside, tl.full(v.shape, bv.to(v.dtype), v.dtype), v)
        tl.store(dst + ys.to(tl.int64) * dst_rs + xs.to(tl.int64) * cn + c, v, mask=store_mask)


@triton.jit
def _warp_linear(src, dst, M00, M01, M02, M10, M11, M12, src_h, src_w, dst_h, dst_w,
                 src_rs, dst_rs, bv0, bv1, bv2, bv3,
                 cn: tl.constexpr, U8: tl.constexpr, U16: tl.constexpr,
                 BORDER: tl.constexpr, TRANSPARENT: tl.constexpr,
                 BORDER_X: tl.constexpr, BORDER_Y: tl.constexpr, FUSED: tl.constexpr,
                 BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    ys = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    xs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]
    m = (ys < dst_h) & (xs < dst_w)

    # static_cast<float>(y*M1 + M2): OpenCV's M[] is FLOAT, so the row term is
    # float arithmetic rounded twice (multiply, add). Computing it in double and
    # narrowing once moves the result by an ulp on ties, which a 0.5 tie turns
    # into a whole output level.
    yf = ys.to(tl.float32)
    # mul_rn/add_rn rather than `a*b + c`: the compiler would contract that into
    # one fma, and a single rounding here moves the row term by an ulp, which a
    # 0.5 tie in the interpolation turns into a whole output level.
    row_x = _triton_add_rn(_triton_mul_rn(yf, M01.to(tl.float32)), M02.to(tl.float32))
    row_y = _triton_add_rn(_triton_mul_rn(yf, M11.to(tl.float32)), M12.to(tl.float32))
    xf = xs.to(tl.float32)
    sx, sy = _mapped_coord(xf, M00, M10, row_x, row_y, FUSED)

    fx = tl.floor(sx)
    fy = tl.floor(sy)
    ax = sx - fx
    ay = sy - fy
    ix = fx.to(tl.int32)
    iy = fy.to(tl.int32)
    ix1 = ix + 1
    iy1 = iy + 1

    if BORDER == 0 or TRANSPARENT:
        # whole neighbourhood outside -> border value; otherwise each tap that is
        # outside alone takes the border value (CV_WARP_SCALAR_FETCH_PIXEL). TRANSPARENT uses the
        # same in-range flags, but substitutes the OLD DESTINATION PIXEL for an out-of-range tap
        # and leaves the weights unchanged (see the lstore comment below).
        outside = (ix >= src_w) | (ix1 < 0) | (iy >= src_h) | (iy1 < 0)
        in0 = (ix >= 0) & (ix < src_w)
        in1 = (ix1 >= 0) & (ix1 < src_w)
        jy0 = (iy >= 0) & (iy < src_h)
        jy1 = (iy1 >= 0) & (iy1 < src_h)
    else:
        outside = m & False
        in0 = m
        in1 = m
        jy0 = m
        jy1 = m

    hw = src_rs
    for c in tl.static_range(cn):
        if BORDER == 0 or TRANSPARENT:
            if c == 0:
                bv = bv0
            elif c == 1:
                bv = bv1
            elif c == 2:
                bv = bv2
            else:
                bv = bv3
            # `other` must have the pointer's element type; the border component is fp32.
            bvs = bv.to(src.dtype.element_ty)
            if TRANSPARENT:
                # Out-of-range taps take the destination pixel's CURRENT value, not a constant.
                bvs = tl.load(dst + ys.to(tl.int64) * dst_rs + xs.to(tl.int64) * cn + c,
                              mask=m, other=0)
            p00 = tl.load(src + iy.to(tl.int64) * hw + ix.to(tl.int64) * cn + c,
                          mask=m & in0 & jy0, other=bvs)
            p01 = tl.load(src + iy.to(tl.int64) * hw + ix1.to(tl.int64) * cn + c,
                          mask=m & in1 & jy0, other=bvs)
            p10 = tl.load(src + iy1.to(tl.int64) * hw + ix.to(tl.int64) * cn + c,
                          mask=m & in0 & jy1, other=bvs)
            p11 = tl.load(src + iy1.to(tl.int64) * hw + ix1.to(tl.int64) * cn + c,
                          mask=m & in1 & jy1, other=bvs)
        else:
            cx0 = _bm(ix, src_w, BORDER_X)
            cx1 = _bm(ix1, src_w, BORDER_X)
            cy0 = _bm(iy, src_h, BORDER_Y)
            cy1 = _bm(iy1, src_h, BORDER_Y)
            p00 = tl.load(src + cy0.to(tl.int64) * hw + cx0.to(tl.int64) * cn + c, mask=m, other=0)
            p01 = tl.load(src + cy0.to(tl.int64) * hw + cx1.to(tl.int64) * cn + c, mask=m, other=0)
            p10 = tl.load(src + cy1.to(tl.int64) * hw + cx0.to(tl.int64) * cn + c, mask=m, other=0)
            p11 = tl.load(src + cy1.to(tl.int64) * hw + cx1.to(tl.int64) * cn + c, mask=m, other=0)

        if U8 or U16:
            f = tl.float32
            p00 = p00.to(f); p01 = p01.to(f); p10 = p10.to(f); p11 = p11.to(f)
        # v0 = fma(ax, p01 - p00, p00); v1 = fma(ax, p11 - p10, p10); out = fma(ay, v1 - v0, v0)
        v0 = _fma(ax, p01 - p00, p00, FUSED)
        v1 = _fma(ax, p11 - p10, p10, FUSED)
        v = _fma(ay, v1 - v0, v0, FUSED)
        if U8:
            v = tl.minimum(tl.maximum(_triton_rint(v), 0.0), 255.0).to(tl.uint8)
        elif U16:
            # v_pack_u(v_round(...)) = saturate_cast<ushort>(float): round, clamp, convert
            v = tl.minimum(tl.maximum(_triton_rint(v), 0.0), 65535.0).to(tl.uint16)
        if BORDER == 0:
            if c == 0:
                bv = bv0
            elif c == 1:
                bv = bv1
            elif c == 2:
                bv = bv2
            else:
                bv = bv3
            v = tl.where(outside & m, tl.full(v.shape, bv.to(v.dtype), v.dtype), v)
        # BORDER_TRANSPARENT (LINEAR): each out-of-range tap is replaced by the OLD DESTINATION
        # pixel with its weight unchanged -- out = sum(w_k*S_k) + (1 - sum(w_k))*dst_old, so the
        # blend degenerates smoothly to "keep the old value" as the point leaves the image. No
        # renormalisation and no border value; the pixel is still written unless the whole 2x2
        # window is outside (ix < -1 || ix >= W || iy < -1 || iy >= H).
        # The destination must already hold its intended contents.
        lstore = m & ~outside if TRANSPARENT else m
        tl.store(dst + ys.to(tl.int64) * dst_rs + xs.to(tl.int64) * cn + c, v, mask=lstore)


@triton.jit
def _bicubic_weights_t(a, FUSED: tl.constexpr):
    """bicubicWeights for one fractional coordinate, float32, in the C++ order.

    w0 = A*(a*b2) and w3 = A*(a2*b): the grouping is A*(a*b^2), not (A*a)*b^2 -- the
    two differ in the last bit and reach the integer store. w1 = fma(a2, fma(A+2, a,
    -(A+3)), 1) is the SIMD form, w2 = 1 - w0 - w1 - w3 in that order.
    """
    a2 = _triton_mul_rn(a, a)
    b = _triton_sub_rn(1.0, a)
    b2 = _triton_mul_rn(b, b)
    w0 = _triton_mul_rn(-0.75, _triton_mul_rn(a, b2))
    w3 = _triton_mul_rn(-0.75, _triton_mul_rn(a2, b))
    w1 = _fma(a2, _fma(1.25, a, -2.25, FUSED), 1.0, FUSED)   # A+2 = 1.25, A+3 = 2.25
    w2 = _triton_sub_rn(_triton_sub_rn(_triton_sub_rn(1.0, w0), w1), w3)
    return w0, w1, w2, w3


@triton.jit
def _warp_cubic(src, dst, M00, M01, M02, M10, M11, M12, src_h, src_w, dst_h, dst_w,
                src_rs, dst_rs, bv0, bv1, bv2, bv3,
                cn: tl.constexpr, U8: tl.constexpr, U16: tl.constexpr,
                BORDER: tl.constexpr, TRANSPARENT: tl.constexpr,
                BORDER_X: tl.constexpr, BORDER_Y: tl.constexpr,
                FUSED: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    """INTER_CUBIC: genericWarp -> bicubicVec/bicubicRef (OpenCV 5.0.0).

    Different family from _warp_linear/_warp_nearest: the column coordinate is
    M_x + M_xx*x with NO fma, computed from the absolute x, so every block computes a
    column the same way and the result does not depend on the destination width.
    The cubic reference therefore does not need destination-width padding.

    Tap layout: ix0 = floor(sx) - 1, then ix0..ix0+3; sx and sy are clamped in float to
    [-max(dim,16), 2*max(dim,16)] first, as bicubicCoeffs does, so a wild matrix cannot
    index unpredictably.
    """
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    ys = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    xs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]
    m = (ys < dst_h) & (xs < dst_w)

    # M_x = float(y)*M(0,1) + M(0,2): a float multiply then a float add, both rounded.
    yf = ys.to(tl.float32)
    row_x = _triton_add_rn(_triton_mul_rn(yf, M01.to(tl.float32)), M02.to(tl.float32))
    row_y = _triton_add_rn(_triton_mul_rn(yf, M11.to(tl.float32)), M12.to(tl.float32))
    xf = xs.to(tl.float32)
    # xbuf[dx] = M_x + M_xx*xf -- NOT fma, and this is what makes CUBIC width-invariant.
    sx = _triton_add_rn(row_x, _triton_mul_rn(xf, M00.to(tl.float32)))
    sy = _triton_add_rn(row_y, _triton_mul_rn(xf, M10.to(tl.float32)))

    # bicubicCoeffs' float clamp before the floor: bigwidth = max(size, 16)
    bigw = tl.maximum(src_w, 16)
    bigh = tl.maximum(src_h, 16)
    sx = tl.minimum(tl.maximum(sx, -bigw.to(tl.float32)), (2 * bigw).to(tl.float32))
    sy = tl.minimum(tl.maximum(sy, -bigh.to(tl.float32)), (2 * bigh).to(tl.float32))

    fx = tl.floor(sx)
    fy = tl.floor(sy)
    ax = _triton_sub_rn(sx, fx)
    ay = _triton_sub_rn(sy, fy)
    ix0 = fx.to(tl.int32) - 1
    iy0 = fy.to(tl.int32) - 1
    wx0, wx1, wx2, wx3 = _bicubic_weights_t(ax, FUSED)
    wy0, wy1, wy2, wy3 = _bicubic_weights_t(ay, FUSED)

    # all_outliers: the whole 4-tap window is outside, i.e. floor(sx)-1+3 < 0 or >= width, which
    # with ix0 = floor(sx)-1 is ix0+3 < 0 or ix0 >= width. The upper half is `ix0 + 4 >= width + 4`
    # (same thing); the lower half must be `ix0 + 3 < 0`, NOT `ix0 + 4 < 0` -- the latter is one
    # pixel too strict and leaves a column/row of pixels written that the CPU leaves untouched
    # (warp_kernels.simd.hpp bicubicCoeffs: `(unsigned)(ix0+4) >= (unsigned)(width+4)` compared
    # against the window origin, and the TRANSPARENT skip is `code == -1`).
    all_out = ((ix0 + 4 >= src_w + 4) | (ix0 + 3 < 0) |
               (iy0 + 4 >= src_h + 4) | (iy0 + 3 < 0))

    wx = (wx0, wx1, wx2, wx3)
    wy = (wy0, wy1, wy2, wy3)

    for c in tl.static_range(cn):
        acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        for r in tl.static_range(4):
            wy_r = wy[r]
            ry = iy0 + r
            rowsum = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            for k in tl.static_range(4):
                xx = ix0 + k
                if BORDER == 0 or TRANSPARENT:
                    ok = (xx >= 0) & (xx < src_w) & (ry >= 0) & (ry < src_h)
                    cx = tl.minimum(tl.maximum(xx, 0), src_w - 1)
                    cy = tl.minimum(tl.maximum(ry, 0), src_h - 1)
                else:
                    cx = _bm(xx, src_w, BORDER_X)
                    cy = _bm(ry, src_h, BORDER_Y)
                    ok = m
                idx = (cy.to(tl.int64) * src_rs + cx.to(tl.int64) * cn + c)
                if BORDER == 0 or TRANSPARENT:
                    if c == 0:
                        bv = bv0
                    elif c == 1:
                        bv = bv1
                    elif c == 2:
                        bv = bv2
                    else:
                        bv = bv3
                    bvs = bv.to(src.dtype.element_ty)
                    if TRANSPARENT:
                        # BORDER_TRANSPARENT (CUBIC): an out-of-range tap is the OLD DESTINATION
                        # pixel with unchanged weights, as in the LINEAR blend. The result is
                        # written unless the whole 4x4 window is outside.
                        bvs = tl.load(dst + ys.to(tl.int64) * dst_rs + xs.to(tl.int64) * cn + c,
                                      mask=m, other=0)
                    v = tl.load(src + idx, mask=m & ok, other=bvs)
                else:
                    v = tl.load(src + idx, mask=m, other=0)
                if U8 or U16:
                    v = v.to(tl.float32)
                # BICUBIC_UPDATE_ACC_VEC: rowsum = fma(v, wx, rowsum)
                rowsum = _fma(v, wx[k], rowsum, FUSED)
            acc = _fma(rowsum, wy_r, acc, FUSED)
        if BORDER == 0:
            if c == 0:
                bv = bv0
            elif c == 1:
                bv = bv1
            elif c == 2:
                bv = bv2
            else:
                bv = bv3
            acc = tl.where(m & all_out, tl.full(acc.shape, bv.to(tl.float32), tl.float32), acc)
        if U8:
            out = tl.minimum(tl.maximum(_triton_rint(acc), 0.0), 255.0).to(tl.uint8)
        elif U16:
            out = tl.minimum(tl.maximum(_triton_rint(acc), 0.0), 65535.0).to(tl.uint16)
        else:
            out = acc
        cstore = m & ~all_out if TRANSPARENT else m
        tl.store(dst + ys.to(tl.int64) * dst_rs + xs.to(tl.int64) * cn + c, out, mask=cstore)
