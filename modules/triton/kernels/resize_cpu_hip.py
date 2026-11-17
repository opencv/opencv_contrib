# Generated HIP resize kernels; regenerate with tools/make_kernels_cpu_hip.py.
# source_sha256: e570e25fbcf090ef3e6f6d0345c643fb322d73e7dffb74165cf6321b927c115c
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
#
# GPU resize kernels reproducing the selected OpenCV CPU sampling and rounding.
# The cpu prefix describes numerical behavior; every kernel here runs on the GPU.
# src/dst_rs and src/dst_bs are row/batch strides in typed elements, not bytes.
# xofs/yofs hold source pixel/row indices; ax/by/byf hold interpolation weights.

import triton
import triton.language as tl
from triton.language.extra import libdevice


# HIP/AMDGCN replacements for missing libdevice primitives: add_rn, mul_rn, div_rn, rint.
# Explicit vector arithmetic preserves separate add/multiply/subtract operations.
# v_rndne_f32 rounds to the nearest even integer.
# Division uses Triton's `/` lowering, not a correctly rounded div_rn primitive.

@triton.jit
def _triton_add_rn(a, b):
    return tl.inline_asm_elementwise("v_add_f32 $0, $1, $2", "=v,v,v", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _triton_mul_rn(a, b):
    return tl.inline_asm_elementwise("v_mul_f32 $0, $1, $2", "=v,v,v", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _triton_div_rn(a, b):
    return a / b

@triton.jit
def _triton_rint(a):
    return tl.inline_asm_elementwise("v_rndne_f32 $0, $1", "=v,v", [a], dtype=tl.float32, is_pure=True, pack=1)



@triton.jit
def _cpu_tile(dst_h, W, cn: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    """Pixel-major tile: BLOCK_H output rows x BLOCK_W output PIXELS (all cn channels each).

    Every per-pixel quantity (offset/weight tables, clamped tap columns, window
    origin) is computed once per pixel and reused across the static channel loop
    to avoid repeating coordinate and weight calculations for each channel.
    """
    rows = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    px = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]
    dst_w = W // cn
    return rows, px, rows < dst_h, px < dst_w, dst_w


# Compile identity: cn and tile dimensions are explicit constexprs. Runtime
# arguments, including pointers, are do_not_specialize, so the C++ runtime need
# not reproduce implicit Triton value/alignment specialization keys.
#
# Tile layout: area kernels count output pixels on grid axis 1; nearest and
# separable kernels count channel values. Both use BLOCK_H output rows.
# Row-sliding nearest/separable kernels reuse per-element tables and horizontal
# results when the source row repeats or advances by one. Terms retain the CPU
# evaluation order to preserve float32 rounding.
@triton.jit(do_not_specialize=['src', 'dst', 'xofs', 'yofs', 'dst_h', 'W', 'src_rs', 'dst_rs', 'src_bs', 'dst_bs'])
def _resize_cpu_nearest_kernel(src, dst, xofs, yofs, dst_h, W, cn: tl.constexpr,
                        src_rs, dst_rs, src_bs, dst_bs,
                        BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    # e indexes destination channel values; px indexes pixels and m masks the row tail.
    # W = output width * cn. so includes the channel, while sb/db select the batch.
    e = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    m = e < W
    px = e // cn
    b = tl.program_id(2).to(tl.int64)
    so = tl.load(xofs + px, mask=m, other=0) * cn + (e - px * cn)
    sb = src + b * src_bs
    db = dst + b * dst_bs
    r0 = tl.program_id(0) * BLOCK_H
    prev = tl.load(yofs + r0)
    v = tl.load(sb + prev.to(tl.int64) * src_rs + so, mask=m)
    for r in range(r0, tl.minimum(r0 + BLOCK_H, dst_h)):
        sy = tl.load(yofs + r)
        if sy != prev:
            v = tl.load(sb + sy.to(tl.int64) * src_rs + so, mask=m)
            prev = sy
        tl.store(db + r.to(tl.int64) * dst_rs + e, v, mask=m)


@triton.jit
def _cpu_hlin(rp, o0, o1, a0, a1, inside, m, U8: tl.constexpr):
    """Horizontal LINEAR sum for one source row; o0/o1 include channel offsets.

    inside selects the two-tap region; beyond it the clamped value is copied.
    U8 retains coefficients scaled by 2048; float operations round separately.
    """
    # rp is a source row pointer; o0/o1 are element offsets and a0/a1 are tap weights.
    v0 = tl.load(rp + o0, mask=m, other=0)
    v1 = tl.load(rp + o1, mask=m, other=0)
    if U8:
        v0 = v0.to(tl.int32)
        v1 = v1.to(tl.int32)
        h = tl.where(inside, v0 * a0 + v1 * a1, v0 * 2048)            # ONE = 2048 beyond xmax
    else:
        h = tl.where(inside, _triton_add_rn(_triton_mul_rn(v0, a0),
                                              _triton_mul_rn(v1, a1)), v0)
    return h


@triton.jit
def _cpu_hcub(rp, c0, c1, c2, c3, a0, a1, a2, a3, m, U8: tl.constexpr):
    """Horizontal CUBIC sum over four clamped taps of one source row.

    U8 keeps the integer coefficient scale for the vertical pass; float sums
    round separately in the CPU order ((t0 + t1) + t2) + t3.
    """
    # c0..c3 are clamped element offsets; a0..a3 are the corresponding tap weights.
    v0 = tl.load(rp + c0, mask=m, other=0)
    v1 = tl.load(rp + c1, mask=m, other=0)
    v2 = tl.load(rp + c2, mask=m, other=0)
    v3 = tl.load(rp + c3, mask=m, other=0)
    if U8:
        h = v0.to(tl.int32) * a0 + v1.to(tl.int32) * a1 + v2.to(tl.int32) * a2 + v3.to(tl.int32) * a3
    else:
        h = _triton_add_rn(_triton_add_rn(_triton_add_rn(_triton_mul_rn(v0, a0),
                                                               _triton_mul_rn(v1, a1)),
                                              _triton_mul_rn(v2, a2)),
                             _triton_mul_rn(v3, a3))
    return h


@triton.jit
def _cpu_row(base, sy, t, src_h, src_rs):
    r = tl.minimum(tl.maximum(sy + t, 0), src_h - 1).to(tl.int64)
    return base + r * src_rs


@triton.jit
def _cpu_tap(sx, t, src_w, cn: tl.constexpr):
    return tl.minimum(tl.maximum(sx + t, 0), src_w - 1) * cn


@triton.jit(do_not_specialize=['src', 'dst', 'xofs', 'ax', 'yofs', 'by', 'byf', 'src_h', 'src_w', 'dst_h', 'W', 'xmax', 'src_rs', 'dst_rs', 'src_bs', 'dst_bs'])
def _resize_cpu_sep_kernel(src, dst, xofs, ax, yofs, by, byf,
                    src_h, src_w, dst_h, W, cn: tl.constexpr, xmax,
                    src_rs, dst_rs, src_bs, dst_bs,
                    KSIZE: tl.constexpr, U8: tl.constexpr,
                    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    """Generic separable path: LINEAR / AREA-upscale emulation (KSIZE=2), CUBIC (KSIZE=4).
    Row-sliding: see the comment above _resize_cpu_nearest_kernel."""
    # xofs/yofs are source base indices; CUBIC starts one pixel/row before the base.
    # ax/by carry fixed-point weights for U8; byf carries floating vertical weights.
    # r0 is the first destination row of this tile.
    e = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    m = e < W
    pm = m
    px = e // cn
    ch = e - px * cn
    b = tl.program_id(2).to(tl.int64)
    sx = tl.load(xofs + px, mask=pm, other=0)
    base = src + b * src_bs
    db = dst + b * dst_bs
    do = e
    r0 = tl.program_id(0) * BLOCK_H
    r1 = tl.minimum(r0 + BLOCK_H, dst_h)
    prev = tl.load(yofs + r0)
    if KSIZE == 2:
        a0 = tl.load(ax + px * 2, mask=pm, other=0)
        a1 = tl.load(ax + px * 2 + 1, mask=pm, other=0)
        o0 = sx * cn + ch
        o1 = tl.minimum(sx + 1, src_w - 1) * cn + ch
        inside = px < xmax
        h0 = _cpu_hlin(_cpu_row(base, prev, 0, src_h, src_rs), o0, o1, a0, a1, inside, m, U8)
        h1 = _cpu_hlin(_cpu_row(base, prev, 1, src_h, src_rs), o0, o1, a0, a1, inside, m, U8)
        for r in range(r0, r1):
            sy = tl.load(yofs + r)
            if sy != prev:
                if sy == prev + 1:
                    h0 = h1
                    h1 = _cpu_hlin(_cpu_row(base, sy, 1, src_h, src_rs), o0, o1, a0, a1, inside, m, U8)
                else:
                    h0 = _cpu_hlin(_cpu_row(base, sy, 0, src_h, src_rs), o0, o1, a0, a1, inside, m, U8)
                    h1 = _cpu_hlin(_cpu_row(base, sy, 1, src_h, src_rs), o0, o1, a0, a1, inside, m, U8)
                prev = sy
            b0 = tl.load(by + r * 2)
            b1 = tl.load(by + r * 2 + 1)
            if U8:
                v = ((((b0 * (h0 >> 4)) >> 16) + ((b1 * (h1 >> 4)) >> 16) + 2) >> 2) & 255
                out = v.to(tl.uint8)
            else:
                out = _triton_add_rn(_triton_mul_rn(h0, b0), _triton_mul_rn(h1, b1))
            tl.store(db + r.to(tl.int64) * dst_rs + do, out, mask=m)
    else:
        c0 = _cpu_tap(sx, -1, src_w, cn) + ch
        c1 = _cpu_tap(sx, 0, src_w, cn) + ch
        c2 = _cpu_tap(sx, 1, src_w, cn) + ch
        c3 = _cpu_tap(sx, 2, src_w, cn) + ch
        a0 = tl.load(ax + px * 4, mask=pm, other=0)
        a1 = tl.load(ax + px * 4 + 1, mask=pm, other=0)
        a2 = tl.load(ax + px * 4 + 2, mask=pm, other=0)
        a3 = tl.load(ax + px * 4 + 3, mask=pm, other=0)
        h0 = _cpu_hcub(_cpu_row(base, prev, -1, src_h, src_rs), c0, c1, c2, c3, a0, a1, a2, a3, m, U8)
        h1 = _cpu_hcub(_cpu_row(base, prev, 0, src_h, src_rs), c0, c1, c2, c3, a0, a1, a2, a3, m, U8)
        h2 = _cpu_hcub(_cpu_row(base, prev, 1, src_h, src_rs), c0, c1, c2, c3, a0, a1, a2, a3, m, U8)
        h3 = _cpu_hcub(_cpu_row(base, prev, 2, src_h, src_rs), c0, c1, c2, c3, a0, a1, a2, a3, m, U8)
        for r in range(r0, r1):
            sy = tl.load(yofs + r)
            if sy != prev:
                if sy == prev + 1:
                    h0 = h1
                    h1 = h2
                    h2 = h3
                    h3 = _cpu_hcub(_cpu_row(base, sy, 2, src_h, src_rs), c0, c1, c2, c3, a0, a1, a2, a3, m, U8)
                else:
                    h0 = _cpu_hcub(_cpu_row(base, sy, -1, src_h, src_rs), c0, c1, c2, c3, a0, a1, a2, a3, m, U8)
                    h1 = _cpu_hcub(_cpu_row(base, sy, 0, src_h, src_rs), c0, c1, c2, c3, a0, a1, a2, a3, m, U8)
                    h2 = _cpu_hcub(_cpu_row(base, sy, 1, src_h, src_rs), c0, c1, c2, c3, a0, a1, a2, a3, m, U8)
                    h3 = _cpu_hcub(_cpu_row(base, sy, 2, src_h, src_rs), c0, c1, c2, c3, a0, a1, a2, a3, m, U8)
                prev = sy
            if U8:
                # SIMD lanes (first (W//8)*8 elements of the row): float path ...
                bf0 = tl.load(byf + r * 4)
                bf1 = tl.load(byf + r * 4 + 1)
                bf2 = tl.load(byf + r * 4 + 2)
                bf3 = tl.load(byf + r * 4 + 3)
                t3 = _triton_mul_rn(h3.to(tl.float32), bf3)
                t2 = _triton_add_rn(_triton_mul_rn(h2.to(tl.float32), bf2), t3)
                t1 = _triton_add_rn(_triton_mul_rn(h1.to(tl.float32), bf1), t2)
                t0 = _triton_add_rn(_triton_mul_rn(h0.to(tl.float32), bf0), t1)
                vf = tl.minimum(tl.maximum(_triton_rint(t0), 0.0), 255.0).to(tl.int32)
                # ... scalar tail: FixedPtCast<int, uchar, 22>
                ib0 = tl.load(by + r * 4)
                ib1 = tl.load(by + r * 4 + 1)
                ib2 = tl.load(by + r * 4 + 2)
                ib3 = tl.load(by + r * 4 + 3)
                s = h0 * ib0 + h1 * ib1 + h2 * ib2 + h3 * ib3
                vi = tl.minimum(tl.maximum((s + (1 << 21)) >> 22, 0), 255)
                out = tl.where(e < (W // 8) * 8, vf, vi).to(tl.uint8)
            else:
                b0 = tl.load(by + r * 4)
                b1 = tl.load(by + r * 4 + 1)
                b2 = tl.load(by + r * 4 + 2)
                b3 = tl.load(by + r * 4 + 3)
                p0 = _triton_mul_rn(h0, b0)
                p1 = _triton_mul_rn(h1, b1)
                p2 = _triton_mul_rn(h2, b2)
                p3 = _triton_mul_rn(h3, b3)
                vec = _triton_add_rn(p0, _triton_add_rn(p1, _triton_add_rn(p2, p3)))
                tail = _triton_add_rn(_triton_add_rn(_triton_add_rn(p0, p1), p2), p3)
                out = tl.where(e < (W // 4) * 4, vec, tail)
            tl.store(db + r.to(tl.int64) * dst_rs + do, out, mask=m)


@triton.jit
def _cpu_f32_tap(rp0, sx0, ky, kx, cn: tl.constexpr, c: tl.constexpr, src_h, src_w, sy0, src_rs, m,
                 MASKED: tl.constexpr):
    col = sx0 + kx
    q = rp0 + ky * src_rs + col * cn + c
    if MASKED:
        ok = m & (sy0 + ky < src_h) & (col < src_w)
        v = tl.load(q, mask=ok, other=0.0)
    else:
        ok = m
        v = tl.load(q)
    return v, ok


@triton.jit
def _cpu_f32_tapv(rp0, sx0, ky, kx, cn: tl.constexpr, CP: tl.constexpr, src_h, src_w, sy0, src_rs, m, ch,
                  MASKED: tl.constexpr):
    """One window element for every channel of a pixel at once: (BLOCK_H, BLOCK_W, CP).

    C3 is loaded as four lanes with lane 3 masked out and added as an exact +0.0, which
    cannot change a finite sum; a masked lane is never read."""
    col = sx0 + kx
    q = (rp0 + ky * src_rs + col * cn)[:, :, None] + ch[None, None, :]
    if MASKED:
        ok = m & (sy0 + ky < src_h) & (col < src_w)
        v = tl.load(q, mask=ok[:, :, None] & (ch[None, None, :] < cn), other=0.0)
    else:
        ok = m
        v = tl.load(q, mask=(ch[None, None, :] < cn) & m[:, :, None], other=0.0)
    return v, ok


@triton.jit
def _cpu_f32_window_v(rp0, sx0, cn: tl.constexpr, CP: tl.constexpr, ch, IX, area, src_h, src_w, sy0, src_rs,
                      m, MASKED: tl.constexpr):
    """resizeAreaFast_ float32 window sum for all channels of a pixel, in CPU order --
    groups of four in row-major k, acc = acc + ((v0 + v1) + v2) + v3, then singles."""
    acc = tl.zeros((m.shape[0], m.shape[1], CP), dtype=tl.float32)
    tot = tl.zeros((m.shape[0], m.shape[1], CP), dtype=tl.float32)
    cnt = tl.zeros((m.shape[0], m.shape[1]), dtype=tl.int32)
    nfull = (area // 4) * 4
    ky = IX * 0
    kx = IX * 0
    for k0 in range(0, nfull, 4):
        v0, o0 = _cpu_f32_tapv(rp0, sx0, ky, kx, cn, CP, src_h, src_w, sy0, src_rs, m, ch, MASKED)
        ky, kx = _cpu_step(ky, kx, IX)
        v1, o1 = _cpu_f32_tapv(rp0, sx0, ky, kx, cn, CP, src_h, src_w, sy0, src_rs, m, ch, MASKED)
        ky, kx = _cpu_step(ky, kx, IX)
        v2, o2 = _cpu_f32_tapv(rp0, sx0, ky, kx, cn, CP, src_h, src_w, sy0, src_rs, m, ch, MASKED)
        ky, kx = _cpu_step(ky, kx, IX)
        v3, o3 = _cpu_f32_tapv(rp0, sx0, ky, kx, cn, CP, src_h, src_w, sy0, src_rs, m, ch, MASKED)
        ky, kx = _cpu_step(ky, kx, IX)
        acc = _triton_add_rn(acc, _triton_add_rn(_triton_add_rn(_triton_add_rn(v0, v1), v2), v3))
        if MASKED:
            tot = tl.where(o0[:, :, None], _triton_add_rn(tot, v0), tot)
            tot = tl.where(o1[:, :, None], _triton_add_rn(tot, v1), tot)
            tot = tl.where(o2[:, :, None], _triton_add_rn(tot, v2), tot)
            tot = tl.where(o3[:, :, None], _triton_add_rn(tot, v3), tot)
            cnt = cnt + o0.to(tl.int32) + o1.to(tl.int32) + o2.to(tl.int32) + o3.to(tl.int32)
    for k in range(nfull, area):
        v, o = _cpu_f32_tapv(rp0, sx0, ky, kx, cn, CP, src_h, src_w, sy0, src_rs, m, ch, MASKED)
        ky, kx = _cpu_step(ky, kx, IX)
        acc = _triton_add_rn(acc, v)
        if MASKED:
            tot = tl.where(o[:, :, None], _triton_add_rn(tot, v), tot)
            cnt = cnt + o.to(tl.int32)
    return acc, tot, cnt


@triton.jit
def _cpu_step(ky, kx, IX):
    """Advance the row-major window position by one (replaces k // IX, k % IX)."""
    kx = kx + 1
    if kx == IX:
        kx = kx * 0
        ky = ky + 1
    return ky, kx


@triton.jit
def _cpu_f32_window(rp0, sx0, cn: tl.constexpr, c: tl.constexpr, IX, area, src_h, src_w, sy0, src_rs,
                    m, MASKED: tl.constexpr):
    """resizeAreaFast_ float32 window sum in CPU order -- groups of four in row-major k,
    acc = acc + ((v0 + v1) + v2) + v3, then singles -- plus the sequential in-image sum
    and count the edge formula uses."""
    acc = tl.zeros(m.shape, dtype=tl.float32)
    tot = tl.zeros(m.shape, dtype=tl.float32)
    cnt = tl.zeros(m.shape, dtype=tl.int32)
    nfull = (area // 4) * 4
    ky = IX * 0
    kx = IX * 0
    for k0 in range(0, nfull, 4):
        v0, o0 = _cpu_f32_tap(rp0, sx0, ky, kx, cn, c, src_h, src_w, sy0, src_rs, m, MASKED)
        ky, kx = _cpu_step(ky, kx, IX)
        v1, o1 = _cpu_f32_tap(rp0, sx0, ky, kx, cn, c, src_h, src_w, sy0, src_rs, m, MASKED)
        ky, kx = _cpu_step(ky, kx, IX)
        v2, o2 = _cpu_f32_tap(rp0, sx0, ky, kx, cn, c, src_h, src_w, sy0, src_rs, m, MASKED)
        ky, kx = _cpu_step(ky, kx, IX)
        v3, o3 = _cpu_f32_tap(rp0, sx0, ky, kx, cn, c, src_h, src_w, sy0, src_rs, m, MASKED)
        ky, kx = _cpu_step(ky, kx, IX)
        acc = _triton_add_rn(acc, _triton_add_rn(_triton_add_rn(_triton_add_rn(v0, v1), v2), v3))
        if MASKED:
            tot = tl.where(o0, _triton_add_rn(tot, v0), tot)
            tot = tl.where(o1, _triton_add_rn(tot, v1), tot)
            tot = tl.where(o2, _triton_add_rn(tot, v2), tot)
            tot = tl.where(o3, _triton_add_rn(tot, v3), tot)
            cnt = cnt + o0.to(tl.int32) + o1.to(tl.int32) + o2.to(tl.int32) + o3.to(tl.int32)
    for k in range(nfull, area):
        v, o = _cpu_f32_tap(rp0, sx0, ky, kx, cn, c, src_h, src_w, sy0, src_rs, m, MASKED)
        ky, kx = _cpu_step(ky, kx, IX)
        acc = _triton_add_rn(acc, v)
        if MASKED:
            tot = tl.where(o, _triton_add_rn(tot, v), tot)
            cnt = cnt + o.to(tl.int32)
    return acc, tot, cnt


@triton.jit(do_not_specialize=['src', 'dst', 'src_h', 'src_w', 'dst_h', 'W', 'dwidth1', 'IX', 'IY', 'src_rs', 'dst_rs', 'src_bs', 'dst_bs'])
def _resize_cpu_area_fast_kernel(src, dst, src_h, src_w, dst_h, W, cn: tl.constexpr, dwidth1, scale, IX, IY,
                                 src_rs, dst_rs, src_bs, dst_bs,
                                 TWO_BY_TWO: tl.constexpr, U8: tl.constexpr,
                                 BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, RY: tl.constexpr):
    """resizeAreaFast_: exact integer ratio, including its count-based edges.

    Tiles wholly inside the full-window region (every window in the image, column
    below dwidth1) take an unmasked branch; only edge tiles evaluate the
    count-based formula and its division. Runtime loops walk the window to
    avoid expanding generated code in proportion to IX*IY for large windows.
    """
    rows, px, rm, pm, dst_w = _cpu_tile(dst_h, W, cn, BLOCK_H, BLOCK_W)
    m = rm & pm
    b = tl.program_id(2).to(tl.int64)
    dw1 = dwidth1 // cn                                   # pixels with a full window in x
    sy0 = rows * IY
    sx0 = px * IX
    rp0 = src + b * src_bs + sy0.to(tl.int64) * src_rs
    dp = dst + b * dst_bs + rows.to(tl.int64) * dst_rs + px * cn
    r_end = tl.program_id(0) * BLOCK_H + BLOCK_H
    p_end = tl.program_id(1) * BLOCK_W + BLOCK_W
    tile_full = (r_end <= dst_h) & (r_end * IY <= src_h) & (p_end <= dw1) & (p_end <= dst_w)
    area = IX * IY
    if tile_full:
        if TWO_BY_TWO:
            # A 2x2 window is the case where consecutive pixels' windows are CONTIGUOUS in
            # memory (pixel px owns columns [2px*cn, 2px*cn + 2cn)), so the tile is ONE 4-D
            # load per source row: (pixel, column, channel); CP = 1 for C1, else 4 (C3 masks
            # lane 3). Both CPU orders are formed from it -- the SIMD-lane order
            # (w0 + w1) + (w2 + w3) for the first (dwidth1/4)*4 elements of a C1/C4 row,
            # ((w0 + w1) + w2) + w3 everywhere else -- and uint8 sums exactly in any order.
            CP: tl.constexpr = 1 if cn == 1 else 4
            ch = tl.arange(0, CP)
            e3 = ((px * (2 * cn))[:, :, None, None] + (tl.arange(0, 2) * cn)[None, None, :, None]
                  + ch[None, None, None, :])
            m3 = m[:, :, None, None] & (ch[None, None, None, :] < cn)
            base0 = (b * src_bs + sy0.to(tl.int64) * src_rs)[:, :, None, None]
            r0v = tl.load(src + base0 + e3, mask=m3, other=0)
            r1v = tl.load(src + (base0 + src_rs) + e3, mask=m3, other=0)
            a00, a01 = tl.split(tl.permute(r0v, (0, 1, 3, 2)))
            a10, a11 = tl.split(tl.permute(r1v, (0, 1, 3, 2)))
            if U8:
                a8 = a00.to(tl.int32) + a01.to(tl.int32) + a10.to(tl.int32) + a11.to(tl.int32)
                outv = ((a8 + 2) >> 2).to(tl.uint8)
            else:
                vsimd = _triton_add_rn(_triton_add_rn(a00, a01), _triton_add_rn(a10, a11))
                vseq = _triton_add_rn(_triton_add_rn(_triton_add_rn(a00, a01), a10), a11)
                isimd = ((cn == 1) | (cn == 4)) & (px[:, :, None] * cn + ch[None, None, :] < (dwidth1 // 4) * 4)
                outv = _triton_mul_rn(tl.where(isimd, vsimd, vseq), scale)
            if cn == 1:
                tl.store(dp, tl.sum(outv, axis=2), mask=m)
            else:
                tl.store(dp[:, :, None] + ch[None, None, :], outv,
                         mask=m[:, :, None] & (ch[None, None, :] < cn))
        elif U8:
            # Integer sum: the order is irrelevant, so plain nested loops (no k // IX), and
            # with RY > 1 the window rows are split over RY lanes and summed at the end --
            # exact, and what keeps small outputs with large windows (16x down to 120x67)
            # from running on a few hundred threads.
            # One vector load per window element for every channel of the pixel
            # (CP lanes; C3 masks lane 3, which adds an exact 0 to an integer sum).
            kr = tl.arange(0, RY)[:, None, None, None]
            CP: tl.constexpr = 1 if cn == 1 else 4
            chv = tl.arange(0, CP)
            rp3 = (rp0 + sx0 * cn)[None, :, :, None]
            acc = tl.zeros([RY, BLOCK_H, BLOCK_W, CP], dtype=tl.int32)
            for ky0 in range(0, IY, RY):
                ky = ky0 + kr
                rp = rp3 + ky * src_rs
                okr = (ky < IY) & (chv[None, None, None, :] < cn)
                for kx in tl.range(0, IX, loop_unroll_factor=4):
                    acc += tl.load(rp + kx * cn + chv[None, None, None, :], mask=okr, other=0).to(tl.int32)
            tot = tl.sum(acc, axis=0)                     # (rows, pixels, CP): rows add exactly
            out8 = tl.minimum(tl.maximum(_triton_rint(_triton_mul_rn(tot.to(tl.float32), scale)), 0.0), 255.0).to(tl.uint8)
            if cn == 1:
                tl.store(dp, tl.sum(out8, axis=2))
            else:
                tl.store(dp[:, :, None] + chv[None, None, :], out8,
                         mask=m[:, :, None] & (chv[None, None, :] < cn))
        else:
            CP: tl.constexpr = 1 if cn == 1 else 4
            chv = tl.arange(0, CP)
            accv, _t, _n = _cpu_f32_window_v(rp0, sx0, cn, CP, chv, IX, area, src_h, src_w, sy0, src_rs, m, False)
            tl.store(dp[:, :, None] + chv[None, None, :],
                     _triton_mul_rn(accv, scale),
                     mask=m[:, :, None] & (chv[None, None, :] < cn))
    else:
        # Edge tiles: masked, count-based edge formula where the window is partial.
        in_img = sy0 < src_h
        full = (sy0 + IY <= src_h) & (px < dw1)
        live = m & in_img
        for c in tl.static_range(cn):
            if U8:
                s = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.int32)
                cnt = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.int32)
                for ky in range(0, IY):
                    rp = rp0 + ky * src_rs
                    rok = live & (sy0 + ky < src_h)
                    for kx in range(0, IX):
                        ok = rok & (sx0 + kx < src_w)
                        s += tl.load(rp + (sx0 + kx) * cn + c, mask=ok, other=0).to(tl.int32)
                        cnt += ok.to(tl.int32)
                gen = tl.minimum(tl.maximum(_triton_rint(_triton_mul_rn(s.to(tl.float32), scale)),
                                            0.0), 255.0).to(tl.int32)
                if TWO_BY_TWO & ((cn == 1) | (cn == 3) | (cn == 4)):
                    fullv = (s + 2) >> 2
                else:
                    fullv = gen
                q_e = _triton_div_rn(s.to(tl.float32), cnt.to(tl.float32))
                edge = tl.where(cnt > 0, tl.minimum(tl.maximum(_triton_rint(q_e), 0.0), 255.0).to(tl.int32), 0)
                out = tl.where(in_img, tl.where(full, fullv, edge), 0).to(tl.uint8)
            else:
                acc, tot, cnt = _cpu_f32_window(rp0, sx0, cn, c, IX, area, src_h, src_w, sy0, src_rs,
                                                live, True)
                fullv = _triton_mul_rn(acc, scale)
                if TWO_BY_TWO & ((cn == 1) | (cn == 4)):
                    okr = live & (sy0 + 1 < src_h)
                    okc = sx0 + 1 < src_w
                    w0 = tl.load(rp0 + sx0 * cn + c, mask=live, other=0.0)
                    w1 = tl.load(rp0 + (sx0 + 1) * cn + c, mask=live & okc, other=0.0)
                    w2 = tl.load(rp0 + src_rs + sx0 * cn + c, mask=okr, other=0.0)
                    w3 = tl.load(rp0 + src_rs + (sx0 + 1) * cn + c, mask=okr & okc, other=0.0)
                    vec = _triton_mul_rn(_triton_add_rn(_triton_add_rn(w0, w1), _triton_add_rn(w2, w3)), scale)
                    fullv = tl.where(px * cn + c < (dwidth1 // 4) * 4, vec, fullv)
                q_e = _triton_div_rn(tot, cnt.to(tl.float32))
                out = tl.where(in_img, tl.where(full, fullv, q_e), 0.0)
            tl.store(dp + c, out, mask=m)


@triton.jit(do_not_specialize=['src', 'dst', 'xs', 'xa', 'xc', 'ys', 'ya', 'yc', 'KX', 'KY', 'dst_h', 'W', 'src_rs', 'dst_rs', 'src_bs', 'dst_bs'])
def _resize_cpu_area_kernel(src, dst, xs, xa, xc, ys, ya, yc, KX, KY,
                     dst_h, W, cn: tl.constexpr, src_rs, dst_rs, src_bs, dst_bs,
                     U8: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    """resizeArea_: h = h + S*alpha over x taps in table order, acc = acc + beta*h over y taps.

    computeResizeAreaTab gives every pixel CONSECUTIVE source columns
    (xs[k] = xs[0] + k) and one weight for every interior tap; only the first and
    the last can be partial. So the x table is read once per pixel -- start column,
    first / interior / last weight -- instead of two loads per tap. The tables follow
    computeResizeAreaTab's contiguous-tap layout. uint8 input is finite,
    so a padded tap (weight 0) adds an exact +0.0 and needs no predicate on the sums;
    float32 can hold inf/NaN (inf * 0 = NaN), so there padded taps stay masked out.
    """
    rows = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    px = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]
    rm = rows < dst_h
    pm = px < W // cn
    m = rm & pm
    b = tl.program_id(2).to(tl.int64)
    ycnt = tl.load(yc + rows, mask=rm, other=0)
    xcnt = tl.load(xc + px, mask=pm, other=0)
    xrow = xa + px * KX
    x0 = tl.load(xs + px * KX, mask=pm, other=0) * cn
    a_first = tl.load(xrow, mask=pm, other=0.0)
    a_mid = tl.load(xrow + 1, mask=pm & (xcnt > 2), other=0.0)
    a_last = tl.load(xrow + tl.maximum(xcnt - 1, 0), mask=pm, other=0.0)
    last = xcnt - 1
    base = src + b * src_bs + x0
    dp = dst + b * dst_bs + rows.to(tl.int64) * dst_rs + px * cn
    acc0 = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    for ky in range(0, KY):
        ylive = ky < ycnt
        sy = tl.load(ys + rows * KY + ky, mask=rm, other=0).to(tl.int64)
        beta = tl.load(ya + rows * KY + ky, mask=rm, other=0.0)
        rp = base + sy * src_rs
        h0 = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
        h1 = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
        h2 = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
        h3 = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
        # Unroll pairs of taps to limit instruction growth for short tap lists.
        for kx in tl.range(0, KX, loop_unroll_factor=2):
            xlive = kx < xcnt
            a = tl.where(kx == 0, a_first, tl.where(kx == last, a_last, a_mid))
            q = rp + kx * cn
            if U8:
                lm = m & xlive
                a = tl.where(xlive, a, 0.0)
                h0 = _triton_add_rn(h0, _triton_mul_rn(tl.load(q, mask=lm, other=0).to(tl.float32), a))
                if cn > 1:
                    h1 = _triton_add_rn(h1, _triton_mul_rn(tl.load(q + 1, mask=lm, other=0).to(tl.float32), a))
                if cn > 2:
                    h2 = _triton_add_rn(h2, _triton_mul_rn(tl.load(q + 2, mask=lm, other=0).to(tl.float32), a))
                if cn > 3:
                    h3 = _triton_add_rn(h3, _triton_mul_rn(tl.load(q + 3, mask=lm, other=0).to(tl.float32), a))
            else:
                lm = m & ylive & xlive
                h0 = tl.where(lm, _triton_add_rn(h0, _triton_mul_rn(tl.load(q, mask=lm, other=0.0), a)), h0)
                if cn > 1:
                    h1 = tl.where(lm, _triton_add_rn(h1, _triton_mul_rn(tl.load(q + 1, mask=lm, other=0.0), a)), h1)
                if cn > 2:
                    h2 = tl.where(lm, _triton_add_rn(h2, _triton_mul_rn(tl.load(q + 2, mask=lm, other=0.0), a)), h2)
                if cn > 3:
                    h3 = tl.where(lm, _triton_add_rn(h3, _triton_mul_rn(tl.load(q + 3, mask=lm, other=0.0), a)), h3)
        if U8:
            acc0 = _triton_add_rn(acc0, _triton_mul_rn(beta, h0))
            acc1 = _triton_add_rn(acc1, _triton_mul_rn(beta, h1))
            acc2 = _triton_add_rn(acc2, _triton_mul_rn(beta, h2))
            acc3 = _triton_add_rn(acc3, _triton_mul_rn(beta, h3))
        else:
            acc0 = tl.where(ylive, _triton_add_rn(acc0, _triton_mul_rn(beta, h0)), acc0)
            acc1 = tl.where(ylive, _triton_add_rn(acc1, _triton_mul_rn(beta, h1)), acc1)
            acc2 = tl.where(ylive, _triton_add_rn(acc2, _triton_mul_rn(beta, h2)), acc2)
            acc3 = tl.where(ylive, _triton_add_rn(acc3, _triton_mul_rn(beta, h3)), acc3)
    if U8:
        tl.store(dp, tl.minimum(tl.maximum(_triton_rint(acc0), 0.0), 255.0).to(tl.uint8), mask=m)
        if cn > 1:
            tl.store(dp + 1, tl.minimum(tl.maximum(_triton_rint(acc1), 0.0), 255.0).to(tl.uint8), mask=m)
        if cn > 2:
            tl.store(dp + 2, tl.minimum(tl.maximum(_triton_rint(acc2), 0.0), 255.0).to(tl.uint8), mask=m)
        if cn > 3:
            tl.store(dp + 3, tl.minimum(tl.maximum(_triton_rint(acc3), 0.0), 255.0).to(tl.uint8), mask=m)
    else:
        tl.store(dp, acc0, mask=m)
        if cn > 1:
            tl.store(dp + 1, acc1, mask=m)
        if cn > 2:
            tl.store(dp + 2, acc2, mask=m)
        if cn > 3:
            tl.store(dp + 3, acc3, mask=m)


# HIP-specific complete-window area kernels.
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
#
# HIP GPU resize fast paths for complete 2x2 area windows in uint8 images.
# Host predicates admit only supported channel counts, alignment and full windows.
# Row/batch strides count uint8 elements (bytes); packed loads convert to word offsets.
# C1 packs four grayscale pixels per word; C4 packs four channels of one pixel.

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit(do_not_specialize=['src', 'dst', 'dst_h', 'dst_w', 'src_rs', 'dst_rs', 'src_bs', 'dst_bs'])
def _hip_area2_packed(src, dst, dst_h, dst_w, src_rs, dst_rs, src_bs, dst_bs,
                      C1: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    """Four independent 8-bit sums in pairs of 16-bit lanes; max sum is 1020.

    BLOCK_W remains output pixels, including when C1 packs four pixels into
    a word. The host proves alignment and complete 2x2 windows before launch.
    """
    GROUPS: tl.constexpr = BLOCK_W // 4 if C1 else BLOCK_W
    y = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    x = tl.program_id(1) * GROUPS + tl.arange(0, GROUPS)[None, :]
    width = dst_w // 4 if C1 else dst_w
    batch = tl.program_id(2).to(tl.int64)
    sp = src.to(tl.pointer_type(tl.uint32)) + batch * (src_bs // 4) + y.to(tl.int64) * (src_rs // 2) + 2 * x
    dp = dst.to(tl.pointer_type(tl.uint32)) + batch * (dst_bs // 4) + y.to(tl.int64) * (dst_rs // 4) + x
    mask = (y < dst_h) & (x < width)
    a = tl.load(sp, mask, 0)
    b = tl.load(sp + 1, mask, 0)
    c = tl.load(sp + src_rs // 4, mask, 0)
    d = tl.load(sp + src_rs // 4 + 1, mask, 0)
    if C1:
        # Separate byte positions into 16-bit lanes; each lane sums one 2x2 window.
        ac = (a & 0x00ff00ff) + ((a >> 8) & 0x00ff00ff) + (c & 0x00ff00ff) + ((c >> 8) & 0x00ff00ff)
        bd = (b & 0x00ff00ff) + ((b >> 8) & 0x00ff00ff) + (d & 0x00ff00ff) + ((d >> 8) & 0x00ff00ff)
        # Add the integer rounding bias to each lane, then divide each sum by four.
        ac = (ac + 0x00020002) >> 2
        bd = (bd + 0x00020002) >> 2
        # Repack the four averaged grayscale pixels into their output byte order.
        value = (ac & 255) | ((ac >> 8) & 0xff00) | ((bd & 255) << 16) | ((bd & 0xff0000) << 8)
    else:
        # C4 keeps channels independent: average even/odd bytes in separate lanes.
        lo = ((a & 0x00ff00ff) + (b & 0x00ff00ff) + (c & 0x00ff00ff) + (d & 0x00ff00ff) + 0x00020002) >> 2
        hi = (((a >> 8) & 0x00ff00ff) + ((b >> 8) & 0x00ff00ff) + ((c >> 8) & 0x00ff00ff) + ((d >> 8) & 0x00ff00ff) + 0x00020002) >> 2
        # Interleave the averaged channels into one four-channel output pixel.
        value = (lo & 0x00ff00ff) | ((hi & 0x00ff00ff) << 8)
    tl.store(dp, value, mask)


@triton.jit(do_not_specialize=['src', 'dst', 'dst_h', 'W', 'src_rs', 'dst_rs', 'src_bs', 'dst_bs'])
def _hip_area2_channels(src, dst, dst_h, W, src_rs, dst_rs, src_bs, dst_bs,
                        BLOCK_E: tl.constexpr):
    """C3: consecutive channel values, without the generic padded fourth lane."""
    e = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    y = tl.program_id(0).to(tl.int64)
    b = tl.program_id(2).to(tl.int64)
    p = src + b * src_bs + y * 2 * src_rs + (e // 3) * 6 + e % 3
    mask = e < W
    a = tl.load(p, mask, 0).to(tl.int32)
    a += tl.load(p + 3, mask, 0).to(tl.int32)
    a += tl.load(p + src_rs, mask, 0).to(tl.int32)
    a += tl.load(p + src_rs + 3, mask, 0).to(tl.int32)
    tl.store(dst + b * dst_bs + y * dst_rs + e, ((a + 2) >> 2).to(tl.uint8), mask)
