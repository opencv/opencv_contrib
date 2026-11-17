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
