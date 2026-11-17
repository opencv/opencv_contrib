# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright 2026, The FlagOS Contributors.
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
