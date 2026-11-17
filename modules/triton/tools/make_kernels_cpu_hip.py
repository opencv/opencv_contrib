#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

# Copyright 2026, The FlagOS Contributors.
"""Generate HIP kernel sources from the module's CUDA-compatible sources.

Run without arguments to write kernels/<op>_cpu_hip.py, or use --check to verify
that the checked-in files match the current sources and this generator.

The source kernels use libdevice operations to preserve floating-point rounding.
The HIP libdevice implementation does not provide every operation, so this tool
substitutes the shims below. Kernels without those imports can be used directly.
Resize appends kernels/resize_hip_fast.py; cvtColor appends kernels/cvtcolor_hip_fast.py.
Edit those maintained sources, not their copies in generated files.

Each generated file records the digest of its local source for traceability.
"""
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = os.path.join(HERE, "..", "kernels")
OPS = ("resize", "cvtcolor", "warpaffine")

ANCHOR = "from triton.language.extra import libdevice\n"

# Common resize arithmetic shim. Keeping this block literal makes the generated
# source reproducible without importing Triton or a backend compiler.
_RESIZE_SHIM = """\


# HIP/AMDGCN replacements for missing libdevice primitives.
# Explicit vector add/multiply instructions prevent multiply-add contraction.
# Division uses Triton's `/` lowering; it is not a correctly rounded div_rn primitive.
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

"""

# Per-primitive bodies, for a kernel whose missing set differs from resize's.
_SHIMS = {
    "rint": """
@triton.jit
def _triton_rint(a):
    return tl.inline_asm_elementwise("v_rndne_f32 $0, $1", "=v,v", [a], dtype=tl.float32, is_pure=True, pack=1)

""",
    "add_rn": """
@triton.jit
def _triton_add_rn(a, b):
    return tl.inline_asm_elementwise("v_add_f32 $0, $1, $2", "=v,v,v", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)

""",
    "mul_rn": """
@triton.jit
def _triton_mul_rn(a, b):
    return tl.inline_asm_elementwise("v_mul_f32 $0, $1, $2", "=v,v,v", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)

""",
    "sub_rn": """
@triton.jit
def _triton_sub_rn(a, b):
    return tl.inline_asm_elementwise("v_sub_f32 $0, $1, $2", "=v,v,v", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)

""",
    "div_rn": """
@triton.jit
def _triton_div_rn(a, b):
    return a / b
""",
}

# Fixed order: two kernels missing the same primitives derive identical text.
MISSING_ON_HIP = ("add_rn", "mul_rn", "div_rn", "sub_rn", "rint")
RESIZE_SET = ("add_rn", "mul_rn", "div_rn")


def shim_for(text: str) -> str:
    """The shim block for the primitives `text` calls and flagtree's hip lacks.

    The common resize set uses a fixed block. Other sets are composed in the
    order of MISSING_ON_HIP and name the primitives they provide.
    """
    used = tuple(n for n in MISSING_ON_HIP if re.search(rf"\blibdevice\.{n}\(", text))
    if not used:
        return ""
    if used == RESIZE_SET:
        return _RESIZE_SHIM
    arith = [n for n in used if n not in ("div_rn", "rint")]
    head = ("\n\n# HIP/AMDGCN replacements for missing libdevice primitives: "
            + ", ".join(used) + ".\n")
    if arith:
        head += "# Explicit vector arithmetic preserves separate add/multiply/subtract operations.\n"
    if "rint" in used:
        head += "# v_rndne_f32 rounds to the nearest even integer.\n"
    if "div_rn" in used:
        head += "# Division uses Triton's `/` lowering, not a correctly rounded div_rn primitive.\n"
    return head + "".join(_SHIMS[n] for n in used)


def paths(op):
    return (os.path.join(KERNELS, f"{op}_cpu.py"), os.path.join(KERNELS, f"{op}_cpu_hip.py"))


def needs_derivation(op):
    src = paths(op)[0]
    return os.path.isfile(src) and ANCHOR in open(src, encoding="utf-8").read()


def derive(op):
    src = paths(op)[0]
    raw = open(src, "rb").read()
    text = raw.decode("utf-8")
    hits = text.count(ANCHOR)
    if hits != 1:
        sys.exit(f"{src}: the libdevice anchor appears {hits} times; the generator needs updating")
    header = (f"# Generated HIP {op} kernels; regenerate with tools/make_kernels_cpu_hip.py.\n"
              f"# source_sha256: {hashlib.sha256(raw).hexdigest()}\n")
    body = text.replace(ANCHOR, ANCHOR + shim_for(text), 1)
    for name in MISSING_ON_HIP:
        body = re.sub(rf"\blibdevice\.{name}\(", f"_triton_{name}(", body)
    left = re.findall(r"libdevice\.(?:add_rn|mul_rn|sub_rn|div_rn|rint)", body)
    if left:
        sys.exit(f"unreplaced libdevice calls remain: {left[:3]}")
    if op == "resize":
        fast = os.path.join(KERNELS, "resize_hip_fast.py")
        body += "\n\n# HIP-specific complete-window area kernels.\n" + open(fast, encoding="utf-8").read()
    elif op == "cvtcolor":
        fast = os.path.join(KERNELS, "cvtcolor_hip_fast.py")
        body += "\n\n# HIP-specific aligned colour kernels.\n" + open(fast, encoding="utf-8").read()
    return header + body


def main():
    ops = [op for op in OPS if needs_derivation(op)]
    if "--check" in sys.argv[1:]:
        stale = []
        for op in ops:
            dst = paths(op)[1]
            have = open(dst, "rb").read() if os.path.exists(dst) else None
            if have != derive(op).encode("utf-8"):
                stale.append(op)
                print(f"kernels/{op}_cpu_hip.py is stale or missing; run tools/make_kernels_cpu_hip.py",
                      file=sys.stderr)
            else:
                print(f"kernels/{op}_cpu_hip.py is current")
        return 1 if stale else 0
    for op in ops:
        dst = paths(op)[1]
        with open(dst, "w", encoding="utf-8") as f:
            f.write(derive(op))
        print(f"wrote {os.path.normpath(dst)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
