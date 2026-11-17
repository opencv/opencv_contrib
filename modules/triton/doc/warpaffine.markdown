Triton warpAffine {#tutorial_triton_warpaffine}
=================

[API reference][triton-api].
[Runtime compilation and cache][runtime-toolchain].
[Streams and resource ownership][runtime-architecture].

Supported inputs
----------------

CUDA callers pass `GpuMat`; HIP callers pass caller-owned `DeviceImage` views.

| Parameter | Supported values |
|---|---|
| Depth | `CV_8U`, `CV_16U`, `CV_32F` |
| Channels | 1, 3, 4 |
| Interpolation | `INTER_NEAREST`, `INTER_LINEAR`, `INTER_CUBIC`; `INTER_AREA` aliases LINEAR |
| Border | CONSTANT, REPLICATE, REFLECT, REFLECT_101, WRAP; TRANSPARENT for NEAREST/LINEAR/AREA |
| Matrix | 2×3 `CV_32F` or `CV_64F`; forward map or `WARP_INVERSE_MAP` |
| Layout | Pitched source/destination ROIs, output reuse, nonoverlapping storage |

For the GpuMat overload, an empty destination size means the source size.
DeviceImage callers preallocate the destination, whose dimensions determine
the output size. A forward matrix is inverted internally.
Integer border components are rounded and saturated to the output
depth; float borders preserve fractional values. `Scalar(v)` specifies
`(v, 0, 0, 0)`, rather than applying `v` to every channel.

LANCZOS4, CUBIC with TRANSPARENT, other depths and channel counts, unknown flags,
and overlapping storage are unsupported. Nonfinite matrices or border values
are rejected. The planner bounds affine terms over the destination rectangle
to avoid float32 intermediate overflow. NEAREST/LINEAR/AREA must fit the safe
int32 sampling range; CUBIC clamps finite coordinates before integer conversion.
The conservative affine bound may reject large terms that would cancel. For
example, for a destination-to-source row `sx = 1e10*x - 1e10*y`, the terms
cancel when x = y. A 2 x 2 destination is still rejected for NEAREST/LINEAR/AREA:
the absolute-term bound is `2e10`, above the safe int32 range, and the off-diagonal
pixels do not benefit from that cancellation. The check covers the whole
destination rectangle, not only pixels where terms cancel.

Calling the function
--------------------

The following CUDA example translates a host image:

```cpp
#include <opencv2/triton.hpp>

cv::cuda::GpuMat src(host), dst;  // host is a cv::Mat
cv::cuda::Stream stream;  // Optional: omitting it uses the default CUDA stream.
const cv::Matx23d matrix(1, 0, 8, 0, 1, 4);
cv::triton::warpAffine(src, dst, matrix, host.size(), cv::INTER_LINEAR,
                       cv::BORDER_REPLICATE, cv::Scalar(), stream);
stream.waitForCompletion();
```

The destination is allocated or reused. The CUDA stream argument is optional
and defaults to `cv::cuda::Stream::Null()`; pass one to order the transform with
other asynchronous work. Calls are asynchronous; retain both allocations until
the selected stream completes. HIP uses the `DeviceImage` overload in
`opencv2/triton/device.hpp`, with preallocated output, six row-major matrix
coefficients and a native stream handle. See `samples/hip/device.cpp`
(`example_triton_device warp`) for allocation and data transfers.

Transparent borders
-------------------

Provide an initialized destination of the requested size and type.
An allocation made by the call has no defined background; `borderValue` does not
initialize it. NEAREST leaves a destination pixel unchanged when its rounded
source coordinate is outside. LINEAR/AREA replace each out-of-range tap with
that pixel's previous destination value, retaining the original weights. A
wholly outside neighborhood leaves the pixel unchanged. Weights are not
renormalized. AREA uses LINEAR sampling here, rather than resize-style area
integration.

<a id="cpu-reference-and-accuracy"></a>
<a id="warp-numerics"></a>
CPU reference and accuracy
--------------------------

A CPU arithmetic **profile** describes how this build rounds coordinate and
interpolation expressions, including fused multiply-add and vector/tail grouping.
It is detected from small fixed inputs, not from the GPU model. This selects the
CPU behavior the GPU emulates; caller images remain on the GPU.

NEAREST/LINEAR/AREA follow the linked OpenCV CPU implementation's vector-path
arithmetic. In that implementation the SIMD body evaluates a coordinate as
`v_fma(M0, x, float(y*M1 + M2))`, while scalar tails can use separate operations.
The expressions may round differently at sampling boundaries. Since the scalar
tail length depends on destination width, ordinary CPU calls at different
widths can disagree in their shared region.

The accuracy reference rounds destination width up to a multiple of 128 and
crops the result to place compared columns in the vector body. For transparent
borders the reference destination is initialized with the same contents as the
GPU destination. NEAREST can select an adjacent source pixel at a rounding
boundary, so exact agreement with an arbitrary natural-width CPU call is not
guaranteed. LINEAR/AREA can also differ at those boundaries.

The runtime identifies fused versus separate multiply/add with fixed probes of
the linked CPU implementation, for each depth, channel count and
`cv::useOptimized()` state. Hardware FMA support alone does not identify the
compiled dispatch path. This bounded first-use calibration does not compute
the caller's output on the CPU; an unidentified arithmetic path is rejected.

CUBIC uses the natural-width CPU reference. Its CPU path evaluates coordinates
from the absolute destination column without the same body/tail split. It uses
4×4 taps, coefficient `A = -0.75`, and a final saturating conversion for integer
output. This reference distinction is checked in the accuracy tests.

| Depth | Error allowed per element |
|---|---|
| `CV_8U` | Exact equality |
| `CV_16U` | `abs(got - ref) <= 1` |
| `CV_32F` | `abs(got - ref) <= min(5e-5 * max(abs(ref), 1.0), 1e-4)` |

The float bound has a hard absolute ceiling of `1e-4` and is not scaled by the
image maximum. These bounds apply against the selected reference.
`cv::cuda::warpAffine` uses different sampling arithmetic and is an optional
performance comparison, rather than the correctness reference.

Execution
---------

Shared parameter validation resolves output metadata before allocation. After
allocation, runtime validation checks the actual layout, overlap and device
ownership. Matrix coefficients and border values are runtime arguments, so
changing them does not itself compile another kernel. Arithmetic mode and
transparent-border mode participate in the compile key.

CUDA uses `1×256` tiles for NEAREST and `2×64` for LINEAR/CUBIC, including AREA.
HIP uses backend-specific launch shapes with small/narrow-image and C3 NEAREST
conditions. These choices do not change sampling semantics.

CUDA provides GpuMat Python bindings. The HIP `DeviceImage` overload is C++ only;
GpuMat calls on HIP raise a catchable `StsNotImplemented`. For captured calls,
follow the graph-resource lifetime rules in cv::triton::releaseGraphResources().

Performance tests warm the kernels and include public submission plus stream
completion. The AREA/TRANSPARENT cases initialize and reuse destination storage;
reading previous destination values is part of their workload. They do not
assume that native CUDA transparent-border behavior is equivalent.

<!-- Link targets: GitHub source paths; Doxygen page labels in the hidden block. -->
<!--
@cond
-->
[triton-api]: ../include/opencv2/triton.hpp
[runtime-toolchain]: maintenance.markdown#runtime-toolchain
[runtime-architecture]: maintenance.markdown#architecture
<!--
@endcond
[triton-api]: @ref triton
[runtime-toolchain]: @ref triton_runtime_toolchain
[runtime-architecture]: @ref triton_architecture
@cond
-->
<!-- @endcond -->
