Triton resize {#tutorial_triton_resize}
=============

[API reference][triton-api].
[CUDA setup][cuda-setup].
[HIP setup][hip-setup].

Supported inputs
----------------

`cv::triton::resize` supports `CV_8U` and `CV_32F`, with 1, 3 or 4 channels.
CUDA callers pass `GpuMat`; HIP callers pass caller-owned `DeviceImage` views.
Interpolation modes are `INTER_NEAREST`, `INTER_LINEAR`, `INTER_CUBIC` and
`INTER_AREA`. AREA supports integer and fractional ratios and upscaled axes.
Same-size requests copy the image without interpolation. Source and destination
may be pitched ROIs with contiguous pixels and channels; their pixel storage must
not overlap. Separate ROIs may share a parent allocation.
Correctly sized GpuMat output is reused; DeviceImage output must be preallocated.

An explicit positive `dsize` takes precedence over `fx` and `fy`. If `dsize` is
empty, both scales must be finite and positive and the rounded dimensions must
fit positive int32 values. The original scales are retained for coordinate
mapping. Sizing and interpolation are checked before output allocation.
`INTER_NEAREST_EXACT`, `INTER_LINEAR_EXACT`, warp flags, other depths and other
channel counts are unsupported. There is no CPU or `cv::cuda` fallback.

Numerical behavior
------------------

CPU compatibility gives callers a stated result when moving an existing CPU
image-processing step to GPU. GPU libraries can use different coordinates and
rounding rules, so the module names a specific reference rather than treating
all implementations as interchangeable.

The reference is the generic OpenCV 5.0.0 CPU resize algorithm with 128-bit SIMD,
without FMA, IPP or HAL substitutions. `CV_8U` output is bit-exact against this
reference. `CV_32F` accuracy checks use the fixed absolute bounds in
[numerical validation criteria][resize-numerics] (`validation/numerics.hpp`).
Those bounds were calibrated on a finite set with `abs(input) <= 2`; they are not
a universal error proof for other amplitudes or arithmetic environments. This is
a test-evidence boundary, not a new production API input-amplitude restriction.
Tests use a portable CPU
implementation in the test sources so that host-library dispatch choices do
not change their expected output.

`INTER_LINEAR` at exact 2× downsampling uses the CPU AREA route. Other CPU
configurations and `cv::cuda::resize` may differ in coordinate mapping,
fixed-point rounding or route selection. Native CUDA timings therefore do not
establish numerical equivalence.

Calling the function
--------------------

The following CUDA example resizes a host image to half its dimensions:

```cpp
#include <opencv2/triton.hpp>

cv::cuda::GpuMat src(host), dst;  // host is a cv::Mat
cv::cuda::Stream stream;  // Optional: omitting it uses the default CUDA stream.
cv::triton::resize(src, dst, cv::Size(host.cols / 2, host.rows / 2),
                   0, 0, cv::INTER_LINEAR, stream);
stream.waitForCompletion();
```

The CUDA stream argument is optional; omitting it uses
`cv::cuda::Stream::Null()`. Pass an explicit stream to order this resize with
other asynchronous work. Retain input and output allocations until the selected
stream completes. HIP applications use the `DeviceImage` overloads of
`cv::triton::resize` from `opencv2/triton/device.hpp` with preallocated output
and a native stream handle. See `samples/hip/device.cpp`
(`example_triton_device resize`) for allocation and data transfers.
The overload with `fx`/`fy` supports scale-derived sizing; the six-argument
form requires explicit `dsize`.

Runtime and deployment
----------------------

See [runtime compilation and cache][runtime-toolchain] for compiler interpreters, launch ABI, cache
keys, persisted artifacts and resource relocation. For graph capture and
asynchronous resource ownership, see [architecture and ownership][runtime-architecture] and
cv::triton::releaseGraphResources().

<!-- Link targets: GitHub source paths; Doxygen page labels in the hidden block. -->
<!--
@cond
-->
[resize-numerics]: ../validation/numerics.hpp
[triton-api]: ../include/opencv2/triton.hpp
[cuda-setup]: ../tutorials/cuda_setup.markdown
[hip-setup]: ../tutorials/hip_setup.markdown
[runtime-toolchain]: maintenance.markdown#runtime-toolchain
[runtime-architecture]: maintenance.markdown#architecture
<!--
@endcond
[resize-numerics]: @ref tutorial_triton_validation
[triton-api]: @ref triton
[cuda-setup]: @ref tutorial_triton_cuda_first_use
[hip-setup]: @ref tutorial_triton_hip_first_use
[runtime-toolchain]: @ref triton_runtime_toolchain
[runtime-architecture]: @ref triton_architecture
@cond
-->
<!-- @endcond -->
