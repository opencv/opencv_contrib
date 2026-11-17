Triton cvtColor {#tutorial_triton_cvtcolor}
===============

The optional Triton module provides explicit `cv::triton::cvtColor` calls.
CUDA callers pass `GpuMat`; HIP callers pass caller-owned `DeviceImage` views.
The table below lists every supported conversion. The native
code table contains 84 distinct numeric codes at CV_8U, 20 at CV_16U and 20 at
CV_32F. Enum aliases share a row. A supported code at one depth does not imply
support at another depth; unsupported combinations raise without CPU fallback.

The code fixes the source and destination channel counts. There is no dstCn
override. Channel permutations, colour/gray, YUV/YCrCb, forward HSV, packed
565/555, bilinear Bayer, XYZ/Lab/Luv and video decode are covered as listed.
Reverse HSV/HLS, forward linear Luv, Bayer VNG/EA and video encoding are refused.

Packed 565/555 use CV_8UC2. Interleaved UYVY/YUY2/YVYU conversion requires even
width; the two Y-channel extraction codes permit odd width. NV12/NV21 take a
single row-pitched CV_8UC1 image with an even width and a height divisible by 3:
its first H rows are luma, followed by H/2 rows of interleaved chroma; output is
H by W. ROI pitch is respected for both planes. Invalid geometry and offsets
outside the runtime integer indexing range are refused.

Calling the function
--------------------

The following CUDA example converts a host BGR image to device-resident gray:

```cpp
#include <opencv2/triton.hpp>

cv::cuda::GpuMat src(host), gray;  // host is a CV_8UC3 cv::Mat
cv::cuda::Stream stream;
cv::triton::cvtColor(src, gray, cv::COLOR_BGR2GRAY, stream);
stream.waitForCompletion();
```

The destination is allocated or reused. Calls are asynchronous; retain both
allocations until the stream completes. HIP uses the `DeviceImage` overload
from `opencv2/triton/device.hpp`, with preallocated output and a native stream
handle. The complete allocation/upload/download example is
`samples/hip/device.cpp` (`example_triton_device cvtcolor`).

<a id="numerical-behavior"></a>
<a id="colour-numerics"></a>
Numerical behavior
------------------

The module compares colour conversions with the linked OpenCV CPU implementation
so users have an explicit compatibility target for existing CPU pipelines.
CPU dispatch, IPP or a custom HAL can change the reference arithmetic.
Integer conversions and float channel copies require exact numerical equality;
this is distinct from the tolerance used for floating-point colour arithmetic.

For finite float gray/YUV/YCrCb calculations, the test allows a small rounding
error **per output element**, scaled by that pixel's input and reference values:

`abs(got - ref) <= 8 * 2^-24 * max(1, abs(ref), input_scale)`.

Here `got` and `ref` are one GPU and CPU output component, and `input_scale` is
the largest absolute input component at the same pixel. All values use the
conversion's channel units; the bound is an absolute channel-value difference,
not a percentage or a whole-image maximum. With values bounded by 1, the bound
is about `9.54e-7`. A distant bright pixel does not enlarge it. This accounts for
float32 operation grouping; it is not a universal error proof for overflow,
subnormals or arbitrary arithmetic environments.

A CPU arithmetic **profile** records an identified evaluation pattern. Fixed
probes of `cv::cvtColor` check whether multiplication/addition round separately
and rows use groups of four values followed by scalar tails. When that pattern
is identified, the GPU follows it and the accuracy test also imposes an absolute
`1e-4` ceiling: **both** bounds must hold. See
`float_cpu_separate4` in `src/runtime/cvtcolor.cpp` and
`meetsCvtContract` in `validation/reference.hpp`.

If identification fails, controls change during probing, or rounding is not the
supported default, `float_cpu_separate4()` returns false. The GPU then uses its
fused multiply-add path; tests apply the scaled bound without the extra `1e-4`
cap. “Unidentified” does not mean the CPU was proved to use FMA, or that exact
agreement with that build was established. The process-local result is keyed
by optimization, IPP/exactness and rounding controls. These bounded probes do
not compute user images on CPU. Copy paths retain exact comparison.

Directed fixtures include unit/255 scales and mixed signed values through
`2*65535`, short rows, vector-boundary widths and pitched ROIs. Their results
apply to the tested build and inputs.

Bindings, streams and checks
----------------------------

CUDA provides GpuMat Python bindings. The HIP `DeviceImage` overload is C++ only.
All three Python GpuMat adapters (resize, cvtColor and warpAffine) return
catchable `StsNotImplemented` on HIP, including default/`None` streams.

Callers retain images and order producers and consumers; the runtime retains
its modules and lookup tables. See [resource lifetime][resource-cleanup].

Warm operations support graph capture. Follow the shared [Graph resources][graph-release]
contract before releasing capture holds; this includes completion and coordination
across all participating threads and contexts.

Accuracy tests cover every admitted code/depth pair, boundary colours, short
axes and SIMD/block-boundary widths, pitched ROIs, output guards, refusals and
lookup-table stream/capture lifetime. CUDA Python numerical tests require an
actual CUDA build; HIP binding tests check the documented safe refusals.
OpenCV performance tests provide representative family/depth cases with warmup;
compilation and allocation are excluded. Public-call wall time includes host
submission and synchronization; it is not isolated kernel or graph time.


<a id="conversion-codes"></a>
Supported conversion codes
--------------------------

Each row names one numeric code. OpenCV enum aliases with the same value are
also accepted; aliases do not add conversions. Depth support is specific to each row.

### Channel order and grayscale

| Conversion code | Source → output channels | Depths |
|---|---|---|
| `COLOR_BGR2BGRA` | 3 → 4 | 8U, 32F |
| `COLOR_BGRA2BGR` | 4 → 3 | 8U, 32F |
| `COLOR_BGR2RGBA` | 3 → 4 | 8U, 32F |
| `COLOR_RGBA2BGR` | 4 → 3 | 8U, 32F |
| `COLOR_BGR2RGB` | 3 → 3 | 8U, 32F |
| `COLOR_BGRA2RGBA` | 4 → 4 | 8U, 32F |
| `COLOR_GRAY2BGR` | 1 → 3 | 8U, 16U, 32F |
| `COLOR_GRAY2RGBA` | 1 → 4 | 8U, 16U, 32F |
| `COLOR_BGR2GRAY` | 3 → 1 | 8U, 16U, 32F |
| `COLOR_RGB2GRAY` | 3 → 1 | 8U, 16U, 32F |
| `COLOR_BGRA2GRAY` | 4 → 1 | 8U, 16U, 32F |
| `COLOR_RGBA2GRAY` | 4 → 1 | 8U, 16U, 32F |

### Video decode and luma extraction

| Conversion code | Source → output channels | Depths |
|---|---|---|
| `COLOR_YUV2GRAY_UYVY` | 2 → 1 | 8U |
| `COLOR_YUV2GRAY_YUY2` | 2 → 1 | 8U |
| `COLOR_YUV2BGR_NV12` | 1 → 3 | 8U |
| `COLOR_YUV2RGB_NV12` | 1 → 3 | 8U |
| `COLOR_YUV2BGRA_NV12` | 1 → 4 | 8U |
| `COLOR_YUV2RGBA_NV12` | 1 → 4 | 8U |
| `COLOR_YUV2BGR_NV21` | 1 → 3 | 8U |
| `COLOR_YUV2RGB_NV21` | 1 → 3 | 8U |
| `COLOR_YUV2BGRA_NV21` | 1 → 4 | 8U |
| `COLOR_YUV2RGBA_NV21` | 1 → 4 | 8U |
| `COLOR_YUV2RGB_UYVY` | 2 → 3 | 8U |
| `COLOR_YUV2BGR_UYVY` | 2 → 3 | 8U |
| `COLOR_YUV2RGB_YUY2` | 2 → 3 | 8U |
| `COLOR_YUV2BGR_YUY2` | 2 → 3 | 8U |
| `COLOR_YUV2RGB_YVYU` | 2 → 3 | 8U |
| `COLOR_YUV2BGR_YVYU` | 2 → 3 | 8U |

### YUV and YCrCb

| Conversion code | Source → output channels | Depths |
|---|---|---|
| `COLOR_BGR2YCrCb` | 3 → 3 | 8U, 16U, 32F |
| `COLOR_RGB2YCrCb` | 3 → 3 | 8U, 16U, 32F |
| `COLOR_YCrCb2BGR` | 3 → 3 | 8U, 16U, 32F |
| `COLOR_YCrCb2RGB` | 3 → 3 | 8U, 16U, 32F |
| `COLOR_BGR2YUV` | 3 → 3 | 8U, 16U, 32F |
| `COLOR_RGB2YUV` | 3 → 3 | 8U, 16U, 32F |
| `COLOR_YUV2BGR` | 3 → 3 | 8U, 16U, 32F |
| `COLOR_YUV2RGB` | 3 → 3 | 8U, 16U, 32F |

### Forward HSV

| Conversion code | Source → output channels | Depths |
|---|---|---|
| `COLOR_BGR2HSV` | 3 → 3 | 8U |
| `COLOR_RGB2HSV` | 3 → 3 | 8U |
| `COLOR_BGR2HSV_FULL` | 3 → 3 | 8U |
| `COLOR_RGB2HSV_FULL` | 3 → 3 | 8U |

### Packed BGR565 and BGR555

| Conversion code | Source → output channels | Depths |
|---|---|---|
| `COLOR_BGR2BGR565` | 3 → 2 | 8U |
| `COLOR_RGB2BGR565` | 3 → 2 | 8U |
| `COLOR_BGRA2BGR565` | 4 → 2 | 8U |
| `COLOR_RGBA2BGR565` | 4 → 2 | 8U |
| `COLOR_GRAY2BGR565` | 1 → 2 | 8U |
| `COLOR_BGR5652BGR` | 2 → 3 | 8U |
| `COLOR_BGR5652RGB` | 2 → 3 | 8U |
| `COLOR_BGR5652BGRA` | 2 → 4 | 8U |
| `COLOR_BGR5652RGBA` | 2 → 4 | 8U |
| `COLOR_BGR5652GRAY` | 2 → 1 | 8U |
| `COLOR_BGR2BGR555` | 3 → 2 | 8U |
| `COLOR_RGB2BGR555` | 3 → 2 | 8U |
| `COLOR_BGRA2BGR555` | 4 → 2 | 8U |
| `COLOR_RGBA2BGR555` | 4 → 2 | 8U |
| `COLOR_GRAY2BGR555` | 1 → 2 | 8U |
| `COLOR_BGR5552BGR` | 2 → 3 | 8U |
| `COLOR_BGR5552RGB` | 2 → 3 | 8U |
| `COLOR_BGR5552BGRA` | 2 → 4 | 8U |
| `COLOR_BGR5552RGBA` | 2 → 4 | 8U |
| `COLOR_BGR5552GRAY` | 2 → 1 | 8U |

### Bilinear Bayer

| Conversion code | Source → output channels | Depths |
|---|---|---|
| `COLOR_BayerBG2BGR` | 1 → 3 | 8U, 16U |
| `COLOR_BayerGB2BGR` | 1 → 3 | 8U, 16U |
| `COLOR_BayerRG2BGR` | 1 → 3 | 8U, 16U |
| `COLOR_BayerGR2BGR` | 1 → 3 | 8U, 16U |
| `COLOR_BayerBG2GRAY` | 1 → 1 | 8U, 16U |
| `COLOR_BayerBG2BGRA` | 1 → 4 | 8U, 16U |

### XYZ, Lab and Luv

| Conversion code | Source → output channels | Depths |
|---|---|---|
| `COLOR_BGR2XYZ` | 3 → 3 | 8U |
| `COLOR_RGB2XYZ` | 3 → 3 | 8U |
| `COLOR_XYZ2BGR` | 3 → 3 | 8U |
| `COLOR_XYZ2RGB` | 3 → 3 | 8U |
| `COLOR_BGR2Lab` | 3 → 3 | 8U |
| `COLOR_RGB2Lab` | 3 → 3 | 8U |
| `COLOR_Lab2BGR` | 3 → 3 | 8U |
| `COLOR_Lab2RGB` | 3 → 3 | 8U |
| `COLOR_BGR2Luv` | 3 → 3 | 8U |
| `COLOR_RGB2Luv` | 3 → 3 | 8U |
| `COLOR_Luv2BGR` | 3 → 3 | 8U |
| `COLOR_Luv2RGB` | 3 → 3 | 8U |
| `COLOR_LBGR2Lab` | 3 → 3 | 8U |
| `COLOR_LRGB2Lab` | 3 → 3 | 8U |
| `COLOR_Lab2LBGR` | 3 → 3 | 8U |
| `COLOR_Lab2LRGB` | 3 → 3 | 8U |
| `COLOR_Luv2LBGR` | 3 → 3 | 8U |
| `COLOR_Luv2LRGB` | 3 → 3 | 8U |

<!-- Link targets: GitHub source paths; Doxygen page labels in the hidden block. -->
<!--
@cond
-->
[resource-cleanup]: maintenance.markdown#cleanup
[graph-release]: maintenance.markdown#graph-release
<!--
@endcond
[resource-cleanup]: @ref cleanup
[graph-release]: @ref triton_graph_release
@cond
-->
<!-- @endcond -->
