# Triton-backed GPU operations

Run `resize`, `cvtColor` and `warpAffine` on GPU through familiar OpenCV-style
C++ and Python calls. This optional, default-off module compiles Triton kernels
on first use; warmed calls execute from C++ without embedding Python. PyTorch
is not required. Calls use `cv::triton::*` explicitly.

## Quickstart

With a supported GPU, toolkit and installed module, the first call is:

```python
import cv2 as cv
import numpy as np

assert cv.triton.haveBackend()
image = np.full((96, 128, 3), 127, dtype=np.uint8)
src = cv.cuda_GpuMat(image)
stream = cv.cuda_Stream()
dst = cv.triton.resize(src, (64, 48), interpolation=cv.INTER_LINEAR, stream=stream)
stream.waitForCompletion()
result = dst.download()
print(result.shape)  # (48, 64, 3)
```

Keep `src` and `dst` alive until the stream completes. This CUDA example needs
OpenCV built with this module; ordinary `cv.resize` calls keep their existing behavior.

To build and install, follow [CUDA setup](tutorials/cuda_setup.markdown)
or [HIP setup](tutorials/hip_setup.markdown). Enable
`BUILD_opencv_triton=ON` and select the intended backend. Those guides own the
complete dependency, CMake and installation commands. A first-call walkthrough
assumes the environment is ready; a source build takes additional time.

| First-use problem | What to check |
|---|---|
| `cv.triton` is missing | Check `cv2.__file__` and the native extension; select the installation built with Triton and Python bindings. |
| `toolchain_unavailable` | A cache miss needs the selected Python interpreter with Triton 3.6.0; check [compiler selection](doc/maintenance.markdown#runtime-toolchain). |
| `depth_or_channels` or `interpolation` | Check the operation's supported type, channel count, mode or conversion code below. Unsupported requests raise, with no automatic fallback. |

See [error messages](doc/maintenance.markdown#error-messages) for layout, device,
cache and compilation errors.

## Capabilities and interfaces

| Operation | Depths | Channels | Modes / reference |
|---|---|---|---|
| [resize](doc/resize.markdown) | 8U, 32F | 1, 3, 4 | NEAREST, LINEAR, CUBIC, AREA, same-size copy; fixed generic CPU algorithm |
| [cvtColor](doc/cvtcolor.markdown) | 8U; selected 16U, 32F | determined by code | Channel/gray, YUV/YCrCb, HSV, packed/Bayer and other conversions; linked CPU reference |
| [warpAffine](doc/warpaffine.markdown) | 8U, 16U, 32F | 1, 3, 4 | NEAREST, LINEAR, CUBIC; AREA aliases LINEAR; selected CPU arithmetic |

The [conversion catalogue](doc/cvtcolor.markdown#conversion-codes) lists the
84 distinct numeric codes at 8U and the 20 supported codes at each of 16U and
32F, grouped by family. Enum aliases share a numeric code.

Pitched images, source and destination ROIs, and output reuse are supported.
Source and destination pixel storage must not overlap. GpuMat entries allocate
or reuse output; `DeviceImage` entries require caller-allocated output.
The operator guides define border modes, shape rules and numerical tolerances.
The module targets selected CPU behavior to make compatibility expectations
explicit; this does not imply equality with every CPU build or `cv::cuda`.

| Backend | Public interface | Toolchain / target examples | Validation scope |
|---|---|---|---|
| CUDA | C++ GpuMat and DeviceImage; Python GpuMat | CUDA 12.8, Triton 3.6.0; H20 `sm_90` | H20 is a documented configuration. Validate the installed build on the selected machine. |
| HIP | C++ DeviceImage; Python queries and safe GpuMat refusal | DTK 26.04 / BW1000 `gfx936`; ROCm 7.2.4 / W7900D `gfx1100` | Separate SDK/architecture configurations; results on one do not validate the other. |

`CUDA` and `HIP` select runtime backends; DTK and ROCm are HIP toolchains, and
`sm_90`/`gfx*` identify compiler targets. These are supported interface/configuration
examples, not a claim that every listed machine or revision has passed tests.
See the [terms](doc/maintenance.markdown#terms) and backend tutorials for details.
HIP GpuMat numerical calls raise catchable `StsNotImplemented`; no HIP Python
device-buffer wrapper is provided.

## Where to go next

| Reader task | Start here |
|---|---|
| Build, install and run | [CUDA tutorial](tutorials/cuda_setup.markdown), [HIP tutorial](tutorials/hip_setup.markdown) |
| Select options and compare numbers | [resize](doc/resize.markdown), [cvtColor](doc/cvtcolor.markdown), [warpAffine](doc/warpaffine.markdown) |
| Use C++ signatures | [GpuMat API](include/opencv2/triton.hpp), [DeviceImage API](include/opencv2/triton/device.hpp) |
| Understand source, JIT/cache and resource lifetime | [maintenance guide](doc/maintenance.markdown) |
| Run accuracy/performance checks | [validation guide](doc/validation.markdown) |

Buildable examples are [resize](samples/resize.cpp), [cvtColor](samples/cvtcolor.cpp),
[warpAffine](samples/warpaffine.cpp) and the [HIP device-buffer sample](samples/hip/device.cpp).
Enable `BUILD_EXAMPLES`, `BUILD_TESTS` and `BUILD_PERF_TESTS` as needed.
`opencv_test_triton` runs accuracy/runtime checks; `opencv_perf_triton` measures
warmed public calls with stream completion. Native CUDA comparisons are off by
default. To include them, set `TRITON_PERF_WITH_NATIVE_BASELINE=ON` and build
`cudawarping` and `cudaimgproc`; these are perf-only dependencies. See the
[performance test commands](doc/validation.markdown#performance-tests).
The [HIP graph helper](doc/hip_graph.markdown) measures replay separately.
