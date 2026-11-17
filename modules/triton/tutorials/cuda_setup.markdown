Building and using the CUDA backend {#tutorial_triton_cuda_first_use}
===================================

Build the CUDA backend, run the public GpuMat entries and check the installation.
Choose a CUDA toolkit, driver and Triton compiler that support the actual device.
PyTorch is not required. See @ref triton_runtime_toolchain for compiler selection,
ABI, cache and resource-location rules.

@anchor tutorial_triton_h20_first_use

A configuration example is Linux x86_64, NVIDIA H20 (`sm_90`), CUDA Toolkit 12.8
and Triton 3.6.0. These values describe a build configuration, not a test result
for every machine or source revision. Follow the
[OpenCV Linux installation guide](https://docs.opencv.org/5.x/d7/d9f/tutorial_linux_install.html)
for general prerequisites. Record the actual driver and loaded libraries when
validating a build.

Configure and build
-------------------

Choose matching OpenCV and contrib sources. Keep build and install directories
outside the source tree. A Python environment with NumPy and matching Python
development headers is needed for bindings. The compiler interpreter can be
the same interpreter or another interpreter that imports Triton 3.6.0.

```sh
python3 -m venv <venv>
<venv>/bin/python -m pip install numpy triton==3.6.0
cmake -S <opencv> -B <build> -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<install> \
  -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules \
  -DBUILD_LIST=core,imgproc,imgcodecs,cudev,triton,python3,python_bindings_generator,ts \
  -DBUILD_opencv_triton=ON -DTRITON_BACKEND=cuda -DWITH_CUDA=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=<cuda-toolkit> \
  -DCUDA_ARCH_BIN=<compute-capability> -DCUDA_ARCH_PTX= \
  -DPYTHON3_EXECUTABLE=<venv>/bin/python \
  -DTRITON_PYTHON_EXECUTABLE=<venv>/bin/python \
  -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON -DBUILD_PERF_TESTS=ON \
  -DINSTALL_TESTS=ON
cmake --build <build> --parallel
cmake --install <build>
```

For the H20 example, use CUDA 12.8 and `CUDA_ARCH_BIN=9.0`.
Check the CMake summary for the selected CUDA toolkit, device architecture,
Python headers/library and module list. `TRITON_PYTHONPATH` is optional if the
compiler interpreter already imports the intended Triton package. Kernels are
compiled on first use, rather than during the OpenCV build.

Native CUDA performance comparisons are disabled by default. To include them,
set `TRITON_PERF_WITH_NATIVE_BASELINE=ON` with `BUILD_PERF_TESTS=ON`, and include
both `cudawarping` and `cudaimgproc` in `BUILD_LIST`. They are perf-only, not
production dependencies. With baselines enabled, an unfiltered
`opencv_perf_triton` run executes both sets. The performance-test section in
`doc/validation.markdown` lists the gtest filters for selecting one set.

`cudev` is required by OpenCV's CUDA build. It provides CUDA template support
and is unrelated to cuBLAS or cuFFT. `BUILD_LIST` is an allowlist: setting
`BUILD_opencv_cudawarping=ON` or `BUILD_opencv_cudaimgproc=ON` alone does not
select a module excluded from that list. Confirm the final `To be built` list
and the registered native perf tests before claiming a comparison.

For a minimal toolkit without cuBLAS/cuFFT development headers, these three
operations and their CUDA timing comparisons can be built with
`-DWITH_CUBLAS=OFF -DWITH_CUFFT=OFF`. Keep `cudev` enabled. These switches disable
BLAS/FFT functionality elsewhere; they do not disable NPP or Triton.

If the Python summary has `Libraries: NO`, no NumPy path, or `python3` is absent
from `To be built`, stop before building. Install matching Python development
files, or pass the selected interpreter's paths explicitly:

```sh
<venv>/bin/python -c 'import sysconfig, numpy; print(sysconfig.get_path("include")); print(sysconfig.get_config_var("LIBDIR") + "/" + sysconfig.get_config_var("LDLIBRARY")); print(numpy.get_include())'
```

Reconfigure with `-DPYTHON3_INCLUDE_DIR=<include>`,
`-DPYTHON3_LIBRARY=<library>` and `-DPYTHON3_NUMPY_INCLUDE_DIRS=<numpy-include>`.
If Python detection was cached before its dependencies were installed, clear
only those cached results using `cmake -U 'PYTHON3*' -S OPENCV_SOURCE -B BUILD_DIR`
**together with all the Python selection arguments above**, then inspect the
summary again. Do not substitute another installation's `cv2` for missing bindings.

Use the installed module
------------------------

Select the installed OpenCV library and Python paths for the local installation
layout. For the layout shown by CMake above, for example:

```sh
export CUDA_VISIBLE_DEVICES=<available-physical-device>
export LD_LIBRARY_PATH=<install>/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=<install>/lib/python3.12/site-packages:${PYTHONPATH:-}
export OPENCV_TRITON_CACHE_DIR=<writable-cache>
```

Use your interpreter's actual `python3.X` directory, not another Python version.
Repeat these exports in a new terminal. For an installation check, leave
`OPENCV_TRITON_RESOURCES` unset and use the installed `cv2`. Check
`cv2.__file__`, the loaded native extension path and
`cv2.getBuildInformation()` to avoid importing another OpenCV installation.
When using OpenCV's package loader, `cv2._native.__file__` identifies the native
extension as well as `cv2.__file__` identifying the loader.

The examples use public GpuMat calls and explicit streams:

```sh
<build>/bin/example_triton_resize
<build>/bin/example_triton_cvtcolor
<build>/bin/example_triton_warpaffine
```

A Python pipeline can use the same operations:

```python
import cv2 as cv
import numpy as np

assert cv.triton.haveBackend()
stream = cv.cuda_Stream()
image = np.random.default_rng(7).integers(0, 256, (96, 128, 3), dtype=np.uint8)
src = cv.cuda_GpuMat()
src.upload(image, stream)
small = cv.triton.resize(src, (64, 48), interpolation=cv.INTER_LINEAR, stream=stream)
gray = cv.triton.cvtColor(small, cv.COLOR_BGR2GRAY, stream=stream)
matrix = np.eye(2, 3, dtype=np.float64)
warped = cv.triton.warpAffine(gray, matrix, (64, 48),
                             flags=cv.INTER_NEAREST, stream=stream)
stream.waitForCompletion()
result = warped.download()
```

Retain inputs and outputs until stream completion. For transparent borders,
initialize the destination before calling warpAffine. See
@ref tutorial_triton_warpaffine for reference arithmetic and accuracy bounds.

Tests and performance
---------------------

```sh
<build>/bin/opencv_test_triton --gtest_list_tests
<build>/bin/opencv_test_triton --gtest_output=xml:accuracy.xml
<venv>/bin/python <opencv_contrib>/modules/triton/misc/python/test/test_triton.py -v
<build>/bin/opencv_perf_triton --gtest_list_tests
```

The Python tests use OpenCV's test helpers; add
`<opencv>/modules/python/test` to `PYTHONPATH` as needed. Some CPU numerical
checks intentionally skip when the linked OpenCV CPU build is not their fixed
reference configuration. Review the skip reason and keep the same installed
`cv2`. Standard OpenCV list/filter/XML options select test cases. Review actual failures and skips,
including tests requiring a second device. To check perf wiring without running
the entire performance matrix, run one representative case from each operation:

```sh
<build>/bin/opencv_perf_triton \
  --gtest_filter='*TritonResize.TritonResize/0:*TritonCvtColor.TritonCvtColor/0:*TritonWarpAffine.TritonWarpAffine/0' \
  --perf_min_samples=10 --perf_force_samples=10 --gtest_output=xml:perf-smoke.xml
```

This checks test wiring only. Select performance workloads with
`--gtest_filter`; warmed public-call timing includes submission and stream
completion. It is distinct from isolated kernel or graph replay timing.

Installation and cache checks
-----------------------------

Use the installed headers in an independent C++ consumer. Check the selected
Python extension and loaded libraries after moving an installation prefix.

Follow @ref triton_runtime_toolchain for separate empty-cache, persisted-cache
without compiler, and empty-cache without compiler checks. See
@ref tutorial_triton_validation for accuracy/performance scope and evidence
requirements. Save results with the actual machine, device, driver, toolkit,
compiler and source/artifact identities; this tutorial does not certify them.

For example, save the following as `consumer/main.cpp`:

```cpp
#include <opencv2/triton.hpp>
#include <opencv2/core.hpp>
#include <iostream>

int main()
{
    if (!cv::triton::haveBackend()) return 1;
    cv::Mat host(96, 128, CV_8UC3, cv::Scalar(64, 128, 192));
    cv::cuda::GpuMat src(host), dst;
    cv::cuda::Stream stream;
    cv::triton::resize(src, dst, cv::Size(64, 48), 0, 0,
                       cv::INTER_LINEAR, stream);
    stream.waitForCompletion();
    cv::Mat result;
    dst.download(result);
    std::cout << result.size() << "\n";
    return result.size() == cv::Size(64, 48) ? 0 : 2;
}
```

Save this as `consumer/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(triton_consumer LANGUAGES CXX)
find_package(OpenCV REQUIRED COMPONENTS core imgproc triton)
add_executable(triton_consumer main.cpp)
target_include_directories(triton_consumer PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(triton_consumer PRIVATE ${OpenCV_LIBS})
```

Build against the installed package, with the environment exports above:

```sh
cmake -S consumer -B consumer-build -G Ninja \
  -DOpenCV_DIR=<install>/lib/cmake/opencv5
cmake --build consumer-build --parallel
OPENCV_TRITON_CACHE_DIR=<new-empty-cache> consumer-build/triton_consumer
OPENCV_TRITON_CACHE_DIR=<new-empty-cache> \
  OPENCV_TRITON_PYTHON=/nonexistent/triton-python consumer-build/triton_consumer
```

The second command uses the same persisted cache in a new process and must
succeed without the compiler interpreter. Repeat with a **different** empty
cache and the nonexistent interpreter to verify a visible `toolchain_unavailable`
exception. That last command is expected to exit with an error, not succeed or
silently fall back. These checks exercise the module artifact cache; they do not
measure a fully cold compiler cache.
