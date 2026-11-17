Maintaining the Triton module {#tutorial_triton_maintenance}
=============================

Read by task: [source and call flow](#source-layout), [terms](#terms),
[errors](#error-messages), [resource lifetime](#cleanup),
[compiler/cache](#runtime-toolchain), [generated sources](#generated-hip-sources).

<a id="source-layout"></a>
Source layout
-------------

All sources and generation tools needed to build the module are in this
module directory. `kernels/` contains GPU implementations; `cpu` in a kernel
filename means it reproduces CPU arithmetic, not that it executes on the CPU. `src/` contains
entry validation, adapters and runtime code. Runtime resize planning is separate
from the CPU output evaluator under `test/`; tests also compare against the
linked OpenCV CPU implementations where documented. Keep algorithmic source
attribution and license notices with the relevant implementation.

Describe current behavior in guides and explain invariants in code comments.
Keep development logs, run reports and build artifacts outside the source tree.
Generated-source notices and integrity hashes remain with their generated files.

<a id="architecture"></a>
Architecture and ownership {#triton_architecture}
--------------------------

The module provides explicit `cv::triton::*` calls. It does not intercept ordinary
`cv::*`, UMat or `cv::cuda::*` calls. Any future integration into main-library
accelerator dispatch is a separate design decision.

For a first source read, follow one call through these layers:

| Layer | Start here | Responsibility |
|---|---|---|
| Public API | `include/opencv2/triton.hpp`, `include/opencv2/triton/device.hpp` | Input/output types, supported options and asynchronous ownership |
| Entry validation | `src/cuda_resize_adapter.cpp`, `src/device_entries.cpp` | Output metadata, actual device views, overlap and stream admission |
| Operation plan | `src/runtime/provider.cpp`, `cvtcolor.cpp`, `warpaffine.cpp` | Sampling rules, kernel choice, compile key and launch shape |
| Backend runtime | `src/runtime/cuda_runtime.cpp` and its generated HIP counterpart | Compile/load, stream submission, caches and resource retention |
| Device computation | `kernels/` | The arithmetic and memory accesses executed on the GPU |
| Validation | `validation/`, `test/`, `perf/` | Shared cases, numerical references, lifecycle checks and timing |

The call flow is the same for the three operations:

```text
cv::triton::<operation>
  → adapter checks shapes, device views and caller stream
  → C++ plan selects sampling rules, tables and launch arguments
  → runtime reuses a compiled binary, or invokes the compile helper on a miss
  → GPU work is submitted to the caller stream
  → caller waits before reading/releasing images; runtime retains its own resources
```

### Names used in the source

These namespaces describe existing layers, not additional public APIs:

| Name | Meaning / entry |
|---|---|
| `cv::triton` | Installed public operations and backend queries |
| `cv::triton::detail` | Private adapter/output validation in `src/entry.hpp` and `src/core.hpp` |
| `cv_triton_runtime` | Private device views, status, keys and backend services in `src/runtime/provider.hpp` |
| `cv_triton_resize` | Resize planning and CPU-compatible sampling tables in `resize_plan.hpp` |
| `cv_triton_color`, `cv_triton_warp` | Colour specifications and affine sampling plans |
| `triton_validation` | Test/perf device adapters, selected by CMake; not installed |
| `triton_graph` | Private HIP graph request parser in `perf/hip/graph_request.hpp` |

Short local aliases do not introduce a new layer: `ref` usually names the
resize reference/planner and `wa` the warp planner. `tr` in the HIP tests names
`cv::triton`, `td` in the test helpers and `D` in the HIP graph tool name
`cv::triton::detail`, and `J` in that tool names `cv_triton_runtime`.
Check the adjacent `namespace ... = ...` declaration for a file's exact alias.

`validation/` contains private shared accuracy/performance support. CMake
selects `device_cuda.cpp` or `device_hip.cpp`; each calls its backend's public
module entry points. These test helpers are not installed. Backend lifecycle
tests stay in their backend-specific source lists. Update common cases and
numerical criteria when adding support, and retain tests for unsupported inputs.

Size-dependent tables are built in C++, uploaded on the caller's stream and
cached. Cross-stream consumers wait on upload events. In-flight calls retain
modules, tables and upload staging bytes even after cache eviction. Warm calls
can be captured on a stream where compilation, module loads and table uploads
are already complete. Release captured resources only under the
[Graph resources](#graph-release) coordination and completion rules below.

<a id="cleanup"></a>
Asynchronous cleanup and context ownership
-----------------------------------------

An in-flight call retains its modules, device tables and upload host bytes until
completion is proven. The decisions in `src/runtime/lifetime.hpp` distinguish
unfinished work from failed cleanup:

| Condition | Cleanup behavior |
|---|---|
| Stream capture is active | Skip completion queries and reclamation, including failed-event metadata reaping. |
| Context is known destroyed | Release host ownership without querying old events or switching to that context. |
| Context state is unknown | Retain ownership and report an error; do not infer destruction. |
| Completion query succeeds | Release references for the completed operation and attempt event cleanup. |
| Completion query is not-ready | Retain resources for a later noncapturing check. A device-side wait alone does not prove upload host bytes can be released. |
| Completion query fails | Retain resources and report the error through submission/reap status, diagnostics and private counters. |
| Upload started without a valid recorded completion marker | Retain ownership until context destruction is established. |

If last-owner cleanup fails, a private process-lifetime retirement queue holds
the entire module/table owner, including upload host bytes. Noncapturing
submissions and explicit reaps retry eligible completion checks without adding
normal-path synchronization. An unknown result from free, unload or event
destruction is not retried as the same irreversible call; that owner remains
held until its context is known destroyed.

While failed resource cleanup remains unresolved, later submissions and explicit
cleanup calls report an error rather than silently accumulating resources.
Recovery requires proven completion/context destruction, or a process restart
for persistent driver failure. Inspect diagnostics and counters; retention until
process exit is not successful cleanup.

There is a distinct completed-event case: if event destruction fails after
completion was proven, module/table references may be released. The failure is
still reported, and the event handle/context identity remain as failed metadata
until context destruction is known or the process exits. That irreversible
destroy is not retried. No unfinished work is implied, so this metadata does not
block unrelated calls. `RuntimeStats::retired_resources` includes these handles.

Primary-context ownership follows these rules:

- The runtime retains one primary context per device for its process-lifetime caches.
- Host threads borrow it, including after `SetCurrent` failure; calls and short-lived
  threads do not add retains.
- HIP's `hipCtxGetId` path and its fallback share that owner. The fallback registry
  adds identity metadata, not another retain.
- Static teardown does not release the owner after the driver may be gone.
  Arbitrary primary-context resets and private contexts are unsupported by this fallback.

<a id="graph-release"></a>
Graph resources {#triton_graph_release}
--------------------------------------

`releaseGraphResources()` drops **all capture holds maintained by this loaded
module's selected backend runtime**, across its host threads and recorded
contexts. It has no graph argument and is not scoped to the calling thread,
current context or one graph. Other independently loaded runtimes and caller
image storage have separate ownership.

The application coordinates this sequence across every participating thread:

1. Stop new related capture, replay and concurrent operator submissions.
2. Confirm every submitted related replay has completed on its actual stream,
   or through an event ordered after it. Waiting on an unrelated stream is insufficient.
3. Destroy or permanently disable all related graph executables, preventing future replay.
4. Call `cv::triton::releaseGraphResources()` while that coordination remains in effect.

Destroying the graph template used for instantiation does not disable its
executables. Destroying an executable is also not evidence that its submitted
GPU work has completed. The runtime does not track those executable handles.
Its mutex protects capture-hold bookkeeping; it does not stop another thread
from capturing or submitting again, and the release call is not a completion
fence. Driver resource destruction may itself block, but that is not a substitute
for the explicit completion check above. Normal submission adds no new global
synchronization for this contract.

Backend API references: CUDA Driver API 12.8.1,
[`cuGraphDestroy` and `cuGraphExecDestroy`](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__GRAPH.html);
HIP 7.2.0,
[`hipGraphDestroy` and `hipGraphExecDestroy`](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/reference/hip_runtime_api/modules/graph_management.html).
The application must satisfy the module's completion/ownership preconditions
rather than infer them from a destroy call returning.

<a id="runtime-toolchain"></a>
Runtime compiler, ABI and cache {#triton_runtime_toolchain}
-------------------------------

The module is disabled by default and requires no PyTorch dependency when
enabled. Nothing is compiled by Triton during the OpenCV build. On first use,
`tools/compile_helper.py` runs in a separate Python process using the configured
Triton 3.6.0 compiler. The C++ execution path does not embed an interpreter.
Build commands and toolkit examples are maintained in
[CUDA setup][cuda-setup] and
[HIP setup][hip-setup].

The compile key includes source identity, kernel signature, compile-time
parameters, launch options, target and compiler version. The runtime checks
manifest and binary hashes before loading disk artifacts. An unsupported
compiler launch ABI is rejected. See [architecture and ownership](#architecture) for table uploads,
stream dependencies and in-flight ownership.

The writable cache defaults to `~/.cache/opencv/triton`. The following runtime
settings override configured defaults:

| Setting | Purpose |
|---|---|
| `OPENCV_TRITON_CACHE_DIR` | Artifact cache directory |
| `OPENCV_TRITON_RESOURCES` | Kernel and helper resource directory |
| `OPENCV_TRITON_PYTHON` | Compiler interpreter |
| `OPENCV_TRITON_PYTHONPATH` | Compiler Python import path |

Installation places resources under `share/opencv5/triton`, resolved relative
to the module library. A disk-cache hit runs without the compiler interpreter;
a cache miss with no usable compiler raises a toolchain error.

The helper stores its private Triton compiler cache under
`<OPENCV_TRITON_CACHE_DIR>/.triton`, alongside the verified module artifacts.
Selecting a new empty module cache therefore starts with an empty compiler cache
as well. Removing only a module artifact subdirectory can reuse `.triton` output.
The helper overrides the parent process's `TRITON_CACHE_DIR`; record the module
cache and its private compiler-cache path when measuring compilation.

Static installations use the configured resource install prefix when the
library-relative path is unavailable. Set `OPENCV_TRITON_RESOURCES` when relocating
a static SDK. Do not infer loaded-library identity from environment variables:
inspect the native extension and actual mapped OpenCV/CUDA/HIP libraries.

`TRITON_PYTHON_EXECUTABLE` and `TRITON_PYTHONPATH` configure the compiler helper;
`PYTHON3_EXECUTABLE` selects the OpenCV binding interpreter. They may differ.
`TRITON_BACKEND` selects CUDA or HIP; `TRITON_ROCM_PATH` configures the HIP toolkit
and the helper's default `ROCM_PATH`. Disabled-module builds do not probe the
module compiler. No toolchain or compiler package is installed by the module.

The supported launcher ABI is Triton 3.6.0, validated separately for CUDA cubin
and HIP hsaco metadata. Runtime argument types, ordered parameters, warp/block
geometry and zero scratch requirements are checked before launch. A matching
version string alone does not validate an incompatible backend or entry ABI.

<a id="generated-hip-sources"></a>
Generated HIP sources
---------------------

Run the local generators from the module directory after changing their inputs:

```sh
python3 tools/make_kernels_cpu_hip.py
python3 tools/port_hip_runtime.py
python3 tools/make_kernels_cpu_hip.py --check
python3 tools/port_hip_runtime.py --check
```

The first generator derives HIP kernel sources, including the fast paths in
`kernels/resize_hip_fast.py` and `kernels/cvtcolor_hip_fast.py`. The second derives
the HIP runtime from `src/runtime/cuda_runtime.cpp` and the HIP-specific substitutions
in `tools/port_hip_runtime.py`. Edit these sources rather than generated files.
Preserve layout and alignment guards in the planners and runtime.
Configure checks reject stale generated files.

Build and test
--------------

Use out-of-source builds. Check disabled-module builds and enable
`BUILD_TESTS`, `BUILD_PERF_TESTS`, `BUILD_EXAMPLES` and Python bindings as needed.
The Python used for bindings and the interpreter used by the runtime compiler
are selected independently. Use the intended installed extension and libraries
when testing an installation or a relocated prefix.

`opencv_test_triton` uses the common backend cases. CUDA builds add CUDA
lifecycle tests; HIP also provides `opencv_test_triton_hip`. Use the standard
OpenCV list/filter/XML options and review failures and skipped tests.

Accuracy tests list private reference/protocol sources explicitly in CMake and
compile them as separate translation units. CUDA input-validation tests also compile
the private colour/warp plans into the test executable, without exporting those
functions from the module. Separate compilation is not an independent algorithmic
oracle: numerical independence still comes from the selected references and
checks. Keep each support source listed once per executable. Do not include a
`.cpp` file from another `.cpp`, or add an OBJECT library merely to hide duplicate
definitions. Check ordinary and world shared/static linking after changing this
boundary. The plan's no-contraction option is source-local, so a world build
does not change imgproc's floating-point compilation.

`opencv_perf_triton` measures warmed public calls with stream completion on an
explicit non-default stream. Generated sanity data checks repeatability;
independent accuracy tests check numerical behavior. Triton cases live in
`perf/perf_*.cpp`; optional native CUDA cases and their SDK headers live in
`perf/baseline/`. Both use the workload definitions in `perf/perf_cases.hpp`.
The baseline sources and CUDA comparison dependencies are excluded unless
`TRITON_PERF_WITH_NATIVE_BASELINE=ON`. The same executable then contains both
sets; use the filters in the validation guide to select one. Preserve comparable
inputs, devices and timing boundaries when reporting performance.

For deployment/cache checks, use [runtime compilation and cache](#runtime-toolchain) and the backend
tutorials. Backend launch defaults and specialized layout guards are shared where
implemented; benchmark each actual device rather than infer per-architecture tuning.

<a id="terms"></a>
Terms used in this module
------------------------

| Term | Meaning here |
|---|---|
| CPU-compatible GPU kernel | GPU code reproducing the selected CPU algorithm/rounding. The `*_cpu.py` files still execute on the GPU. See the operator numerical guides. |
| Profile | A pattern of arithmetic observed in the linked CPU build, such as fused versus separate multiply/add or row grouping. It is not a GPU model or a performance profile. [Colour][colour-numerics] and [warp][warp-numerics] select it differently. |
| Route | A chosen algorithm branch, such as resize copy, nearest, separable interpolation, integer-area or fractional-area sampling. |
| Plan | C++ output metadata, coordinate/coefficient tables and launch choices prepared for a request. It is separate from GPU execution; see `src/runtime/provider.hpp`. |
| Owner | The object keeping a resource alive. Caller image/stream owners and runtime module/table owners have separate [lifetimes](#cleanup). |
| Step / pitch | Bytes from one image row to the next in public views. Kernel `rs` arguments count typed elements, after division by the element size. |
| DTK / ROCm | HIP toolchains and runtimes from Hygon / AMD respectively. A common HIP API does not imply identical optional queries or targets. |
| `sm_90` / `gfx936` / `gfx1100` | Compiler architecture targets; `gfx*` does not identify the host machine or establish test coverage. |
| cubin / hsaco | CUDA device binary / HIP code object loaded by the backend. A filename or matching compiler version is not sufficient to validate its launch ABI. |
| JIT cache | Persisted binary plus manifest keyed by source, target, types and compile options. This is distinct from a loaded-context module cache and the compiler's own cache. |
| `constexpr` | A Triton compile-time argument. Its value selects a binary specialization; ordinary runtime arguments keep their values at launch time. |
| `do_not_specialize` | Suppresses Triton's implicit value/alignment specialization for the listed runtime arguments. The explicit compile key still identifies types, constexprs and options. |
| Guard band | Test-only bytes outside a destination ROI, checked for unintended writes. It is not padding the operation is allowed to overwrite. |
| FMA / SIMD | Fused multiply-add rounds once; separate multiply/add round twice. SIMD processes lanes together; scalar tails can follow different expression grouping. |

<a id="error-messages"></a>
Reading an error
----------------

Most private runtime statuses reach the public entry as:
`cv::triton::<operation>: <message> (<reason>)`. In Python this is a `cv.error`;
in C++ it is a `cv::Exception`. The message contains the failing condition;
the final lowercase token is the private `Reason` name, not the exception code.

`requireStatus` in `src/entry.cpp` maps an **Unsupported** status to
`StsUnsupportedFormat` for type/mode/route mismatches and `StsBadArg` for other
request errors. An **Error** status becomes `StsError`. Some adapter checks
raise directly, for example HIP GpuMat calls with `StsNotImplemented`, so not
every exception has a reason suffix. The following names come from
`reason_name` in `src/runtime/provider.cpp`:

| Private Reason → suffix | Likely cause | Action |
|---|---|---|
| `DepthOrChannels` → `depth_or_channels` | Unsupported depth or channel count for the operation/code | Use the supported type/code combination in the operator guide. |
| `Interpolation` → `interpolation` | Unsupported interpolation, colour code, flags or border option | Check the operation's supported options; this name also covers colour-code rejection. |
| `EmptyOrSize` → `empty_or_size` | Empty view, invalid output size/scale or packed-video geometry | Check positive dimensions, `dsize`/scale rules and conversion-specific shape. |
| `TypeMismatch` → `type_mismatch` | Source and destination types disagree | Allocate the code-specific output type/channel count. |
| `StepLayout` → `step_layout` | Misaligned pointer/step, too short a row or view outside allocation | Supply byte pitch for the full row and a valid ROI inside its parent allocation. |
| `IndexRange` → `index_range` | Dimensions/strides, coordinates or offsets exceed supported finite/index ranges | Reduce the view or transform; do not truncate large values. |
| `InPlaceOrOverlap` → `in_place_or_overlap` | Source and destination pixel storage overlaps | Use disjoint image views. |
| `ContextMismatch` → `context_mismatch` | Buffer/stream/context identity is incompatible or cannot be established | Use allocations and streams from the selected device/context; inspect the detailed message. |
| `RouteNotEnabled` → `route_not_enabled` | This build does not serve the selected route | Check the installed module's supported routes/resources. |
| `ToolchainUnavailable` → `toolchain_unavailable` | Cache miss without a usable interpreter/compiler, or compiler mismatch | Select a Python with Triton 3.6.0 via `OPENCV_TRITON_PYTHON` and, if needed, `OPENCV_TRITON_PYTHONPATH`. |
| `CompileFailed` → `compile_failed` | Helper/compiler rejected the request | Read the compiler diagnostic and check selected package/target compatibility. |
| `CompileTimeout` → `compile_timeout` | Helper exceeded its deadline | Check compiler progress, process/resource availability and the detailed timeout. |
| `ManifestInvalid` → `manifest_invalid` | Artifact identity/hash/metadata is inconsistent | Check cache permissions and integrity; allow recompilation with a valid toolchain. |
| `LoadFailed` → `load_failed` | Driver cannot load the device binary | Check actual driver, architecture and compiler ABI compatibility. |
| `DeviceAlloc` → `device_alloc` | Device allocation or table setup failed | Check available memory and the underlying driver diagnostic. |
| `LaunchFailed` → `launch_failed` | Submission, event or cleanup operation failed | Check the driver message and [resource-lifetime rules](#cleanup); the suffix alone does not prove a kernel execution fault. |

`None` (`none`) is the no-error reason. If the exception says work had already
been submitted, the output is undefined: preserve the storage until safe
completion/teardown, and do not silently rerun the request elsewhere.

<!-- Link targets: GitHub source paths; Doxygen page labels in the hidden block. -->
<!--
@cond
-->
[cuda-setup]: ../tutorials/cuda_setup.markdown
[hip-setup]: ../tutorials/hip_setup.markdown
[colour-numerics]: cvtcolor.markdown#numerical-behavior
[warp-numerics]: warpaffine.markdown#cpu-reference-and-accuracy
<!--
@endcond
[cuda-setup]: @ref tutorial_triton_cuda_first_use
[hip-setup]: @ref tutorial_triton_hip_first_use
[colour-numerics]: @ref colour-numerics
[warp-numerics]: @ref warp-numerics
@cond
-->
<!-- @endcond -->
