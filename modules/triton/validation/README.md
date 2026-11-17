# Private validation support

This directory contains shared test cases, numerical comparisons and backend
adapters. Its headers are not installed as a public API. `reference.hpp`,
`resize_cases.hpp`, `cvtcolor_cases.hpp` and `numerics.hpp` supply common inputs
and criteria to the accuracy and performance tests.

Resize uses the portable CPU evaluator under `test/`, modeling the documented
generic CPU algorithm independently of linked-library dispatch choices.
CvtColor compares against the linked `cv::cvtColor`. WarpAffine uses the linked
CPU vector-path reference for NEAREST/LINEAR/AREA and a natural-width reference
for CUBIC. These choices and their limits are documented in the operator guides;
sharing fixtures does not establish equivalence across arbitrary CPU builds.

`device.hpp` is a private adapter interface. CMake selects `device_cuda.cpp`,
which calls public GpuMat entries, or `device_hip.cpp`, which calls public device-
buffer entries. Both preserve parent/ROI lifetimes, pitch and output reuse.
`Stream::waitFor` enqueues a producer dependency rather than waiting on the host.
Setup and result transfers are blocking and outside performance cycles.

Performance cycles include public submission plus caller-stream completion.
They do not isolate kernel execution or graph replay. Native CUDA comparisons
are optional and are excluded on HIP. Backend-specific context, stream, graph
and resource-lifetime checks remain in their own source lists. The supplemental
HIP executable groups checks and reports internal counts separately from
registered tests; standard OpenCV list/filter/XML options operate on its groups.

Production adapters share `src/entry.hpp` metadata resolution and request
validation. GpuMat entries allocate or reuse output storage; device-buffer entries
validate caller-provided output storage before launch. The runtime owns compiled
modules and tables.
On DTK, device/range and stream-context queries are available, while
`HIP_POINTER_ATTRIBUTE_CONTEXT` does not identify allocation context. Tests must
preserve that limitation rather than infer foreign-context ownership.
