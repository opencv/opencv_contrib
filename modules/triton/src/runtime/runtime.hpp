// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright 2026, The FlagOS Contributors.
// Module-private CUDA/HIP JIT runtime. Not installed.
//
// resize(), cvtColor() and warpAffine() validate requests before device submission.
// resize supports NEAREST, LINEAR, CUBIC and AREA, including fractional AREA;
// the other operators define their supported combinations in their provider headers.
//
// Binaries are resolved through the process-local module cache, the verified disk
// cache, or a separate Torch-free compile-helper process. Python is not embedded.
// Resize coordinate/coefficient tables are built in C++, uploaded on the caller's
// stream, and made visible to other streams through events. Launch arguments are
// checked against the verified manifest and supported compiler ABI.
//
// Modules and tables remain alive until their launches complete. Captured graphs
// retain resources until releaseCapturePins() is called. Cache eviction drops only
// the cache reference; potentially blocking driver cleanup runs outside the state
// mutex. First use of a key may compile, load a module, allocate and upload tables.
#ifndef OPENCV_TRITON_RUNTIME_HPP
#define OPENCV_TRITON_RUNTIME_HPP
#include <cstdint>
#include <string>
#include "opencv2/core.hpp"
#include "../core.hpp"   // module-internal; src/core.hpp is not on the include path
#include "provider.hpp"
#include "cvtcolor.hpp"
#include "warpaffine.hpp"
namespace cv
{
namespace triton
{
namespace runtime
{

/** Is the selected device backend usable for `device`? Pure query: no allocation, no submission. */
bool deviceAvailable(int device);
// One private context lifetime owner for public admission and runtime keys.
CV_EXPORTS cv_triton_runtime::Status contextIdentity(int device, std::uintptr_t &native,
                                                     std::uint64_t &generation);
const char *name(); // backend runtime identifier

/** Run one resize. `req` views must belong to the context current on the calling thread.
 *  Returns Submitted (work is on ex.stream), Unsupported (nothing submitted, dst untouched)
 *  or Error. `route` (if given) receives the route name. */
cv_triton_runtime::Status resize(const cv_triton_runtime::ResizeRequest &req,
                                 const cv_triton_runtime::ExecContext &ex,
                                 std::string *route = nullptr);

/** Run one cvtColor. Same contract as resize(): Submitted, Unsupported (nothing submitted,
 *  dst untouched) or Error. `opName` receives the conversion's name. */
cv_triton_runtime::Status cvtColor(const cv_triton_color::Request &req,
                                   const cv_triton_runtime::ExecContext &ex,
                                   std::string *opName = nullptr);

/** Run one warpAffine. Same contract as resize(): Submitted, Unsupported (nothing submitted,
 *  dst untouched) or Error. */
cv_triton_runtime::Status warpAffine(const cv_triton_warp::Request &req,
                                     const cv_triton_runtime::ExecContext &ex);

}
}
} // namespace cv::triton::runtime

namespace cv
{
namespace triton
{
namespace detail
{

/** ImageView -> the runtime's device view (alloc_base/bytes = the owning allocation, 0 = unknown). */
cv_triton_runtime::DeviceView toDeviceView(const ImageView &v, int device,
                                           std::uintptr_t alloc_base, std::size_t alloc_bytes);
/** The common entry of resizeCore() and the GpuMat adapter. dsize empty = fx/fy given (the
 *  caller's values, passed through unchanged). Raises the mapped OpenCV error. */
void resizeDevice(const cv_triton_runtime::DeviceView &src,
                  const cv_triton_runtime::DeviceView &dst, Size dsize, double fx, double fy,
                  int interpolation, int device, std::uintptr_t stream);
/** cvtColor counterpart: validates, calls runtime::cvtColor, raises the mapped OpenCV error. */
void cvtColorDevice(const cv_triton_color::Request &req, int device, std::uintptr_t stream);
/** warpAffine counterpart: validates, calls runtime::warpAffine, raises the mapped OpenCV error. */
void warpAffineDevice(const cv_triton_warp::Request &req, int device, std::uintptr_t stream);

}
}
} // namespace cv::triton::detail

namespace cv
{
namespace triton
{
namespace detail
{

/** Test-only introspection and control (test/test_runtime.cpp). Exported because OpenCV
 *  builds with hidden visibility; declared only in this private header. */
struct RuntimeStats
{
    long long helper_compiles = 0; //!< helper responses "compiled"
    long long helper_cached = 0;   //!< helper responses "cached" (another process built it)
    long long disk_hits = 0;       //!< verified artifact loaded without running the helper
    long long disk_rejected = 0;   //!< artifact on disk that failed verification
    long long module_loads = 0, module_hits = 0, module_evictions = 0, module_frees = 0;
    long long table_uploads = 0, table_hits = 0, table_evictions = 0, table_frees = 0;
    long long deferred = 0;           //!< evictions of a resource still held by an in-flight launch
    long long cross_stream_waits = 0; //!< launches that waited on another stream's table upload
    long long launches = 0, copies = 0;
    long long cleanup_errors = 0, retired_resources = 0;
    std::string last_cleanup_error;
    long long pending = 0;           //!< launches whose completion has not been observed yet
    long long capture_launches = 0;  //!< launches recorded into a capturing stream
    long long capture_pin_holds = 0; //!< resources currently pinned for captured graphs
    std::string last_route, last_key, last_artifact;
};
CV_EXPORTS RuntimeStats runtimeStats();
CV_EXPORTS void resetRuntimeStats();
/** Cache capacities in entries (defaults 64 modules, 256 table sets). Shrinking evicts now. */
CV_EXPORTS void setCacheLimits(int modules, int tables);
/** Observe completed launches now (normally done at every call). */
CV_EXPORTS void reapCompleted();
/** Release all capture holds across this loaded runtime's contexts and host threads.
 * Caller coordination and completion are required: stop new related capture/replay/
 * submissions, wait for all submitted replays on their streams/events, then destroy or
 * permanently disable related graph executables. Neither template nor executable
 * destruction is a completion fence. This is not a per-graph/current-context release.
 * See cv::triton::releaseGraphResources() and doc/maintenance.markdown, Graph resources.
 * Internal locking protects bookkeeping only; this function does not establish GPU completion.
 */
CV_EXPORTS void releaseCapturePins();
/** Fill newly allocated table buffers with 0xFF before the upload, so a launch that read a
 *  table before its upload completed produces garbage instead of plausible numbers. */
CV_EXPORTS void setPoisonNewTables(bool on);
/** Resolved paths: "resources=<dir>;kernels=<file>;helper=<file>;cache=<dir>;python=<exe>;compiler=<id>". */
CV_EXPORTS std::string runtimeConfig();

}
}
} // namespace cv::triton::detail
#endif
