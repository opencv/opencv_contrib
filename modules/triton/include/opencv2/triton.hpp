// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Triton-backed image operations. See the module README and usage guides for
// supported inputs, numerical behavior and build requirements.
#ifndef OPENCV_TRITON_HPP
#define OPENCV_TRITON_HPP
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/imgproc.hpp"   // INTER_LINEAR / INTER_NEAREST
/** @defgroup triton Triton-backed operations
 *
 * Optional GPU operations compiled on first use by Python/Triton. PyTorch is not required.
 * CUDA uses GpuMat; HIP uses opencv2/triton/device.hpp. Calls are asynchronous.
 * Unsupported inputs raise an exception without CPU or cv::cuda fallback.
 *
 * @see @ref tutorial_triton_resize
 * @see @ref tutorial_triton_cvtcolor
 * @see @ref tutorial_triton_warpaffine
 * @see @ref tutorial_triton_cuda_first_use
 * @see @ref tutorial_triton_hip_first_use
 * @see @ref tutorial_triton_maintenance
 * @see @ref tutorial_triton_validation
 */

namespace cv
{
namespace triton
{

//! @addtogroup triton
//! @{

/** @brief Return whether the selected backend has a usable device.
 * CUDA checks its current device; HIP checks logical device 0 after visibility filtering.
 * Use haveDevice() from opencv2/triton/device.hpp to query an explicit device index.
 * Does not check compiler availability; a cache miss may still raise a toolchain error.
 */
CV_EXPORTS_W bool haveBackend();

/** @brief Name of the active backend, or an empty string if none. */
CV_EXPORTS_W String backendName();

/** @brief Release all capture holds maintained by this loaded module's backend runtime.
 *
 * Caller preconditions: stop related captures, replays and concurrent submissions;
 * confirm all submitted replays complete through their streams/events; then destroy
 * or permanently disable every related graph executable before calling this function.
 * Destroying a graph template does not disable its executables; destroying an executable
 * does not establish completion of its in-flight work.
 *
 * Holds are process-local to this loaded runtime and shared across its threads/contexts,
 * not selected by graph or current context. Internal locking does not coordinate callers
 * or provide a GPU completion fence. See @ref triton_graph_release for the full contract.
 */
CV_EXPORTS_W void releaseGraphResources();

/** @brief Resize a GPU image.
 *
 * Scales src to dsize, or derives the size from fx/fy. Output keeps src's type
 * and channel count; a same-size request copies the image without interpolation.
 * Supports CV_8U/CV_32F, 1/3/4 channels and pitched ROIs. Source and destination
 * must not overlap. Numerical behavior follows the selected CPU reference;
 * see @ref tutorial_triton_resize.
 *
 * @param src Source GpuMat.
 * @param dst Destination GpuMat; allocated if needed, otherwise reused.
 * @param dsize Output size. If empty, derive each dimension as cvRound(input size * scale).
 * @param fx Horizontal scale; finite and positive when dsize is empty, otherwise ignored.
 * @param fy Vertical scale; finite and positive when dsize is empty, otherwise ignored.
 * @param interpolation INTER_NEAREST, INTER_LINEAR, INTER_CUBIC or INTER_AREA.
 * @param stream Optional CUDA stream. If omitted, the default stream
 *               (`cuda::Stream::Null()`) is used. Calls are asynchronous;
 *               retain input/output storage until this stream completes.
 */
CV_EXPORTS void resize(InputArray src, OutputArray dst, Size dsize, double fx = 0, double fy = 0,
                       int interpolation = INTER_LINEAR,
                       cuda::Stream &stream = cuda::Stream::Null());

/** @overload
 * @brief GpuMat form of resize.
 * The Python name is `cv2.triton.resize`; see @ref tutorial_triton_resize.
 */
CV_EXPORTS void resize(const cuda::GpuMat &src, CV_OUT cuda::GpuMat &dst, Size dsize, double fx = 0,
                       double fy = 0, int interpolation = INTER_LINEAR,
                       cuda::Stream &stream = cuda::Stream::Null());

/** @brief Convert a GPU image between colour spaces.
 *
 * Supports channel permutations, gray, YUV/YCrCb, forward HSV, packed BGR565/555,
 * bilinear Bayer, XYZ/Lab/Luv and video decoding. CV_8U is supported for all
 * listed codes; selected conversions also support CV_16U/CV_32F. See
 * @ref tutorial_triton_cvtcolor for the code/depth table and numerical bounds.
 *
 * Output retains src's depth; code determines its channels. Packed BGR565/555
 * uses CV_8UC2. NV12/NV21 takes a CV_8UC1 image of size W by 3H/2 and produces
 * W by H; W must be even and source rows divisible by three. Other conversions
 * preserve size; 4:2:2 colour decoding requires even width. Added alpha is 255,
 * 65535 or 1 for CV_8U, CV_16U or CV_32F, respectively.
 *
 * @param src Source GpuMat with the depth/channels required by code. Pitched ROIs are supported.
 * @param dst Destination GpuMat; allocated or reused at the code's output size/type.
 *            Source and destination must not overlap.
 * @param code Supported cv::COLOR_* value; no dstCn override is provided.
 * @param stream Optional CUDA stream. If omitted, the default stream
 *               (`cuda::Stream::Null()`) is used. Calls are asynchronous;
 *               retain input/output storage until this stream completes.
 */
CV_EXPORTS void cvtColor(InputArray src, OutputArray dst, int code,
                         cuda::Stream &stream = cuda::Stream::Null());

/** @overload
 * @brief GpuMat form of cvtColor.
 * The Python name is `cv2.triton.cvtColor`.
 */
CV_EXPORTS void cvtColor(const cuda::GpuMat &src, CV_OUT cuda::GpuMat &dst, int code,
                         cuda::Stream &stream = cuda::Stream::Null());

/** @brief Convert a CUDA GpuMat; see cvtColor for supported inputs and conversions.
 * The Python stream argument is optional; omitted or `None` selects the default
 * CUDA stream. Calls are asynchronous.
 */
CV_EXPORTS_AS(cvtColor)
void cvtColorForPython(const cuda::GpuMat &src, CV_OUT cuda::GpuMat &dst, int code,
                       Ptr<cuda::Stream> stream = Ptr<cuda::Stream>());

/** @brief Apply an affine transformation to a GPU image.
 *
 * With WARP_INVERSE_MAP, samples src at (M00*x + M01*y + M02,
 * M10*x + M11*y + M12) for each destination pixel (x, y); otherwise M is inverted
 * first. Output retains src's type and channels. In-place operation is unsupported.
 * Supports CV_8U/CV_16U/CV_32F, 1/3/4 channels and pitched ROIs.
 *
 * BORDER_TRANSPARENT preserves wholly outside pixels. For LINEAR/AREA, each
 * outside tap contributes the previous destination value with its original
 * weight. Supply initialized dst; borderValue does not initialize it.
 * Numerical behavior uses a selected CPU sampling path and can differ from a
 * natural-width cv::warpAffine call. See @ref tutorial_triton_warpaffine.
 *
 * @param src Source GpuMat.
 * @param dst Destination GpuMat; allocated if needed, otherwise reused. Must not overlap src.
 *            BORDER_TRANSPARENT requires initialized storage of the output size/type.
 * @param M 2x3 CV_32F/CV_64F forward matrix; interpreted as an inverse map with WARP_INVERSE_MAP.
 * @param dsize Output size; empty means src.size().
 * @param flags INTER_NEAREST, INTER_LINEAR, INTER_CUBIC or INTER_AREA (LINEAR alias),
 *              optionally combined with WARP_INVERSE_MAP.
 * @param borderMode BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_REFLECT_101,
 *                   BORDER_WRAP or BORDER_TRANSPARENT. CUBIC with TRANSPARENT is unsupported.
 * @param borderValue Constant-border value per channel; Scalar(v) means (v, 0, 0, 0).
 * @param stream Optional CUDA stream. If omitted, the default stream
 *               (`cuda::Stream::Null()`) is used. Calls are asynchronous;
 *               retain input/output storage until this stream completes.
 */
CV_EXPORTS void warpAffine(InputArray src, OutputArray dst, InputArray M, Size dsize,
                           int flags = INTER_LINEAR, int borderMode = BORDER_CONSTANT,
                           const Scalar &borderValue = Scalar(),
                           cuda::Stream &stream = cuda::Stream::Null());

/** @overload
 *
 * The form exposed to Python:
 * `cv2.triton.warpAffine(gpu_src, M, dsize[, dst[, flags[, borderMode[, borderValue[, stream]]]]]) -> dst`.
 */
CV_EXPORTS void warpAffine(const cuda::GpuMat &src, CV_OUT cuda::GpuMat &dst, InputArray M,
                           Size dsize, int flags = INTER_LINEAR, int borderMode = BORDER_CONSTANT,
                           const Scalar &borderValue = Scalar(),
                           cuda::Stream &stream = cuda::Stream::Null());

/** @brief Resize a CUDA GpuMat; see resize for supported inputs and interpolation.
 * The Python stream argument is optional; omitted or `None` selects the default
 * CUDA stream. Calls are asynchronous.
 */
CV_EXPORTS_AS(resize)
void resizeForPython(const cuda::GpuMat &src, CV_OUT cuda::GpuMat &dst, Size dsize, double fx = 0,
                     double fy = 0, int interpolation = INTER_LINEAR,
                     Ptr<cuda::Stream> stream = Ptr<cuda::Stream>());

/** @brief Transform a CUDA GpuMat; see warpAffine for sampling and border behavior.
 * The Python stream argument is optional; omitted or `None` selects the default
 * CUDA stream. Calls are asynchronous.
 */
CV_EXPORTS_AS(warpAffine)
void warpAffineForPython(const cuda::GpuMat &src, CV_OUT cuda::GpuMat &dst, InputArray M,
                         Size dsize, int flags = INTER_LINEAR, int borderMode = BORDER_CONSTANT,
                         const Scalar &borderValue = Scalar(),
                         Ptr<cuda::Stream> stream = Ptr<cuda::Stream>());

//! @}

}
} // namespace cv::triton
#endif  // OPENCV_TRITON_HPP
