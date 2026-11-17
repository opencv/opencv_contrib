// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright 2026, The FlagOS Contributors.
// Caller-owned GPU images and native streams. Output is preallocated; calls are
// asynchronous and do not fall back to CPU. Retain all storage until completion.
// Source, destination and stream must belong to the selected device/context.
// device is the backend device index. stream is hipStream_t/CUstream encoded as
// an integer, or 0 for the default stream. Unlike the CUDA GpuMat overloads, each
// DeviceImage call requires an explicit stream argument; pass 0 for the default.
// Source and destination must not overlap.
#ifndef OPENCV_TRITON_DEVICE_HPP
#define OPENCV_TRITON_DEVICE_HPP
#include <cstddef>
#include <cstdint>
#include "opencv2/core.hpp"
namespace cv
{
namespace triton
{

//! @addtogroup triton
//! @{

/** @brief One device-resident 2-D image: the address of its first element, its size, type and
 * row pitch in BYTES. A ROI view is described by its own first element and the parent's pitch. */
struct CV_EXPORTS DeviceImage
{
    std::uintptr_t ptr = 0; //!< device address of element (0, 0)
    int rows = 0, cols = 0;
    int type = 0;         //!< OpenCV type, e.g. CV_8UC3
    std::size_t step = 0; //!< row pitch in bytes; must cover a full packed row (zero is invalid)
};

/** @brief Return whether device is usable on the configured backend.
 * @param device Backend device index.
 */
CV_EXPORTS bool haveDevice(int device);

/** @brief Resize a device image to an explicit size.
 * @param src Source image; supported types match cv::triton::resize.
 * @param dst Preallocated output of size dsize and src.type; must not overlap src.
 * @param dsize Positive output size.
 * @param interpolation INTER_NEAREST, INTER_LINEAR, INTER_CUBIC or INTER_AREA.
 * @param device Backend device index owning both images and stream.
 * @param stream Required native hipStream_t/CUstream encoded as an integer; 0 selects default.
 * Retain image storage until stream completion.
 */
CV_EXPORTS void resize(const DeviceImage &src, const DeviceImage &dst, Size dsize,
                       int interpolation, int device, std::uintptr_t stream);

/** @brief Resize using explicit size or scale factors.
 * @param src Source image; supported types match cv::triton::resize.
 * @param dst Preallocated output of the resolved size and src.type; must not overlap src.
 * @param dsize Output size; if empty, derive it from fx/fy. Explicit size takes precedence.
 * @param fx Horizontal scale; finite and positive when dsize is empty, otherwise ignored.
 * @param fy Vertical scale; finite and positive when dsize is empty, otherwise ignored.
 * @param interpolation INTER_NEAREST, INTER_LINEAR, INTER_CUBIC or INTER_AREA.
 * @param device Backend device index owning both images and stream.
 * @param stream Required native stream handle; 0 selects default. Retain storage until completion.
 */
CV_EXPORTS void resize(const DeviceImage &src, const DeviceImage &dst, Size dsize, double fx,
                       double fy, int interpolation, int device, std::uintptr_t stream);

/** @brief Convert a device image between colour spaces.
 * @param src Source image with the code's required depth and channels.
 * @param dst Preallocated code-specific output; must not overlap src. NV12/NV21
 *            output has 2/3 of src.rows; other conversions preserve size.
 * @param code Supported cv::COLOR_* value; see @ref tutorial_triton_cvtcolor.
 * @param device Backend device index owning both images and stream.
 * @param stream Required native stream handle; 0 selects default. Retain storage until completion.
 */
CV_EXPORTS void cvtColor(const DeviceImage &src, const DeviceImage &dst, int code, int device,
                         std::uintptr_t stream);

/** @brief Apply an affine transformation to a device image.
 * @param src Source image; supported types match cv::triton::warpAffine.
 * @param dst Preallocated output; determines output size and must have src.type.
 *            Must not overlap src. BORDER_TRANSPARENT requires initialized contents.
 * @param M Six row-major coefficients of the 2x3 transformation matrix.
 * @param flags Supported interpolation, optionally combined with WARP_INVERSE_MAP.
 * @param borderMode Supported border mode; see cv::triton::warpAffine.
 * @param borderValue Per-channel value used by BORDER_CONSTANT.
 * @param device Backend device index owning both images and stream.
 * @param stream Required native stream handle; 0 selects default. Retain storage until completion.
 */
CV_EXPORTS void warpAffine(const DeviceImage &src, const DeviceImage &dst, const double M[6],
                           int flags, int borderMode, const Scalar &borderValue, int device,
                           std::uintptr_t stream);

//! @}

}
} // namespace cv::triton
#endif  // OPENCV_TRITON_DEVICE_HPP
