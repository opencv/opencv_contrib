// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright 2026, The FlagOS Contributors.
// Private backend-neutral image views and resize entry. Not installed.
// Public adapters validate native buffers and streams before calling the runtime.
#ifndef OPENCV_TRITON_CORE_INTERNAL_HPP
#define OPENCV_TRITON_CORE_INTERNAL_HPP
#include <cstddef>
#include <cstdint>
#include "opencv2/core.hpp"
namespace cv
{
namespace triton
{

//! @addtogroup triton
//! @{

/** @brief Describes one device-resident image buffer for the core API.
 *
 * Carries byte pitch, not element stride, because that is what device
 * pointer arithmetic needs. whole_* records the parent allocation so a ROI
 * view addresses correctly (both backends already key off the real step,
 * the same contract a ROI GpuMat reports) -- set whole_rows/whole_cols equal
 * to rows/cols and offset_x/offset_y to 0 for a non-ROI/top-level buffer.
 *
 * MODULE-INTERNAL: see the note at the top of this file.
 */
struct ImageView
{
    std::uintptr_t ptr; //!< device pointer, as an integer
    int rows;
    int cols;
    int type;         //!< OpenCV type, e.g. CV_8UC3
    std::size_t step; //!< row pitch IN BYTES
    int whole_rows;   //!< parent allocation height
    int whole_cols;   //!< parent allocation width
    int offset_y;     //!< ROI offset within the parent
    int offset_x;

    ImageView()
        : ptr(0), rows(0), cols(0), type(0), step(0), whole_rows(0), whole_cols(0), offset_y(0),
          offset_x(0)
    {
    }

    bool isRoi() const
    {
        return offset_x != 0 || offset_y != 0 || whole_rows != rows || whole_cols != cols;
    }
    int channels() const
    {
        return CV_MAT_CN(type);
    }
    int depth() const
    {
        return CV_MAT_DEPTH(type);
    }
};

/** @brief Device/stream handle for the core API, as plain integers.
 *
 * `device` and `stream` must identify the context that owns src/dst. Neither
 * this struct nor resizeCore() creates a device context of its own: that is
 * the caller's (or the adapter's) responsibility, same as the private
 * backend::Context it maps onto.
 */
struct ExecutionContext
{
    int device;            //!< platform device ordinal
    std::uintptr_t stream; //!< platform stream handle, may be 0

    ExecutionContext() : device(-1), stream(0)
    {
    }
};

/** @brief Resize an image using a Triton-generated kernel -- core entry point.
 *
 * This performs every check cv::triton::resize(GpuMat...) performs (depth,
 * channels, interpolation, explicit dsize, in-place rejection, overlap
 * rejection -- ROI src/dst are accepted, not rejected) against the
 * ImageView/ExecutionContext pair directly, with no GpuMat, no cuda::Stream
 * and no CUDA/HIP header dependency. The caller preallocates `dst` at `dsize`;
 * this layer does not allocate output storage.
 *
 * @param src   Source image view, CV_8U or CV_32F, 1/3/4 channels, ROI
 *              views accepted.
 * @param dst   Destination image view, already sized to dsize, ROI views
 *              accepted, not aliased with src.
 * @param dsize Explicit output size. Must equal dst's size.
 * @param interpolation INTER_NEAREST, INTER_LINEAR, INTER_CUBIC or INTER_AREA.
 * @param ctx   Device and stream that own src/dst.
 * @param scaleX Kernel-space horizontal scale (src cols per dst pixel), or 0 to derive the
 *               computation from dsize alone. When both scales are given, their inverses are
 *               used as cv::resize's fx/fy, and dsize must equal cvRound(src * (1/scale)).
 *               1/(1/fx) is not always fx in double: a caller holding the original fx/fy that
 *               needs bit-exactness with cv::resize(fx, fy) should use the GpuMat entry,
 *               which passes them through unchanged.
 * @param scaleY Kernel-space vertical scale (src rows per dst pixel). 0
 *               derives it as src.rows / dsize.height; see scaleX.
 */
void resizeCore(const ImageView &src, const ImageView &dst, Size dsize, int interpolation,
                const ExecutionContext &ctx, double scaleX = 0, double scaleY = 0);

//! @}

}
} // namespace cv::triton
#endif  // OPENCV_TRITON_CORE_INTERNAL_HPP
