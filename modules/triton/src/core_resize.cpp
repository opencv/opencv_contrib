// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Platform-agnostic core of cv::triton::resize. Both entries -- resizeCore() (device pointers)
// and the GpuMat adapter (cuda_resize_adapter.cpp) -- reduce to detail::resizeDevice(), which
// hands the request to the runtime JIT (src/runtime/runtime.hpp) and maps its status to an
// OpenCV error. Numerical behavior follows the CPU cv::resize formulas. There is
// no CPU or cv::cuda fallback: a successful call has submitted the requested
// computation or the identity copy.
#include "precomp.hpp"
#include "runtime/cvtcolor.hpp"
#include "runtime/warpaffine.hpp"
#include "runtime/runtime.hpp"
#include "entry.hpp"
namespace cv
{
namespace triton
{

namespace detail
{

cv_triton_runtime::DeviceView toDeviceView(const ImageView &v, int device,
                                           std::uintptr_t alloc_base, std::size_t alloc_bytes)
{
    cv_triton_runtime::DeviceView d;
    d.ctx.device = device; // the runtime identifies the context itself (cuCtxGetId)
    d.ptr = v.ptr;         // the first element of the view: the ROI offset is already applied
    d.rows = v.rows;
    d.cols = v.cols;
    d.channels = CV_MAT_CN(v.type);
    const int depth = CV_MAT_DEPTH(v.type);
    if (depth != CV_8U && depth != CV_32F && depth != CV_16U)
        CV_Error(Error::StsUnsupportedFormat,
                 "cv::triton: only CV_8U, CV_16U and CV_32F are supported");
    d.type = depth == CV_8U ? cv_triton_runtime::ElemType::U8
                            : (depth == CV_16U ? cv_triton_runtime::ElemType::U16
                                               : cv_triton_runtime::ElemType::F32);
    d.step = v.step;
    d.alloc_base = alloc_base;
    d.alloc_bytes = alloc_bytes;
    return d;
}

void resizeDevice(const cv_triton_runtime::DeviceView &src,
                  const cv_triton_runtime::DeviceView &dst, Size dsize, double fx, double fy,
                  int interpolation, int device, std::uintptr_t stream)
{
    cv_triton_runtime::ResizeRequest req;
    req.src = src;
    req.dst = dst;
    req.dsize_w = dsize.width;
    req.dsize_h = dsize.height;
    req.fx = fx;
    req.fy = fy;
    req.interpolation = interpolation;
    cv_triton_runtime::ExecContext ex;
    ex.ctx = req.src.ctx;
    ex.ctx.device = device;
    ex.stream = stream;
    const cv_triton_runtime::Status s = runtime::resize(req, ex);
    requireStatus(s, "resize", true);
    // Asynchronous: everything is on `stream`. No implicit synchronisation.
}

void warpAffineDevice(const cv_triton_warp::Request &req, int device, std::uintptr_t stream)
{
    cv_triton_runtime::ExecContext ex;
    ex.ctx = req.src.ctx;
    ex.ctx.device = device;
    ex.stream = stream;
    const cv_triton_runtime::Status s = runtime::warpAffine(req, ex);
    requireStatus(s, "warpAffine", true);
    // Asynchronous: same contract as resizeDevice().
}

void cvtColorDevice(const cv_triton_color::Request &req, int device, std::uintptr_t stream)
{
    cv_triton_runtime::ExecContext ex;
    ex.ctx = req.src.ctx;
    ex.ctx.device = device;
    ex.stream = stream;
    const cv_triton_runtime::Status s = runtime::cvtColor(req, ex);
    requireStatus(s, "cvtColor", true);
    // Asynchronous: same contract as resizeDevice().
}

} // namespace detail

void resizeCore(const ImageView &src, const ImageView &dst, Size dsize, int interpolation,
                const ExecutionContext &ctx, double scaleX, double scaleY)
{
    if (dsize.width <= 0 || dsize.height <= 0)
        CV_Error(Error::StsUnsupportedFormat,
                 "cv::triton::resizeCore: dsize must be explicit and non-empty");
    // The shared view supports all operator depths; resize admits only CV_8U/CV_32F.
    // Reject other depths before compilation with StsUnsupportedFormat.
    for (const ImageView *v : {&src, &dst})
    {
        const int depth = CV_MAT_DEPTH(v->type);
        if (depth != CV_8U && depth != CV_32F)
            CV_Error(
                Error::StsUnsupportedFormat,
                cv::format("cv::triton::resize: depth %d is not supported (CV_8U and CV_32F only)",
                           depth));
    }
    if (!runtime::deviceAvailable(ctx.device))
        CV_Error(Error::StsNotImplemented,
                 "cv::triton::resizeCore: no usable backend device for ctx.device; "
                 "this module never falls back to CPU or cv::cuda");
    // scaleX/scaleY are kernel-space (source pixels per destination pixel); the computation uses
    // the inverse, as cv::resize's fx/fy. 1/(1/fx) is not always fx in double, so a caller that
    // holds the original fx/fy and needs bit-exactness with cv::resize(fx, fy) should use the
    // GpuMat entry, which passes them through unchanged.
    const double fx = scaleX > 0 ? 1.0 / scaleX : 0.0;
    const double fy = scaleY > 0 ? 1.0 / scaleY : 0.0;
    // dsize is explicit here, so cv::resize's precedence ignores fx/fy unless both scales are
    // given: then they are the inverse scales and dsize must already equal cvRound(src * f).
    Size ds = dsize;
    if (fx > 0 && fy > 0)
    {
        if (dsize != Size(cvRound(src.cols * fx), cvRound(src.rows * fy)))
            CV_Error(Error::StsBadArg,
                     "cv::triton::resizeCore: dsize disagrees with scaleX/scaleY");
        ds = Size();
    }
    detail::resizeDevice(detail::toDeviceView(src, ctx.device, 0, 0),
                         detail::toDeviceView(dst, ctx.device, 0, 0), ds, fx, fy, interpolation,
                         ctx.device, ctx.stream);
}

}
} // namespace cv::triton
