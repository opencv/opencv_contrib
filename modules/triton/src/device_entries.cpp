// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright 2026, The FlagOS Contributors.
// The device-pointer entries (opencv2/triton/device.hpp). Backend-neutral: compiled into every
// build. The checks are the GpuMat entries' own (cuda_resize_adapter.cpp), stated again here
// because those entries validate GpuMat arguments this layer does not have; the runtime then
// applies the operation's full normalisation, exactly as for a GpuMat call.
#include "precomp.hpp"
#include "opencv2/triton/device.hpp"
#include "runtime/cvtcolor.hpp"
#include "runtime/warpaffine.hpp"
#include "runtime/runtime.hpp"
#include "entry.hpp"
#ifdef OPENCV_TRITON_HAVE_HIP_ADAPTER
#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#else
#include <cuda.h>
#endif
namespace cv
{
namespace triton
{

namespace
{
#ifdef OPENCV_TRITON_HAVE_HIP_ADAPTER
// ROCm uses `type`; some DTK headers retain `memoryType`. Keep both source
// layouts compatible without guessing the toolkit from the platform name.
template <class Attributes>
auto allocationMemoryType(const Attributes &value, int) -> decltype(value.type)
{
    return value.type;
}
template <class Attributes>
auto allocationMemoryType(const Attributes &value, long) -> decltype(value.memoryType)
{
    return value.memoryType;
}
#endif
cv_triton_runtime::DeviceView view(const DeviceImage &m, int device)
{
    if (m.ptr == 0 || m.rows <= 0 || m.cols <= 0)
        CV_Error(Error::StsBadArg,
                 "cv::triton: a device image needs a non-null pointer and a positive size");
    ImageView v;
    v.ptr = m.ptr;
    v.rows = m.rows;
    v.cols = m.cols;
    v.type = m.type;
    v.step = m.step;
    v.whole_rows = m.rows;
    v.whole_cols = m.cols;
    // Native queries below fill the allocation bounds and current context identity.
    auto out = detail::toDeviceView(v, device, 0, 0);
#ifdef OPENCV_TRITON_HAVE_HIP_ADAPTER
    hipCtx_t current = nullptr;
    hipDevice_t actual = 0;
    unsigned long long generation = 0;
    void *base = nullptr;
    std::size_t bytes = 0;
    hipPointerAttribute_t attributes{};
    // DTK does not implement HIP_POINTER_ATTRIBUTE_CONTEXT. Query allocation
    // device/range, and obtain the active context's real lifetime identity.
    std::uintptr_t native = 0;
    std::uint64_t lifetime = 0;
    detail::requireStatus(runtime::contextIdentity(device, native, lifetime),
                          "HIP context identity");
    current = reinterpret_cast<hipCtx_t>(native);
    generation = lifetime;
    if (hipCtxGetDevice(&actual) != hipSuccess || actual != device ||
        hipPointerGetAttributes(&attributes, reinterpret_cast<void *>(m.ptr)) != hipSuccess ||
        attributes.device != device || allocationMemoryType(attributes, 0) != hipMemoryTypeDevice ||
        hipMemGetAddressRange(&base, &bytes, reinterpret_cast<void *>(m.ptr)) != hipSuccess)
        CV_Error(Error::StsBadArg,
                 "device image must be a device allocation on the current HIP device");
    out.ctx.backend = cv_triton_runtime::Backend::HIP;
#else
    CUcontext current = nullptr, owner = nullptr;
    CUdevice actual = 0;
    unsigned long long generation = 0;
    CUdeviceptr base = 0;
    std::size_t bytes = 0;
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || !current ||
        cuCtxGetDevice(&actual) != CUDA_SUCCESS || actual != device ||
        cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, m.ptr) != CUDA_SUCCESS ||
        owner != current || cuCtxGetId(current, &generation) != CUDA_SUCCESS ||
        cuMemGetAddressRange(&base, &bytes, m.ptr) != CUDA_SUCCESS)
        CV_Error(Error::StsBadArg, "device image must belong to the current CUDA context/device");
    out.ctx.backend = cv_triton_runtime::Backend::CUDA;
#endif
    out.ctx.native = reinterpret_cast<std::uintptr_t>(current);
    out.ctx.generation = generation;
    out.alloc_base = (std::uintptr_t)base;
    out.alloc_bytes = bytes;
    detail::requireStatus(cv_triton_runtime::validate_view(out, "device image"), "device view");
    return out;
}

void requireDevice(int device, const char *op)
{
    if (!runtime::deviceAvailable(device))
        CV_Error(Error::StsNotImplemented,
                 cv::format("cv::triton::%s: no usable %s device %d; this module never falls back "
                            "to CPU or cv::cuda",
                            op, runtime::name(), device));
}

void validateStream(std::uintptr_t stream)
{
    if (!stream)
        return;
#ifdef OPENCV_TRITON_HAVE_HIP_ADAPTER
    hipCtx_t current = nullptr, owner = nullptr;
    using GetStreamContext = hipError_t (*)(hipStream_t, hipCtx_t *);
    static const auto getContext =
        reinterpret_cast<GetStreamContext>(dlsym(RTLD_DEFAULT, "hipStreamGetCtx"));
    bool valid = hipCtxGetCurrent(&current) == hipSuccess && current;
    if (valid && getContext)
        valid = getContext(reinterpret_cast<hipStream_t>(stream), &owner) == hipSuccess &&
                owner == current;
    else if (valid)
    {
        // Standard ROCm has a stream-device query but no public stream-context
        // query. Current contextIdentity already admits retained primaries only.
        // Arbitrary same-device private-stream provenance is unsupported and
        // must not be claimed as proven by this device-only fallback.
        hipDevice_t device = -1, actual = -1;
        valid = hipCtxGetDevice(&device) == hipSuccess &&
                hipStreamGetDevice(reinterpret_cast<hipStream_t>(stream), &actual) == hipSuccess &&
                actual == device;
    }
    if (!valid)
#else
    CUcontext current = nullptr, owner = nullptr;
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS ||
        cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &owner) != CUDA_SUCCESS ||
        owner != current)
#endif
        CV_Error(Error::StsBadArg, "stream must belong to the current device context");
}

} // namespace

bool haveDevice(int device)
{
    return runtime::deviceAvailable(device);
}

void resize(const DeviceImage &src, const DeviceImage &dst, Size dsize, int interpolation,
            int device, std::uintptr_t stream)
{
    if (dsize.width <= 0 || dsize.height <= 0)
        CV_Error(Error::StsBadArg, "cv::triton::resize: dsize must be explicit and non-empty");
    resize(src, dst, dsize, 0.0, 0.0, interpolation, device, stream);
}

void resize(const DeviceImage &src, const DeviceImage &dst, Size dsize, double fx, double fy,
            int interpolation, int device, std::uintptr_t stream)
{
    const auto source = view(src, device);
    const auto output = detail::resizeOutput(source, dsize, fx, fy, interpolation);
    if (dst.rows != output.size.height || dst.cols != output.size.width || dst.type != output.type)
        CV_Error(Error::StsBadArg, "cv::triton::resize: dst must have resolved size and type");
    if (src.ptr == dst.ptr)
        CV_Error(Error::StsBadArg, "cv::triton::resize: in-place operation is not supported");
    requireDevice(device, "resize");
    validateStream(stream);
    detail::resizeDevice(source, view(dst, device), dsize, fx, fy, interpolation, device, stream);
}

void cvtColor(const DeviceImage &src, const DeviceImage &dst, int code, int device,
              std::uintptr_t stream)
{
    const auto source = view(src, device);
    const auto output = detail::cvtColorOutput(source, code);
    if (dst.rows != output.size.height || dst.cols != output.size.width || dst.type != output.type)
        CV_Error(Error::StsBadArg,
                 "cv::triton::cvtColor: dst must have the code's output size and type");
    if (src.ptr == dst.ptr)
        CV_Error(Error::StsBadArg, "cv::triton::cvtColor: in-place conversion is not supported");
    requireDevice(device, "cvtColor");
    validateStream(stream);
    cv_triton_color::Request req;
    req.src = view(src, device);
    req.dst = view(dst, device);
    req.code = code;
    detail::cvtColorDevice(req, device, stream);
}

void warpAffine(const DeviceImage &src, const DeviceImage &dst, const double M[6], int flags,
                int borderMode, const Scalar &borderValue, int device, std::uintptr_t stream)
{
    if (flags & ~(INTER_MAX | WARP_INVERSE_MAP))
        CV_Error(Error::StsUnsupportedFormat, "cv::triton::warpAffine: unsupported flags");
    if (!M)
        CV_Error(Error::StsBadArg, "cv::triton::warpAffine: matrix pointer must not be null");
    const int interpolation = flags & INTER_MAX;
    if (interpolation != INTER_NEAREST && interpolation != INTER_LINEAR &&
        interpolation != INTER_CUBIC && interpolation != INTER_AREA)
        CV_Error(Error::StsUnsupportedFormat,
                 cv::format("cv::triton::warpAffine: interpolation %d is not served by this module "
                            "(INTER_NEAREST, INTER_LINEAR, INTER_CUBIC and INTER_AREA); "
                            "INTER_LANCZOS4 is refused, not approximated",
                            interpolation));
    if (interpolation == INTER_CUBIC && borderMode == BORDER_TRANSPARENT)
        CV_Error(Error::StsUnsupportedFormat,
                 "cv::triton::warpAffine: BORDER_TRANSPARENT with INTER_CUBIC is not served");
    const int depth = CV_MAT_DEPTH(src.type);
    if (depth != CV_8U && depth != CV_16U && depth != CV_32F)
        CV_Error(Error::StsUnsupportedFormat,
                 cv::format("cv::triton::warpAffine: depth %d is not served by this module "
                            "(CV_8U, CV_16U and CV_32F)",
                            depth));
    if (dst.type != src.type)
        CV_Error(Error::StsBadArg, "cv::triton::warpAffine: dst must have src's type");
    if (src.ptr == dst.ptr)
        CV_Error(Error::StsBadArg, "cv::triton::warpAffine: in-place operation is not supported");
    requireDevice(device, "warpAffine");
    cv_triton_warp::Request req;
    req.src = view(src, device);
    validateStream(stream);
    req.dst = view(dst, device);
    for (int i = 0; i < 6; ++i)
        req.M[i] = M[i];
    req.interpolation = interpolation;
    req.border_mode = borderMode;
    for (int i = 0; i < 4; ++i)
        req.border[i] = borderValue[i];
    req.inverse_map = (flags & WARP_INVERSE_MAP) != 0;
    detail::warpAffineDevice(req, device, stream);
}

}
} // namespace cv::triton
