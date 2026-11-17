// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright 2026, The FlagOS Contributors.
// Thin CUDA/GpuMat adapter. Native ownership and output allocation live here;
// metadata/admission, plans and status translation live in the common private entry/provider.
// No image copies, replacement streams, numerical rules or kernel selection are added here.
#include "precomp.hpp"
#include <cuda.h>
#include <limits>
#include "runtime/cvtcolor.hpp"
#include "runtime/warpaffine.hpp"
#include "runtime/runtime.hpp"
#include "entry.hpp"
namespace cv
{
namespace triton
{

bool haveBackend()
{
    // Pure query: no device memory is allocated and no work is submitted. Whether the
    // compile toolchain is present matters only on a cache miss and is reported then.
    if (cuda::getCudaEnabledDeviceCount() < 1)
        return false;
    return runtime::deviceAvailable(cuda::getDevice());
}

String backendName()
{
    return haveBackend() ? String(runtime::name()) : String();
}

void releaseGraphResources()
{
    // The whole graph lifecycle responsibility of this module: the runtime pins the compiled
    // module and its tables when a launch is recorded into a capturing stream, and cannot know
    // when the caller's graph is destroyed. See the declaration for the contract and the cost.
    detail::releaseCapturePins();
}

namespace
{

ImageView toImageView(const cuda::GpuMat &m)
{
    Size whole;
    Point ofs;
    m.locateROI(whole, ofs);

    ImageView v;
    v.ptr = reinterpret_cast<std::uintptr_t>(m.cudaPtr());
    v.rows = m.rows;
    v.cols = m.cols;
    v.type = m.type();
    v.step = m.step;
    v.whole_rows = whole.height;
    v.whole_cols = whole.width;
    v.offset_y = ofs.y;
    v.offset_x = ofs.x;
    return v;
}

// Validate actual CUDA ownership, rather than labelling every pointer with the
// caller's selected device. Query before OutputArray::create so even a resized
// foreign destination is rejected without touching its allocation.
void checkCudaBuffer(const cuda::GpuMat &m, const char *name, bool layout)
{
    if (m.empty())
        CV_Error(Error::StsBadArg, String(name) + " is empty");
    CUcontext current = nullptr, owner = nullptr;
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || !current ||
        cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT,
                              reinterpret_cast<CUdeviceptr>(m.cudaPtr())) != CUDA_SUCCESS ||
        owner != current)
        CV_Error(Error::StsBadArg,
                 String(name) + " must belong to the current CUDA context/device");
    if (!layout)
        return;
    CUdeviceptr base = 0;
    size_t bytes = 0;
    if (cuMemGetAddressRange(&base, &bytes, reinterpret_cast<CUdeviceptr>(m.cudaPtr())) !=
        CUDA_SUCCESS)
        CV_Error(Error::StsBadArg, String(name) + " has no verifiable CUDA allocation range");
    const auto view = detail::toDeviceView(toImageView(m), cuda::getDevice(),
                                           static_cast<std::uintptr_t>(base), bytes);
    const auto checked = cv_triton_runtime::validate_view(view, name);
    if (!checked.ok())
        CV_Error(Error::StsBadArg, checked.message);
}

void checkCudaArguments(const cuda::GpuMat &src, OutputArray dst, cuda::Stream &stream)
{
    if (src.empty())
        CV_Error(Error::StsBadArg, "src is empty");
    CUcontext current = nullptr;
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS)
        CV_Error(Error::StsBadArg, "cannot query the current CUDA context");
    // OpenCV's current-device convention also applies on a new host thread.
    // Initialize its selected runtime context before checking borrowed buffers;
    // never switch an already-current driver context to make a foreign pointer fit.
    if (!current)
        cuda::setDevice(cuda::getDevice());
    checkCudaBuffer(src, "src", true);
    if (dst.kind() == _InputArray::CUDA_GPU_MAT)
    {
        const cuda::GpuMat pre = dst.getGpuMat();
        if (!pre.empty())
            checkCudaBuffer(pre, "dst", false);
    }
    const CUstream handle = reinterpret_cast<CUstream>(cuda::StreamAccessor::getStream(stream));
    if (handle)
    {
        CUcontext owner = nullptr;
        if (cuCtxGetCurrent(&current) != CUDA_SUCCESS ||
            cuStreamGetCtx(handle, &owner) != CUDA_SUCCESS || owner != current)
            CV_Error(Error::StsBadArg, "stream must belong to the current CUDA context/device");
    }
}

cv_triton_runtime::ContextId currentIdentity()
{
    cv_triton_runtime::ContextId id;
    CUcontext context = nullptr;
    CUdevice device = 0;
    unsigned long long generation = 0;
    if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context ||
        cuCtxGetDevice(&device) != CUDA_SUCCESS || cuCtxGetId(context, &generation) != CUDA_SUCCESS)
        CV_Error(Error::StsBadArg, "cv::triton: cannot identify the current CUDA context");
    id.backend = cv_triton_runtime::Backend::CUDA;
    id.device = static_cast<int>(device);
    id.native = reinterpret_cast<std::uintptr_t>(context);
    id.generation = generation;
    return id;
}

cv_triton_runtime::DeviceView deviceView(const cuda::GpuMat &image,
                                         const cv_triton_runtime::ContextId &context)
{
    auto view = detail::toDeviceView(toImageView(image), context.device,
                                     reinterpret_cast<std::uintptr_t>(image.datastart),
                                     static_cast<std::size_t>(image.dataend - image.datastart));
    view.ctx = context;
    return view;
}

void rejectOverlap(const cuda::GpuMat &src, OutputArray dst, const char *operation)
{
    if (dst.kind() == _InputArray::CUDA_GPU_MAT)
    {
        const cuda::GpuMat before = dst.getGpuMat();
        if (!before.empty())
        {
            checkCudaBuffer(before, "dst", true);
            const auto context = currentIdentity();
            if (cv_triton_runtime::views_overlap(deviceView(src, context),
                                                 deviceView(before, context)))
                CV_Error(
                    Error::StsBadArg,
                    cv::format("cv::triton::%s: overlapping images are not supported", operation));
        }
    }
}

cuda::GpuMat prepareDestination(OutputArray output, const detail::OutputInfo &info)
{
    output.create(info.size, info.type);
    cuda::GpuMat dst = output.getGpuMat();
    checkCudaBuffer(dst, "dst", true); // create may have changed both address and layout
    return dst;
}

} // namespace

void resize(InputArray input, OutputArray output, Size dsize, double fx, double fy,
            int interpolation, cuda::Stream &stream)
{
    CV_Assert(input.kind() == _InputArray::CUDA_GPU_MAT);
    const cuda::GpuMat src = input.getGpuMat();
    checkCudaArguments(src, output, stream);
    const auto context = currentIdentity();
    const auto source = deviceView(src, context);
    const auto info = detail::resizeOutput(source, dsize, fx, fy, interpolation);
    // Check the existing output before create can replace its storage and hide overlap.
    rejectOverlap(src, output, "resize");
    const cuda::GpuMat dst = prepareDestination(output, info);
    const auto native = reinterpret_cast<std::uintptr_t>(cuda::StreamAccessor::getStream(stream));
    // Preserve original fx/fy when dsize was omitted; do not derive them from rounded sizes.
    detail::resizeDevice(source, deviceView(dst, context), dsize.empty() ? Size() : info.size,
                         dsize.empty() ? fx : 0.0, dsize.empty() ? fy : 0.0, interpolation,
                         context.device, native);
}

void resize(const cuda::GpuMat &src, cuda::GpuMat &dst, Size dsize, double fx, double fy,
            int interpolation, cuda::Stream &stream)
{
    resize(InputArray(src), OutputArray(dst), dsize, fx, fy, interpolation, stream);
}

void cvtColor(InputArray input, OutputArray output, int code, cuda::Stream &stream)
{
    CV_Assert(input.kind() == _InputArray::CUDA_GPU_MAT);
    const cuda::GpuMat src = input.getGpuMat();
    checkCudaArguments(src, output, stream);
    const auto context = currentIdentity();
    cv_triton_color::Request request;
    request.src = deviceView(src, context);
    request.code = code;
    const auto info = detail::cvtColorOutput(request.src, code);
    rejectOverlap(src, output, "cvtColor");
    const cuda::GpuMat dst = prepareDestination(output, info);
    request.dst = deviceView(dst, context);
    detail::cvtColorDevice(
        request, context.device,
        reinterpret_cast<std::uintptr_t>(cuda::StreamAccessor::getStream(stream)));
}

void warpAffine(InputArray input, OutputArray output, InputArray matrix, Size dsize, int flags,
                int borderMode, const Scalar &borderValue, cuda::Stream &stream)
{
    CV_Assert(input.kind() == _InputArray::CUDA_GPU_MAT);
    const cuda::GpuMat src = input.getGpuMat();
    checkCudaArguments(src, output, stream);
    const auto context = currentIdentity();
    auto request =
        detail::warpRequest(deviceView(src, context), matrix, flags, borderMode, borderValue);
    const auto info = detail::warpOutput(request, dsize);
    rejectOverlap(src, output, "warpAffine");
    const cuda::GpuMat dst = prepareDestination(output, info);
    request.dst = deviceView(dst, context);
    detail::warpAffineDevice(
        request, context.device,
        reinterpret_cast<std::uintptr_t>(cuda::StreamAccessor::getStream(stream)));
}

void cvtColor(const cuda::GpuMat &src, cuda::GpuMat &dst, int code, cuda::Stream &stream)
{
    cvtColor(InputArray(src), OutputArray(dst), code, stream);
}

void warpAffine(const cuda::GpuMat &src, cuda::GpuMat &dst, InputArray matrix, Size dsize,
                int flags, int borderMode, const Scalar &borderValue, cuda::Stream &stream)
{
    warpAffine(InputArray(src), OutputArray(dst), matrix, dsize, flags, borderMode, borderValue,
               stream);
}

}
} // namespace cv::triton
