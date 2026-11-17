// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright 2026, The FlagOS Contributors.
// Test/perf adapter for the public HIP DeviceImage entries, not a production allocator.
// ROI fixtures share the parent allocation; stream fixtures own streams they create.
// Blocking upload/download delimit host comparisons. Callers retain fixtures in flight.
#include "device.hpp"
#include "opencv2/triton/device.hpp"
#include "../src/entry.hpp"
#include <hip/hip_runtime_api.h>
namespace triton_validation
{
namespace
{
void checked(hipError_t code)
{
    if (code != hipSuccess)
        CV_Error(cv::Error::GpuApiCallError, hipGetErrorString(code));
}

struct Allocation
{
    void *data = nullptr;
    ~Allocation()
    {
        if (data)
            (void)hipFree(data);
    }
};
}

struct Stream::Impl
{
    hipStream_t stream = nullptr;
    bool owned;
    explicit Impl(bool useDefault) : owned(!useDefault)
    {
        if (owned)
            checked(hipStreamCreate(&stream));
    }
    ~Impl()
    {
        if (owned)
            (void)hipStreamDestroy(stream);
    }
};

struct Image::Impl
{
    std::shared_ptr<Allocation> allocation;
    cv::Size dimensions;
    int dtype = 0;
    std::size_t pitch = 0;
    unsigned char *first = nullptr;
};

bool available()
{
    return cv::triton::haveDevice(0);
}
const char *backend()
{
    return "hip";
}
Stream::Stream(bool useDefault) : impl(std::make_shared<Impl>(useDefault))
{
}
void Stream::wait() const
{
    checked(hipStreamSynchronize(impl->stream));
}

void Stream::waitFor(const Stream &producer) const
{
    hipEvent_t event = nullptr;
    checked(hipEventCreateWithFlags(&event, hipEventDisableTiming));
    hipError_t code = hipEventRecord(event, producer.impl->stream);
    if (code == hipSuccess)
        code = hipStreamWaitEvent(impl->stream, event, 0);
    (void)hipEventDestroy(event);
    checked(code);
}

Image::Image() : impl(std::make_shared<Impl>())
{
}
Image::Image(cv::Size size, int type) : Image()
{
    create(size, type);
}
Image::Image(const Image &parent, cv::Rect roi) : Image()
{
    CV_Assert(roi.width > 0 && roi.height > 0 &&
              (roi & cv::Rect(cv::Point(), parent.size())) == roi);
    *impl = *parent.impl;
    impl->dimensions = roi.size();
    impl->first += roi.y * impl->pitch + roi.x * CV_ELEM_SIZE(impl->dtype);
}

void Image::create(cv::Size size, int type)
{
    if (impl->first && impl->dimensions == size && impl->dtype == type)
        return;
    CV_Assert(size.width > 0 && size.height > 0);
    auto next = std::make_shared<Impl>();
    next->dimensions = size;
    next->dtype = type;
    next->allocation = std::make_shared<Allocation>();
    checked(hipMallocPitch(&next->allocation->data, &next->pitch,
                           static_cast<std::size_t>(size.width) * CV_ELEM_SIZE(type), size.height));
    next->first = static_cast<unsigned char *>(next->allocation->data);
    impl = next;
}

void Image::upload(const cv::Mat &host)
{
    create(host.size(), host.type());
    checked(hipMemcpy2D(impl->first, impl->pitch, host.data, host.step,
                        static_cast<std::size_t>(host.cols) * host.elemSize(), host.rows,
                        hipMemcpyHostToDevice));
}

cv::Mat Image::download() const
{
    cv::Mat host(size(), type());
    checked(hipMemcpy2D(host.data, host.step, impl->first, impl->pitch,
                        static_cast<std::size_t>(host.cols) * host.elemSize(), host.rows,
                        hipMemcpyDeviceToHost));
    return host;
}

cv::Size Image::size() const
{
    return impl->dimensions;
}
int Image::type() const
{
    return impl->dtype;
}
std::size_t Image::step() const
{
    return impl->pitch;
}
std::uintptr_t Image::address() const
{
    return reinterpret_cast<std::uintptr_t>(impl->first);
}

namespace
{
cv::triton::DeviceImage image(const Image &value)
{
    cv::triton::DeviceImage out;
    out.ptr = value.address();
    out.rows = value.size().height;
    out.cols = value.size().width;
    out.type = value.type();
    out.step = value.step();
    return out;
}

cv_triton_runtime::DeviceView view(const Image &value)
{
    cv_triton_runtime::DeviceView out;
    out.ptr = value.address();
    out.rows = value.size().height;
    out.cols = value.size().width;
    out.channels = CV_MAT_CN(value.type());
    out.step = value.step();
    const int depth = CV_MAT_DEPTH(value.type());
    switch (depth)
    {
    case CV_8U:
        out.type = cv_triton_runtime::ElemType::U8;
        break;
    case CV_16U:
        out.type = cv_triton_runtime::ElemType::U16;
        break;
    case CV_16S:
        out.type = cv_triton_runtime::ElemType::S16;
        break;
    case CV_32F:
        out.type = cv_triton_runtime::ElemType::F32;
        break;
    default:
        CV_Error(cv::Error::StsUnsupportedFormat, "unsupported validation image depth");
    }
    return out;
}

void rejectAlias(const Image &src, const Image &dst)
{
    if (!src.address() || !dst.address())
        return;
    const auto source = view(src), destination = view(dst);
    const auto sourceStatus = cv_triton_runtime::validate_view(source, "src");
    if (!sourceStatus.ok())
        CV_Error(cv::Error::StsBadArg, sourceStatus.message);
    const auto destinationStatus = cv_triton_runtime::validate_view(destination, "dst");
    if (!destinationStatus.ok())
        CV_Error(cv::Error::StsBadArg, destinationStatus.message);
    if (cv_triton_runtime::views_overlap(source, destination))
        CV_Error(cv::Error::StsBadArg, "overlapping images are not supported");
}
}

void resize(const Image &src, Image &dst, cv::Size size, double fx, double fy, int interpolation,
            Stream &stream)
{
    const auto info = cv::triton::detail::resizeOutput(view(src), size, fx, fy, interpolation);
    rejectAlias(src, dst);
    dst.create(info.size, info.type);
    cv::triton::resize(image(src), image(dst), size, fx, fy, interpolation, 0,
                       reinterpret_cast<std::uintptr_t>(stream.impl->stream));
}

void cvtColor(const Image &src, Image &dst, int code, Stream &stream)
{
    const auto info = cv::triton::detail::cvtColorOutput(view(src), code);
    rejectAlias(src, dst);
    dst.create(info.size, info.type);
    cv::triton::cvtColor(image(src), image(dst), code, 0,
                         reinterpret_cast<std::uintptr_t>(stream.impl->stream));
}

void warpAffine(const Image &src, Image &dst, const cv::Mat &matrix, cv::Size size, int flags,
                int border, const cv::Scalar &value, Stream &stream)
{
    const auto request = cv::triton::detail::warpRequest(view(src), matrix, flags, border, value);
    const auto info = cv::triton::detail::warpOutput(request, size);
    rejectAlias(src, dst);
    dst.create(info.size, info.type);
    cv::triton::warpAffine(image(src), image(dst), request.M, flags, border, value, 0,
                           reinterpret_cast<std::uintptr_t>(stream.impl->stream));
}
}
