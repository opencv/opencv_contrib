// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright 2026, The FlagOS Contributors.
#include "device.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/triton.hpp"
namespace triton_validation
{
struct Stream::Impl
{
    cv::cuda::Stream owned;
    cv::cuda::Stream *stream;
    explicit Impl(bool useDefault) : owned(cv::cuda::Stream::Null()), stream(nullptr)
    {
        if (useDefault)
            stream = &cv::cuda::Stream::Null();
        else
        {
            owned = cv::cuda::Stream();
            stream = &owned;
        }
    }
};
struct Image::Impl
{
    cv::cuda::GpuMat image;
};
bool available()
{
    if (cv::cuda::getCudaEnabledDeviceCount() < 1)
        return false;
    cv::cuda::setDevice(cv::cuda::getDevice());
    return cv::triton::haveBackend();
}
const char *backend()
{
    return "cuda";
}
Stream::Stream(bool useDefault) : impl(std::make_shared<Impl>(useDefault))
{
}
void Stream::wait() const
{
    impl->stream->waitForCompletion();
}
void Stream::waitFor(const Stream &producer) const
{
    cv::cuda::Event event(cv::cuda::Event::DISABLE_TIMING);
    event.record(*producer.impl->stream);
    impl->stream->waitEvent(event);
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
    impl->image = parent.impl->image(roi);
}
void Image::upload(const cv::Mat &host)
{
    impl->image.upload(host);
}
cv::Mat Image::download() const
{
    cv::Mat host;
    impl->image.download(host);
    return host;
}
void Image::create(cv::Size size, int type)
{
    impl->image.create(size, type);
}
cv::Size Image::size() const
{
    return impl->image.size();
}
int Image::type() const
{
    return impl->image.type();
}
std::size_t Image::step() const
{
    return impl->image.step;
}
std::uintptr_t Image::address() const
{
    return reinterpret_cast<std::uintptr_t>(impl->image.cudaPtr());
}
void resize(const Image &src, Image &dst, cv::Size size, double fx, double fy, int interpolation,
            Stream &stream)
{
    cv::triton::resize(src.impl->image, dst.impl->image, size, fx, fy, interpolation,
                       *stream.impl->stream);
}
void cvtColor(const Image &src, Image &dst, int code, Stream &stream)
{
    cv::triton::cvtColor(src.impl->image, dst.impl->image, code, *stream.impl->stream);
}
void warpAffine(const Image &src, Image &dst, const cv::Mat &matrix, cv::Size size, int flags,
                int border, const cv::Scalar &value, Stream &stream)
{
    cv::triton::warpAffine(src.impl->image, dst.impl->image, matrix, size, flags, border, value,
                           *stream.impl->stream);
}
}
