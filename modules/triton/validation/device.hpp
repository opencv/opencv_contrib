// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright 2026, The FlagOS Contributors.
// Test/perf support only; never installed or linked into the production module.
// Backend source is selected once by CMake. Calls go through the public operator API:
// GpuMat overloads on CUDA and DeviceImage overloads on HIP.
#ifndef OPENCV_TRITON_VALIDATION_DEVICE_HPP
#define OPENCV_TRITON_VALIDATION_DEVICE_HPP
#include <memory>
#include <cstdint>
#include "opencv2/core.hpp"
namespace triton_validation
{
bool available();
const char *backend();
class Stream
{
  public:
    struct Impl;
    explicit Stream(bool useDefault = false);
    // Wait on the host for this stream to complete.
    void wait() const;
    // Add a device-side dependency; the host does not wait for the producer.
    void waitFor(const Stream &producer) const;
    std::shared_ptr<Impl> impl;
};
class Image
{
  public:
    struct Impl;
    Image();
    Image(cv::Size size, int type);
    Image(const Image &parent, cv::Rect roi); // shared allocation, independent view/pitch
    // Blocking transfers used for setup/checking, outside performance cycles.
    void upload(const cv::Mat &host);
    cv::Mat download() const;
    // Reuse matching storage; otherwise allocate the requested size/type.
    void create(cv::Size size, int type);
    cv::Size size() const;
    int type() const;
    std::size_t step() const;
    std::uintptr_t address() const;
    std::shared_ptr<Impl> impl;
};
void resize(const Image &src, Image &dst, cv::Size size, double fx, double fy, int interpolation,
            Stream &stream);
void cvtColor(const Image &src, Image &dst, int code, Stream &stream);
void warpAffine(const Image &src, Image &dst, const cv::Mat &matrix, cv::Size size, int flags,
                int border, const cv::Scalar &value, Stream &stream);
}
#endif
