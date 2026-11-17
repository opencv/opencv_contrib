// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright 2026, The FlagOS Contributors.
// HIP DeviceImage example: caller-owned buffers and an explicit stream, no Torch or GpuMat.
#include <hip/hip_runtime_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/triton/device.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
static void checked(hipError_t code)
{
    if (code != hipSuccess)
        throw std::runtime_error(hipGetErrorString(code));
}

struct Buffer
{
    void *ptr = nullptr;
    std::size_t bytes;
    explicit Buffer(std::size_t size) : bytes(size)
    {
        checked(hipMalloc(&ptr, bytes));
    }
    ~Buffer()
    {
        if (ptr)
            (void)hipFree(ptr);
    }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
};

int main(int argc, char **argv)
{
    hipStream_t stream = nullptr;
    try
    {
        const std::string op = argc > 1 ? argv[1] : "resize";
        if (op != "resize" && op != "warp" && op != "cvtcolor")
            throw std::runtime_error("usage: example_triton_device [resize|warp|cvtcolor]");
        if (!cv::triton::haveDevice(0))
            throw std::runtime_error("no usable HIP device");
        checked(hipStreamCreate(&stream));
        const int rows = 96, cols = 128;
        cv::Mat host(rows, cols, op == "cvtcolor" ? CV_8UC3 : CV_8UC1);
        for (int y = 0; y < rows; ++y)
            for (int x = 0; x < cols; ++x)
                for (int c = 0; c < host.channels(); ++c)
                    host.ptr<uchar>(y)[x * host.channels() + c] =
                        static_cast<uchar>((x * 3 + y * 7 + c * 59) % 256);
        const cv::Size size = op == "resize" ? cv::Size(63, 47) : host.size();
        Buffer input(host.total() * host.elemSize()), output(static_cast<std::size_t>(size.area()));
        checked(hipMemcpyAsync(input.ptr, host.data, input.bytes, hipMemcpyHostToDevice, stream));
        cv::triton::DeviceImage src, dst;
        src.ptr = reinterpret_cast<std::uintptr_t>(input.ptr);
        src.rows = rows;
        src.cols = cols;
        src.type = host.type();
        src.step = host.step;
        dst.ptr = reinterpret_cast<std::uintptr_t>(output.ptr);
        dst.rows = size.height;
        dst.cols = size.width;
        dst.type = CV_8UC1;
        dst.step = size.width;
        cv::Mat reference;
        if (op == "resize")
        {
            cv::triton::resize(src, dst, size, cv::INTER_AREA, 0,
                               reinterpret_cast<std::uintptr_t>(stream));
            cv::resize(host, reference, size, 0, 0, cv::INTER_AREA);
        }
        else if (op == "cvtcolor")
        {
            cv::triton::cvtColor(src, dst, cv::COLOR_BGR2GRAY, 0,
                                 reinterpret_cast<std::uintptr_t>(stream));
            cv::cvtColor(host, reference, cv::COLOR_BGR2GRAY);
        }
        else
        {
            const double values[6] = {.9, .43589, -7, -.43589, .9, 21};
            cv::triton::warpAffine(src, dst, values, cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                                   cv::BORDER_REPLICATE, cv::Scalar(), 0,
                                   reinterpret_cast<std::uintptr_t>(stream));
            cv::Mat M(2, 3, CV_64F, const_cast<double *>(values)), wide;
            // The NEAREST/LINEAR reference is the selected CPU vector path.
            cv::warpAffine(host, wide, M, cv::Size(256, size.height),
                           cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
            reference = wide(cv::Rect(0, 0, size.width, size.height)).clone();
        }
        cv::Mat got(size, CV_8UC1);
        checked(hipMemcpyAsync(got.data, output.ptr, output.bytes, hipMemcpyDeviceToHost, stream));
        checked(hipStreamSynchronize(stream));
        const double error = cv::norm(got, reference, cv::NORM_INF);
        std::cout << op << ": " << size << ", max error=" << error << std::endl;
        checked(hipStreamDestroy(stream));
        stream = nullptr;
        return error == 0 ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        if (stream)
        {
            (void)hipStreamSynchronize(stream);
            (void)hipStreamDestroy(stream);
        }
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
