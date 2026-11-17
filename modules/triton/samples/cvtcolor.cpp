// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// CUDA GpuMat example: upload -> cv::triton::cvtColor -> download.
//
// Supported code/depth combinations are listed in doc/cvtcolor.markdown; support for
// one code at CV_8U does not imply support at CV_16U or CV_32F. Unsupported requests
// raise cv::Error::StsUnsupportedFormat without fallback.
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/triton.hpp>
int main(int argc, char **argv)
{
    if (!cv::triton::haveBackend())
    {
        std::cerr << "No Triton backend for this device.\n";
        return 1;
    }
    std::cout << "backend: " << cv::triton::backendName() << "\n";

    cv::Mat host = (argc > 1) ? cv::imread(argv[1], cv::IMREAD_COLOR)
                              : cv::Mat(720, 1280, CV_8UC3, cv::Scalar(64, 128, 192));
    if (host.empty())
    {
        std::cerr << "failed to read input\n";
        return 1;
    }

    cv::cuda::GpuMat src(host), dst;
    cv::cuda::Stream stream;

    // Asynchronous: the call returns after submitting, and src/dst must stay alive until the
    // work completes. The destination is created if it is empty and reused if it already has
    // the right size and type.
    cv::triton::cvtColor(src, dst, cv::COLOR_BGR2GRAY, stream);
    stream.waitForCompletion(); // wait before reading the result

    cv::Mat out;
    dst.download(out);
    std::cout << "converted " << host.size() << " BGR -> GRAY " << out.size() << " ("
              << out.channels() << " channel)\n";
    if (argc > 2)
        cv::imwrite(argv[2], out);
    return 0;
}
