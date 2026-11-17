// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright 2026, The FlagOS Contributors.
// CUDA GpuMat example: upload -> cv::triton::resize -> download.
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

    // Supported subset: CV_8U or CV_32F, C1/C3/C4, explicit dsize or fx/fy,
    // copy, nearest, linear, cubic, integer and fractional area -- see doc/resize.markdown.
    cv::Mat host = (argc > 1) ? cv::imread(argv[1], cv::IMREAD_COLOR)
                              : cv::Mat(720, 1280, CV_8UC3, cv::Scalar(64, 128, 192));
    if (host.empty())
    {
        std::cerr << "failed to read input\n";
        return 1;
    }

    cv::cuda::GpuMat src(host);
    cv::cuda::GpuMat dst(host.rows / 2, host.cols / 2, host.type());
    cv::cuda::Stream stream;

    cv::triton::resize(src, dst, dst.size(), 0, 0, cv::INTER_LINEAR, stream);
    stream.waitForCompletion(); // async call: wait before using the result

    cv::Mat out;
    dst.download(out);
    std::cout << "resized " << host.size() << " -> " << out.size() << "\n";
    if (argc > 2)
        cv::imwrite(argv[2], out);
    return 0;
}
