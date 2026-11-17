// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright 2026, The FlagOS Contributors.
// CUDA GpuMat example: upload -> cv::triton::warpAffine -> download.
//
// The 2x3 matrix is interpreted exactly as cv::warpAffine interprets it: the forward map,
// inverted internally, unless cv::WARP_INVERSE_MAP is set. The interpolation flags, dtypes and
// border modes are listed in doc/warpaffine.markdown -- INTER_NEAREST, INTER_LINEAR,
// INTER_CUBIC and INTER_AREA (LINEAR alias), CV_8U / CV_16U / CV_32F, channels 1/3/4. Inputs outside
// that list (INTER_LANCZOS4, other depths) raise cv::Error::StsUnsupportedFormat rather than
// falling back, so an unsupported request is a visible error, not a silent change of result.
#include <cmath>
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
                              : cv::Mat(480, 640, CV_8UC3, cv::Scalar(64, 128, 192));
    if (host.empty())
    {
        std::cerr << "failed to read input\n";
        return 1;
    }

    // Rotate by 33 degrees about the image centre, scaled by 0.8. Written out rather than
    // built with cv::getRotationMatrix2D: that lives in the geometry module in OpenCV 5.x, and a
    // minimal example should not pull in a module the operation itself does not need.
    const double angle = 33.0 * CV_PI / 180.0, scale = 0.8;
    const double cx = host.cols * 0.5, cy = host.rows * 0.5;
    const double a = std::cos(angle) * scale, b = std::sin(angle) * scale;
    cv::Mat M(2, 3, CV_64F);
    M.at<double>(0, 0) = a;
    M.at<double>(0, 1) = -b;
    M.at<double>(0, 2) = (1 - a) * cx + b * cy;
    M.at<double>(1, 0) = b;
    M.at<double>(1, 1) = a;
    M.at<double>(1, 2) = (1 - a) * cy - b * cx;

    cv::cuda::GpuMat src(host), dst;
    cv::cuda::Stream stream;

    // Asynchronous: the call returns after submitting, and src/dst must stay alive until the
    // work completes. The destination is created if it is empty and reused if it already has
    // the right size and type. The border value is a Scalar with one component per channel, and
    // it keeps full precision, so a CV_32F border of 0.5 would stay 0.5.
    cv::triton::warpAffine(src, dst, M, host.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                           cv::Scalar(0, 0, 0), stream);
    stream.waitForCompletion(); // wait before reading the result

    cv::Mat out;
    dst.download(out);
    std::cout << "warped " << host.size() << " by 33 deg / 0.8 -> " << out.size() << " ("
              << out.channels() << " channel)\n";
    if (argc > 2)
        cv::imwrite(argv[2], out);
    return 0;
}
