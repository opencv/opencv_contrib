// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv { namespace augment {

GaussianBlur::GaussianBlur(Size _minKernelSize, Size _maxKernelSize, float _minSigmaX, float _maxSigmaX, float _minSigmaY, float _maxSigmaY)
{
    minKernelSize = _minKernelSize;
    maxKernelSize = _maxKernelSize;
    minSigmaX = _minSigmaX;
    minSigmaY = _minSigmaY;
    maxSigmaX = _maxSigmaX;
    maxSigmaY = _maxSigmaY;
    sameXY = false;
}

GaussianBlur::GaussianBlur(int _minKernelSize, int _maxKernelSize, float _minSigma, float _maxSigma)
{
    minKernelSize = Size(_minKernelSize, _minKernelSize);
    maxKernelSize = Size(_maxKernelSize, _maxKernelSize);
    minSigmaX = minSigmaY = _minSigma;
    maxSigmaX = maxSigmaY = _maxSigma;
    sameXY = true;
}

GaussianBlur::GaussianBlur(int _kernelSize, float sigma)
{
    minKernelSize = maxKernelSize = Size(_kernelSize, _kernelSize);
    minSigmaX = minSigmaY = maxSigmaY = maxSigmaX = sigma;
    sameXY = true;
}

GaussianBlur::GaussianBlur(Size _kernelSize, float _sigmaX, float _sigmaY)
{
    minKernelSize = maxKernelSize = _kernelSize;
    minSigmaX = maxSigmaX = _sigmaX;
    minSigmaY = maxSigmaY = _sigmaY;
    sameXY = false;
}

void GaussianBlur::init(const Mat&)
{
    kernelSize.height = Transform::rng.uniform(minKernelSize.height / 2, maxKernelSize.height / 2 + maxKernelSize.height % 2) * 2 + 1; //generate only random odd numbers
    kernelSize.width = sameXY? kernelSize.height : Transform::rng.uniform(minKernelSize.width / 2, maxKernelSize.width / 2 + maxKernelSize.width % 2) * 2 + 1; //generate only random odd numbers
    sigmaX = Transform::rng.uniform(minSigmaX, maxSigmaX);
    sigmaY = sameXY? sigmaX : Transform::rng.uniform(minSigmaY, maxSigmaY);
}

void GaussianBlur::image(InputArray src, OutputArray dst)
{
    cv::GaussianBlur(src, dst, kernelSize, sigmaX, sigmaY);
}

Rect2f GaussianBlur::rectangle(const Rect2f& src)
{
    return src;
}

}}
