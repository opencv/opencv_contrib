// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv { namespace augment {

GaussianBlur::GaussianBlur(Size minKernelSize, Size maxKernelSize, float minSigmaX, float maxSigmaX, float minSigmaY, float maxSigmaY)
{
    _minKernelSize = minKernelSize;
    _maxKernelSize = maxKernelSize;
    _minSigmaX = minSigmaX;
    _minSigmaY = minSigmaY;
    _maxSigmaX = maxSigmaX;
    _maxSigmaY = maxSigmaY;
    _sameXY = false;
}

GaussianBlur::GaussianBlur(int minKernelSize, int maxKernelSize, float minSigma, float maxSigma)
{
    _minKernelSize = Size(minKernelSize, minKernelSize);
    _maxKernelSize = Size(maxKernelSize, maxKernelSize);
    _minSigmaX = _minSigmaY = minSigma;
    _maxSigmaX = _maxSigmaY = maxSigma;
    _sameXY = true;
}

GaussianBlur::GaussianBlur(int kernelSize, float sigma)
{
    _minKernelSize = _maxKernelSize = Size(kernelSize, kernelSize);
    _minSigmaX = _minSigmaY = _maxSigmaY = _maxSigmaX = sigma;
    _sameXY = true;
}

GaussianBlur::GaussianBlur(Size kernelSize, float sigmaX, float sigmaY)
{
    _minKernelSize = _maxKernelSize = kernelSize;
    _minSigmaX = _maxSigmaX = sigmaX;
    _minSigmaY = _maxSigmaY = sigmaY;
    _sameXY = false;
}

void GaussianBlur::init(const Mat&)
{
    _kernelSize.height = Transform::rng.uniform(_minKernelSize.height / 2, _maxKernelSize.height / 2 + _maxKernelSize.height % 2) * 2 + 1; //generate only random odd numbers
    _kernelSize.width = _sameXY? _kernelSize.height : Transform::rng.uniform(_minKernelSize.width / 2, _maxKernelSize.width / 2 + _maxKernelSize.width % 2) * 2 + 1; //generate only random odd numbers
    _sigmaX = Transform::rng.uniform(_minSigmaX, _maxSigmaX);
    _sigmaY = _sameXY? _sigmaX : Transform::rng.uniform(_minSigmaY, _maxSigmaY);
}

void GaussianBlur::image(InputArray src, OutputArray dst)
{
    cv::GaussianBlur(src, dst, _kernelSize, _sigmaX, _sigmaY);
}

}}
