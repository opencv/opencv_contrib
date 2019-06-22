// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv { namespace augment {

GaussianBlur::GaussianBlur(Size minKernelSize, Size maxKernelSize, double minSigmaX, double maxSigmaX, double minSigmaY, double maxSigmaY)
{
    this->minKernelSize = minKernelSize;
    this->maxKernelSize = maxKernelSize;
    this->minSigmaX = minSigmaX;
    this->minSigmaY = minSigmaY;
    this->maxSigmaX = maxSigmaX;
    this->maxSigmaY = maxSigmaY;
    sameXY = false;
}

GaussianBlur::GaussianBlur(int minKernelSize, int maxKernelSize, double minSigma, double maxSigma)
{
    this->minKernelSize = Size(minKernelSize, minKernelSize);
    this->maxKernelSize = Size(maxKernelSize, maxKernelSize);
    minSigmaX = minSigmaY = minSigma;
    maxSigmaX = maxSigmaY = maxSigma;
    sameXY = true;
}

GaussianBlur::GaussianBlur(int kernelSize, double sigma)
{
    minKernelSize = maxKernelSize = Size(kernelSize, kernelSize);
    minSigmaX = minSigmaY = maxSigmaY = maxSigmaX = sigma;
    sameXY = true;
}

GaussianBlur::GaussianBlur(Size kernelSize, double sigmaX, double sigmaY)
{
    minKernelSize = maxKernelSize = kernelSize;
    minSigmaX = maxSigmaX = sigmaX;
    minSigmaY = maxSigmaY = sigmaY;
    sameXY = false;
}

void GaussianBlur::init(const Mat& image)
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

}}
