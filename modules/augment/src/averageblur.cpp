// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv { namespace augment {

AverageBlur::AverageBlur(Size minKernelSize, Size maxKernelSize)
{
    this->minKernelSize = minKernelSize;
    this->maxKernelSize = maxKernelSize;
    sameXY = false;
}

AverageBlur::AverageBlur(int minKernelSize, int maxKernelSize)
{
    this->minKernelSize = Size(minKernelSize, minKernelSize);
    this->maxKernelSize = Size(maxKernelSize, maxKernelSize);
    sameXY = true;
}

AverageBlur::AverageBlur(int kernelSize)
{
    minKernelSize = maxKernelSize = Size(kernelSize, kernelSize);
    sameXY = true;
}

AverageBlur::AverageBlur(Size kernelSize)
{
    minKernelSize = maxKernelSize = kernelSize;
    sameXY = false;
}

void AverageBlur::init(const Mat& image)
{
    kernelSize.height = Transform::rng.uniform(minKernelSize.height / 2, maxKernelSize.height / 2 + maxKernelSize.height % 2) * 2 + 1; //generate only random odd numbers
    kernelSize.width = sameXY? kernelSize.height : Transform::rng.uniform(minKernelSize.width / 2, maxKernelSize.width / 2 + maxKernelSize.width % 2) * 2 + 1; //generate only random odd numbers
}

void AverageBlur::image(InputArray src, OutputArray dst)
{
    cv::blur(src, dst, kernelSize);
}

}}
