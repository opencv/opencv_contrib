// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv { namespace augment {

AverageBlur::AverageBlur(Size _minKernelSize, Size _maxKernelSize)
{
    minKernelSize = _minKernelSize;
    maxKernelSize = _maxKernelSize;
    sameXY = false;
}

AverageBlur::AverageBlur(int _minKernelSize, int _maxKernelSize)
{
    minKernelSize = Size(_minKernelSize, _minKernelSize);
    maxKernelSize = Size(_maxKernelSize, _maxKernelSize);
    sameXY = true;
}

AverageBlur::AverageBlur(int _kernelSize)
{
    minKernelSize = maxKernelSize = Size(_kernelSize, _kernelSize);
    sameXY = true;
}

AverageBlur::AverageBlur(Size _kernelSize)
{
    minKernelSize = maxKernelSize = _kernelSize;
    sameXY = false;
}

void AverageBlur::init(const Mat&)
{
    kernelSize.height = Transform::rng.uniform(minKernelSize.height / 2, maxKernelSize.height / 2 + maxKernelSize.height % 2) * 2 + 1; //generate only random odd numbers
    kernelSize.width = sameXY? kernelSize.height : Transform::rng.uniform(minKernelSize.width / 2, maxKernelSize.width / 2 + maxKernelSize.width % 2) * 2 + 1; //generate only random odd numbers
}

void AverageBlur::image(InputArray src, OutputArray dst)
{
    cv::blur(src, dst, kernelSize);
}

}}
