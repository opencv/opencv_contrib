// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv { namespace augment {

AverageBlur::AverageBlur(Size _minKernelSize, Size _maxKernelSize, const std::vector<int>& _borderTypes)
{
    if (_borderTypes.size() > 0) 
        borderTypes = _borderTypes;

    else
        borderTypes = { BORDER_CONSTANT , BORDER_REPLICATE , BORDER_REFLECT, BORDER_REFLECT_101};

    minKernelSize = _minKernelSize;
    maxKernelSize = _maxKernelSize;
    kernelSize = minKernelSize;
    sameXY = false;
}

AverageBlur::AverageBlur(Size _kernelSize, int _borderType)
{
    borderTypes = { _borderType };
    minKernelSize = maxKernelSize = _kernelSize;
    kernelSize = minKernelSize;
    sameXY = false;
}

AverageBlur::AverageBlur(int _minKernelSize, int _maxKernelSize, int _borderType)
{
    borderTypes = { _borderType };
    minKernelSize = Size(_minKernelSize, _minKernelSize);
    maxKernelSize = Size(_maxKernelSize, _maxKernelSize);
    kernelSize = minKernelSize;
    sameXY = true;
}

AverageBlur::AverageBlur(int _kernelSize)
{
    borderTypes = { BORDER_DEFAULT };
    minKernelSize = maxKernelSize = Size(_kernelSize, _kernelSize);
    kernelSize = minKernelSize;
    sameXY = true;
}

void AverageBlur::init(const Mat&)
{
    if (minKernelSize != maxKernelSize)
    {
        kernelSize.height = Transform::rng.uniform(minKernelSize.height / 2, maxKernelSize.height / 2 + maxKernelSize.height % 2) * 2 + 1; //generate only random odd numbers
        kernelSize.width = sameXY ? kernelSize.height : Transform::rng.uniform(minKernelSize.width / 2, maxKernelSize.width / 2 + maxKernelSize.width % 2) * 2 + 1; //generate only random odd numbers
    }
    int index = Transform::rng.uniform(0, int(borderTypes.size()));
    borderType = borderTypes[index];
}

void AverageBlur::image(InputArray src, OutputArray dst)
{
    cv::blur(src, dst, kernelSize, Point(-1, -1), borderType);
}

Rect2f AverageBlur::rectangle(const Rect2f& src)
{
    return src;
}

}}
