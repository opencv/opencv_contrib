// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv { namespace augment {

GaussianBlur::GaussianBlur(int minKernelSize, int maxKernelSize, float minSigma, float maxSigma)
{
    this->minKernelSize = minKernelSize;
    this->maxKernelSize = maxKernelSize;
    this->minSigma = minSigma;
    this->maxSigma = maxSigma;
}

GaussianBlur::GaussianBlur(int minKernelSize, int maxKernelSize)
{
    this->minKernelSize = minKernelSize;
    this->maxKernelSize = maxKernelSize;
    minSigma = maxSigma = 0; //the sigma will be decided depending on the kernel size
}

void GaussianBlur::init(const Mat& image)
{
    currentKernelSize = Transform::rng.uniform(minKernelSize, maxKernelSize+1); //+1 because the RNG.uniform excludes the max number
    if (currentKernelSize % 2 == 0)
    {
        if (currentKernelSize > minKernelSize) currentKernelSize--;
        else if (currentKernelSize < maxKernelSize) currentKernelSize++;
    }
    currentSigma = Transform::rng.uniform(minSigma, maxSigma);
}

void GaussianBlur::image(InputArray src, OutputArray dst)
{
    cv::GaussianBlur(src, dst, Size(currentKernelSize, currentKernelSize), currentSigma);
}

}}
