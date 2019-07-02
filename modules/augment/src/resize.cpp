// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv { namespace augment {

Resize::Resize(Size size,const std::vector<int>& interpolations)
{
    this->size = size;
    if (interpolations.size() > 0)
        this->interpolations = interpolations;
    else
        this->interpolations = { INTER_NEAREST , INTER_LINEAR , INTER_AREA , INTER_CUBIC , INTER_LANCZOS4 };
}

Resize::Resize(Size size, int interpolation)
{
    this->size = size;
    interpolations = { interpolation };
}

void Resize::init(const Mat& srcImage)
{
    Transform::init(srcImage);
    int index = Transform::rng.uniform(0, interpolations.size());
    interpolation = interpolations[index];
}

void Resize::image(InputArray src, OutputArray dst)
{
    resize(src, dst, size, 0, 0, interpolation);
}

void Resize::mask(InputArray src, OutputArray dst)
{
    resize(src, dst, size, 0, 0, INTER_NEAREST);
}

Point2f Resize::point(const Point2f& src)
{
    float x, y;
    x = (src.x / (srcImageCols - 1))*(size.width - 1);
    y = (src.y / (srcImageRows - 1))*(size.height - 1);
    return Point2f(x, y);
}

}}
