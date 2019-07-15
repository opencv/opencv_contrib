// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv { namespace augment {

Resize::Resize(Size _size,const std::vector<int>& _interpolations)
{
    size = _size;
    if (_interpolations.size() > 0)
        interpolations = _interpolations;
    else
        interpolations = { INTER_NEAREST , INTER_LINEAR , INTER_AREA , INTER_CUBIC , INTER_LANCZOS4 };
}

Resize::Resize(Size _size, int _interpolation)
{
    size = _size;
    interpolations = { _interpolation };
}

void Resize::init(const Mat& srcImage)
{
    Transform::init(srcImage);
    int index = Transform::rng.uniform(0, int(interpolations.size()));
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
