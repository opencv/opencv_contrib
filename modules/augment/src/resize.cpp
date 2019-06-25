// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv { namespace augment {

Resize::Resize(Size size, std::vector<int>& interpolations)
{
    this->size = size;
    this->interpolations = interpolations;
    useSize = true;
}

Resize::Resize(Size size, int interpolation)
{
    this->size = size;
    interpolations = std::vector<int>({ interpolation });
    useSize = true;
}

Resize::Resize(float fx, float fy, std::vector<int>& interpolations)
{
    this->fx = fx;
    this->fy = fy;
    this->interpolations = interpolations;
    useSize = false;
}

Resize::Resize(float fx, float fy, int interpolation)
{
    this->fx = fx;
    this->fy = fy;
    interpolations = std::vector<int>({ interpolation });
    useSize = false;
}


void Resize::init(const Mat& srcImage)
{
    Transform::init(srcImage);
    int index = Transform::rng.uniform(0, interpolations.size());
    interpolation = interpolations[index];
}

void Resize::image(InputArray src, OutputArray dst)
{
    if(useSize)
        resize(src, dst, size, 0, 0, interpolation);
    else
        resize(src, dst, size, fx, fy, interpolation);
}

Point2f Resize::point(const Point2f& src)
{
    float x, y;
    if (useSize)
    {
        x = (src.x / srcImageCols)*size.width;
        y = (src.y / srcImageRows)*size.height;
    }
    else 
    {
        x = src.x*fx;
        y = src.y*fy;
    }
    return Point2f(x, y);
}

}}
