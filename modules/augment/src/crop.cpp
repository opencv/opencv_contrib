// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv { namespace augment {



Crop::Crop(Size _size)
{
    size = _size;
    origin = Point(0, 0);
    randomXY = true;
    randomSize = false;
}

Crop::Crop(Size _minSize, Size _maxSize)
{
    minSize = _minSize;
    maxSize = _maxSize;
    origin = Point(0, 0);
    randomXY = true;
    randomSize = true;
}

Crop::Crop(Size _size, Point _origin)
{
    size = _size;
    origin = _origin;
    randomXY = false;
    randomSize = false;
}

Crop::Crop(Size _minSize, Size _maxSize, Point _origin)
{
    minSize = _minSize;
    maxSize = _maxSize;
    origin = _origin;
    randomXY = false;
    randomSize = true;
}

void Crop::init(const Mat& srcImage)
{
    Transform::init(srcImage);

    if (randomSize)
    {
        size.width = Transform::rng.uniform(minSize.width, maxSize.width);
        size.height = Transform::rng.uniform(minSize.height, maxSize.height);
    }

    else
        CV_Assert(size.width <= srcImageCols && size.height <= srcImageRows);

    if (randomXY)
    {
        int differenceX = srcImageCols - size.width; //the amount of pixels available to shift the center of the new image in X directon
        int differenceY = srcImageRows - size.height; //the amount of pixels available to shift the center of the new image in Y directon
        origin.x = Transform::rng.uniform(0, differenceX);
        origin.y = Transform::rng.uniform(0, differenceY);
    }
}

void Crop::image(InputArray src, OutputArray dst)
{
    Rect rect(origin.x, origin.y, size.width, size.height);
    Mat dstMat;
    Mat srcMat = src.getMat();
    dstMat = srcMat(rect);
    dstMat.copyTo(dst);
}


Point2f Crop::point(const Point2f& src)
{
    return Point2f(src.x - origin.x, src.y - origin.y);
}


Rect2f Crop::rectangle(const Rect2f& src)
{
    return basicRectangle(src);
}

}}
