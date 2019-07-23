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
    scale = 1.f;
}

Crop::Crop(Size _minSize, Size _maxSize)
{
    minSize = _minSize;
    maxSize = _maxSize;
    origin = Point(0, 0);
    randomXY = true;
    randomSize = true;
    scale = 1.f;
}

Crop::Crop(Size _size, Point _origin)
{
    size = _size;
    origin = _origin;
    randomXY = false;
    randomSize = false;
    scale = 1.f;
}

Crop::Crop(Size _minSize, Size _maxSize, Point _origin)
{
    minSize = _minSize;
    maxSize = _maxSize;
    origin = _origin;
    randomXY = false;
    randomSize = true;
    scale = 1.f;
}

void Crop::init(const Mat& srcImage)
{
    Transform::init(srcImage);
    
    if (randomSize)
    {
        size.width = Transform::rng.uniform(minSize.width, maxSize.width);
        size.height = Transform::rng.uniform(minSize.height, maxSize.height);
    }

    if (randomXY)
    {
        int differenceX = srcImageCols - size.width; //the amount of pixels available to shift the center of the new image in X directon
        int differenceY = srcImageRows - size.height; //the amount of pixels available to shift the center of the new image in Y directon
        origin.x = Transform::rng.uniform(0, differenceX);
        origin.y = Transform::rng.uniform(0, differenceY);
    }

    if (size.width > srcImageCols || size.height > srcImageRows)
        calculateScale();

}

void Crop::image(InputArray src, OutputArray dst)
{
    if (size.width > srcImageCols || size.height > srcImageRows)
    {
        boxImage(src, dst);
        return;
    }

    Rect rect(origin.x, origin.y, size.width, size.height);
    Mat dstMat;
    Mat srcMat = src.getMat();
    dstMat = srcMat(rect);
    dstMat.copyTo(dst);
}

void Crop::boxImage(InputArray src, OutputArray dst)
{
    Mat srcMat = src.getMat();
    Mat resized;
    resize(src, resized, Size(), scale, scale, INTER_NEAREST);
    dst.create(size, srcMat.type());
    Mat dstMat = dst.getMat();
    Rect rect(int((dstMat.cols - resized.cols) / 2), int((dstMat.rows - resized.rows) / 2), resized.cols, resized.rows);
    resized.copyTo(dstMat(rect));
}

Point2f Crop::point(const Point2f& src)
{
    if (size.width > srcImageCols || size.height > srcImageRows)
    {
        int resizedCols = int(scale*srcImageCols);
        int resizedRows = int(scale*srcImageRows);
        return Point2f(src.x / (srcImageCols - 1)*(resizedCols - 1) + int((size.width - resizedCols) / 2),
                     src.y / (srcImageRows - 1)*(resizedRows - 1) + int((size.height - resizedRows) / 2));
    }

    return Point2f(src.x - origin.x, src.y - origin.y);
}


Rect2f Crop::rectangle(const Rect2f& src)
{
    if (size.width > srcImageCols || size.height > srcImageRows)
    {
        Point2f tl = point(Point2f(src.x, src.y));
        return Rect2f(tl.x , tl.y, src.width*scale, src.height*scale);
    }

    return Rect2f(src.x - origin.x, src.y - origin.y, src.width, src.height);
}

void Crop::calculateScale()
{
    float scaleW, scaleH;
    scaleW = float(size.width) / srcImageCols;
    scaleH = float(size.height) / srcImageRows;
    scale = std::min(scaleW, scaleH);
}

}}
