// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv { namespace augment {



CropToFixedSize::CropToFixedSize(int _width, int _height)
{
    width = _width;
    height = _height;
}

void CropToFixedSize::init(const Mat& srcImage)
{
    Transform::init(srcImage);
    CV_Assert(width <= srcImageCols && height <= srcImageRows);
    int differenceX = srcImageCols - width; //the amount of pixels available to shift the center of the new image in X directon
    int differenceY = srcImageRows - height; //the amount of pixels available to shift the center of the new image in Y directon
    originX = Transform::rng.uniform(0, differenceX);
    originY = Transform::rng.uniform(0, differenceY);
}

void CropToFixedSize::image(InputArray src, OutputArray dst)
{
    Rect rect(originX, originY, width, height);
    Mat dstMat;
    Mat srcMat = src.getMat();
    dstMat = srcMat(rect);
    dstMat.copyTo(dst);
}


Point2f CropToFixedSize::point(const Point2f& src)
{ 
    return Point2f(src.x - originX, src.y - originY);
}

}}
