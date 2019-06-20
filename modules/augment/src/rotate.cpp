// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv { namespace augment {

Rotate::Rotate(Vec2f angleRange)
{
    this->angleRange = angleRange;
}

Rotate::Rotate(float angle) { angleRange = Vec2f({ angle , angle }); }

void Rotate::init(Mat srcImage) 
{
    Transform::init(srcImage);
    float currentAngle = Transform::rng.uniform(angleRange[0], angleRange[1]);
    rotationMat = getRotationMatrix2D(Point2f(srcImageCols / 2, srcImageRows / 2), currentAngle, 1);
}

void Rotate::image(InputArray _src, OutputArray _dst)
{
    warpAffine(_src, _dst, rotationMat, Size(srcImageCols, srcImageRows));
}

Point2f Rotate::point(const Point2f& src)
{
    Mat srcM = (Mat_<double>(3, 1) << src.x, src.y, 1);
    Mat dstM = rotationMat*srcM;
    Point2f dst(dstM.at<double>(0,0), dstM.at<double>(1,0));
    return dst;
}

}}
