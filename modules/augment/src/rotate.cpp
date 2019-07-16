// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv { namespace augment {

Rotate::Rotate(float _minAngle, float _maxAngle)
{
    minAngle = _minAngle;
    maxAngle = _maxAngle;
}

Rotate::Rotate(float angle)
{
    minAngle = angle;
    maxAngle = angle;
}

void Rotate::init(const Mat& srcImage)
{
    Transform::init(srcImage);
    float currentAngle = Transform::rng.uniform(minAngle, maxAngle);
    rotationMat = getRotationMatrix2D(Point2f(float(srcImageCols) / 2, float(srcImageRows) / 2), currentAngle, 1);
}

void Rotate::image(InputArray src, OutputArray dst)
{
    warpAffine(src, dst, rotationMat, src.size());
}

Point2f Rotate::point(const Point2f& src)
{
    Mat srcM = (Mat_<double>(3, 1) << src.x, src.y, 1);
    Mat dstM = rotationMat*srcM;
    Point2f dst(float(dstM.at<double>(0,0)), float(dstM.at<double>(1,0)));
    return dst;
}

}}
