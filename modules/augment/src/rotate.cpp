// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv { namespace augment {

Rotate::Rotate(float _minAngle, float _maxAngle, std::vector<int> _interpolations, std::vector<int> _borderTypes, const Scalar&  _borderValue)
{
    minAngle = _minAngle;
    maxAngle = _maxAngle;
    angle = minAngle;

    if (_borderTypes.size() > 0)
        borderTypes = _borderTypes;
    else
        borderTypes = { BORDER_CONSTANT , BORDER_REPLICATE , BORDER_WRAP , BORDER_REFLECT_101 };

    borderType = BORDER_CONSTANT;
    borderValue = _borderValue;

    if (_interpolations.size() > 0)
        interpolations = _interpolations;
    else interpolations = { INTER_NEAREST , INTER_LINEAR , INTER_CUBIC , INTER_AREA  };

    interpolation = INTER_LINEAR;

    random = true;
}

Rotate::Rotate(float _angle, int _interpolation, int _borderType, int _borderValue)
{
    angle = _angle;
    interpolation = _interpolation;
    borderType = _borderType;
    borderValue = _borderValue;
    random = false;

}

void Rotate::init(const Mat& srcImage)
{
    Transform::init(srcImage);
    if (random)
    {
        angle = Transform::rng.uniform(minAngle, maxAngle);
        int indexInterpolation = Transform::rng.uniform(0, int(interpolations.size()));
        interpolation = interpolations[indexInterpolation];
        int indexBorderType = Transform::rng.uniform(0, int(borderTypes.size()));
        borderType = borderTypes[indexBorderType];
    }
    rotationMat = getRotationMatrix2D(Point2f(float(srcImageCols) / 2, float(srcImageRows) / 2), angle, 1);
}

void Rotate::image(InputArray src, OutputArray dst)
{
    warpAffine(src, dst, rotationMat, src.size(), interpolation, borderType, borderValue);
}

Point2f Rotate::point(const Point2f& src)
{
    Mat srcM = (Mat_<double>(3, 1) << src.x, src.y, 1);
    Mat dstM = rotationMat*srcM;
    Point2f dst(float(dstM.at<double>(0,0)), float(dstM.at<double>(1,0)));
    return dst;
}

}}
