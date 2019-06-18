// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv {
namespace augment {

    Rotate::Rotate(Range _angleRange)
    {
        this->angleRange = _angleRange;
        this->resetRandom();
    }

    Rotate::Rotate() : Rotate::Rotate(Range(0, 180)) {}

    void Rotate::resetRandom()
    {
        this->currentAngle = rng.uniform(this->angleRange.start, this->angleRange.end + 1);
    }

    void Rotate::image(InputArray _src, OutputArray _dst)
    {
        Mat src = _src.getMat();
        Mat rotationMat = getRotationMatrix2D(Point2f(src.cols / 2, src.rows / 2), this->currentAngle, 1);
        warpAffine(src, _dst, rotationMat, Size(src.cols, src.rows));
    }

    Point2f Rotate::point(InputArray image, const Point2f& src)
    {
        Mat srcM = (Mat_<double>(3, 1) << src.x, src.y, 1);
        Mat imageMat = image.getMat();
        Mat rotationMat = getRotationMatrix2D(Point2f(imageMat.cols / 2, imageMat.rows / 2), this->currentAngle, 1);
        Mat dstM = rotationMat*srcM;
        Point2f dst(dstM.at<double>(0,0), dstM.at<double>(1,0));
        return dst;
    }

}
}
