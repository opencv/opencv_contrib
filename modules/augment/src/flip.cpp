// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv {
namespace augment {


    FlipHorizontal::FlipHorizontal() {}


    void FlipHorizontal::image(InputArray _src, OutputArray _dst)
    {
        Mat src = _src.getMat();
        _dst.create(src.size(), src.type());
        Mat dst = _dst.getMat();
        cv::flip(src, dst, 1);
    }

    Point2d FlipHorizontal::point(InputArray image, Point2d& src)
    {
        Mat imageM = image.getMat();
        return Point2d(imageM.size().width - 1 - src.x, src.y);
    }





    FlipVertical::FlipVertical() {}


    void FlipVertical::image(InputArray _src, OutputArray _dst)
    {
        Mat src = _src.getMat();
        _dst.create(src.size(), src.type());
        Mat dst = _dst.getMat();
        cv::flip(src, dst, 0);
    }

    Point2d FlipVertical::point(InputArray image, Point2d& src)
    {
        Mat imageM = image.getMat();
        return Point2d(src.x, imageM.size().height - 1 - src.y);
    }
}
}
