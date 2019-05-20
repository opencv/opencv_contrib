// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv {
namespace augment {


    FlipHorizontal::FlipHorizontal() {}

    void FlipHorizontal::image(InputArray _src, OutputArray _dst)
    {
        cv::flip(_src, _dst, 1);
    }

    Point2f FlipHorizontal::point(InputArray image,const Point2f& src)
    {
        Mat imageM = image.getMat();
        return Point2f(imageM.cols - 1 - src.x, src.y);
    }


    FlipVertical::FlipVertical() {}

    void FlipVertical::image(InputArray _src, OutputArray _dst)
    {
        cv::flip(_src, _dst, 0);
    }

    Point2f FlipVertical::point(InputArray image,const Point2f& src)
    {
        Mat imageM = image.getMat();
        return Point2f(src.x, imageM.rows - 1 - src.y);
    }
}
}
