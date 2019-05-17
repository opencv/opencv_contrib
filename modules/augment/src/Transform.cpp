// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#include "precomp.hpp"
#include <iostream>
namespace cv {
namespace augment {

    Transform::Transform(const Scalar& _proability) : probability(_proability) {}

    Transform::~Transform() {}

    Scalar Transform::getProbability() { return this->probability; }
        
    void Transform::setProbability(Scalar& _probability) { this->probability = _probability; }
        
    void Transform::image(InputArray _src, OutputArray _dst)
    {
        Mat src = _src.getMat();
        _dst.create(src.size(), src.type());
        Mat dst = _dst.getMat();
        src.copyTo(dst);
    }

    Point2d Transform::point(InputArray, Point2d& src)
    {
        return src;
    }

    Scalar Transform::rectangle(InputArray image,const Scalar& src)
    {
        double x1 = src[0],
            y1 = src[1],
            x2 = src[2],
            y2 = src[3];

        Point2d tl(x1, y1);
        Point2d bl(x1, y2);
        Point2d tr(x2, y1);
        Point2d br(x2, y2);

        Point2d tl_transformed = this->point(image, tl);
        Point2d bl_transformed = this->point(image, bl);
        Point2d tr_transformed = this->point(image, tr);
        Point2d br_transformed = this->point(image, br);

        double x1_transformed = std::min({ tl_transformed.x, bl_transformed.x, tr_transformed.x, br_transformed.x });
        double y1_transformed = std::min({ tl_transformed.y, bl_transformed.y, tr_transformed.y, br_transformed.y });
        double x2_transformed = std::max({ tl_transformed.x, bl_transformed.x, tr_transformed.x, br_transformed.x });
        double y2_transformed = std::max({ tl_transformed.y, bl_transformed.y, tr_transformed.y, br_transformed.y });

        Scalar output_box({ x1_transformed ,y1_transformed ,x2_transformed ,y2_transformed });
        return output_box;
    }


    void Transform::polygon(InputArray image, InputArray _src, OutputArray _dst)
    {
        Mat src = _src.getMat();
        CV_Assert(src.cols == 2);
        
        //making sure input matrix is double
        int type = src.type();
        uchar depth = type & CV_MAT_DEPTH_MASK;
        if (depth != CV_64F)
        {
            src.convertTo(src, CV_64F);
        }

        _dst.create(src.size(), CV_64F);
        Mat dst = _dst.getMat();

        for (size_t i = 0; i < src.rows; i++)
        {
            Mat src_row = src.row(i);
            Point2d src_point = Point2d(src_row.at<double>(0), src_row.at<double>(1));
            Point2d dst_point = this->point(image, src_point);
            dst.at<double>(i,0) = dst_point.x;
            dst.at<double>(i,1) = dst_point.y;
        }

    }


        

}
}
