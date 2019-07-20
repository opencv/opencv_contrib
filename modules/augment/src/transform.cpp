// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#include "precomp.hpp"
namespace cv { namespace augment {

Transform::Transform() {}

Transform::~Transform() {}

void Transform::image(InputArray _src, OutputArray _dst)
{
    _src.copyTo(_dst);
}

Point2f Transform::point(const Point2f& src)
{
    return src;
}

Rect2f Transform::rectangle(const Rect2f& src)
{
    float x1 = src.x,
        y1 = src.y,
        x2 = src.x + src.width,
        y2 = src.y + src.height;

    Point2f tl(x1, y1);
    Point2f bl(x1, y2);
    Point2f tr(x2, y1);
    Point2f br(x2, y2);

    Point2f tl_transformed = this->point(tl);
    Point2f bl_transformed = this->point(bl);
    Point2f tr_transformed = this->point(tr);
    Point2f br_transformed = this->point(br);

    float x1_transformed = std::min({ tl_transformed.x, bl_transformed.x, tr_transformed.x, br_transformed.x });
    float y1_transformed = std::min({ tl_transformed.y, bl_transformed.y, tr_transformed.y, br_transformed.y });
    float x2_transformed = std::max({ tl_transformed.x, bl_transformed.x, tr_transformed.x, br_transformed.x });
    float y2_transformed = std::max({ tl_transformed.y, bl_transformed.y, tr_transformed.y, br_transformed.y });

    Rect2f output_box(Point2f(x1_transformed ,y1_transformed),  Point2f(x2_transformed ,y2_transformed));
    return output_box;
}

Rect2f Transform::basicRectangle(const Rect2f& src)
{
    float x1 = src.x,
        y1 = src.y,
        x2 = src.x + src.width,
        y2 = src.y + src.height;

    Point2f tl(x1, y1);
    Point2f br(x2, y2);

    Point2f tl_transformed = this->point(tl);
    Point2f br_transformed = this->point(br);

    Rect2f output_box(tl_transformed, br_transformed);
    return output_box;
}

void Transform::points(InputArray _src, OutputArray _dst)
{
    Mat src = _src.getMat();
    CV_Assert(src.cols == 2);

    //making sure input matrix is float
    int type = src.type();
    uchar depth = type & CV_MAT_DEPTH_MASK;
    if (depth != CV_32F)
    {
        src.convertTo(src, CV_32F);
    }

    _dst.create(src.size(), CV_32F);
    Mat dst = _dst.getMat();

    for (int i = 0; i < src.rows; i++)
    {
        Mat src_row = src.row(i);
        Point2f src_point = Point2f(src_row.at<float>(0), src_row.at<float>(1));
        Point2f dst_point = this->point(src_point);
        dst.at<float>(i,0) = dst_point.x;
        dst.at<float>(i,1) = dst_point.y;
    }

}

void Transform::rectangles(InputArray _src, OutputArray _dst)
{
    Mat src = _src.getMat();
    CV_Assert(src.cols == 4);

    //making sure input matrix is float
    int type = src.type();
    uchar depth = type & CV_MAT_DEPTH_MASK;
    if (depth != CV_32F)
    {
        src.convertTo(src, CV_32F);
    }

    _dst.create(src.size(), CV_32F);
    Mat dst = _dst.getMat();

    for (int i = 0; i < src.rows; i++)
    {
        Mat src_row = src.row(i);
        Rect2f src_rect = Rect2f(src_row.at<float>(0), src_row.at<float>(1), src_row.at<float>(2), src_row.at<float>(3));
        Rect2f dst_rect = this->rectangle(src_rect);
        dst.at<float>(i, 0) = dst_rect.x;
        dst.at<float>(i, 1) = dst_rect.y;
        dst.at<float>(i, 2) = dst_rect.width;
        dst.at<float>(i, 3) = dst_rect.height;
    }

}

void Transform::polygons(std::vector<Mat> src, OutputArrayOfArrays dst)
{
    dst.create(int(src.size()), 1, 0, -1, true);

    for (unsigned int i = 0; i < src.size(); i++)
    {
        Mat src_row = src[i];
        dst.create(src_row.size(), CV_32F, i, true);
        Mat dst_row = dst.getMat(i);
        this->points(src_row, dst_row);
    }
}

void Transform::mask(InputArray src, OutputArray dst)
{
    image(src, dst);
}


void Transform::init(const Mat& srcImage)
{
    srcImageRows = srcImage.rows;
    srcImageCols = srcImage.cols;
}

RNG Transform::rng;

}}