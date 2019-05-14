// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#include "precomp.hpp"

namespace cv {
namespace augment {

    Transform::Transform(const Scalar& _proability) : probability(_proability) {}

    Transform::~Transform() {}

    Scalar Transform::getProbability() { return this->probability; }
        
    void Transform::setProbability(Scalar& _probability) { this->probability = probability; }
        
    void Transform::image(InputArray _src, OutputArray _dst)
    {
        Mat src = _src.getMat();
        _dst.create(src.size(), src.type());
        Mat dst = _dst.getMat();
        src.copyTo(dst);
    }

    Point2d Transform::point(InputArray image, Point2d& src)
    {
        return src;
    }

    Scalar Transform::rect(InputArray image, Scalar box)
    {
        double x1 = box[0],
            y1 = box[1],
            x2 = box[2],
            y2 = box[3];

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


        return Scalar(x1_transformed, y1_transformed, x2_transformed, y2_transformed);
    }
        

}
}
