/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/


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
