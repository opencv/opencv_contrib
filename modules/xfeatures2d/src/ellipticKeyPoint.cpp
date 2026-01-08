// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

namespace cv {
namespace xfeatures2d {

Elliptic_KeyPoint::Elliptic_KeyPoint(Point2f _pt, float _angle, Size _axes, float _size, float _si) :
    KeyPoint(_pt,_size,_angle), axes(_axes), si(_si) {
}

Elliptic_KeyPoint::Elliptic_KeyPoint(){

}

Elliptic_KeyPoint::~Elliptic_KeyPoint() {
}

}
}
