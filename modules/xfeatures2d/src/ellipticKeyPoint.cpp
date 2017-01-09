// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

namespace cv {
namespace xfeatures2d {

Elliptic_KeyPoint::Elliptic_KeyPoint(Point _centre, double _phi, Size _axes, float _size, float _si) :
    KeyPoint(_centre,_size), centre(_centre), axes(_axes), phi(_phi), size(_size), si(_si) {
}

Elliptic_KeyPoint::Elliptic_KeyPoint(){

}

Elliptic_KeyPoint::~Elliptic_KeyPoint() {
}

}
}
