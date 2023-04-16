// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/sink.hpp"

namespace cv {
namespace viz {

Sink::Sink(std::function<bool(const cv::UMat&)> consumer) :
        consumer_(consumer) {
}

Sink::Sink() {

}
Sink::~Sink() {
}

bool Sink::isReady() {
    if (consumer_)
        return true;
    else
        return false;
}

bool Sink::isOpen() {
    return open_;
}

void Sink::operator()(const cv::UMat& frame) {
    open_ = consumer_(frame);
}
} /* namespace v4d */
} /* namespace kb */
