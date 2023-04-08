// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/source.hpp"

namespace cv {
namespace viz {

Source::Source(std::function<bool(cv::UMat&)> generator, float fps) :
        generator_(generator), fps_(fps) {
}

Source::Source() :
        open_(false), fps_(0) {
}

Source::~Source() {
}

bool Source::isReady() {
    if (generator_)
        return true;
    else
        return false;
}

bool Source::isOpen() {
    return open_;
}

float Source::fps() {
    return fps_;
}

std::pair<uint64_t, cv::UMat&> Source::operator()() {
    open_ = generator_(frame_);
    return {count_++, frame_};
}
} /* namespace viz2d */
} /* namespace kb */
