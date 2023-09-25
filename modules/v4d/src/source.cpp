// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/source.hpp"

namespace cv {
namespace v4d {

Source::Source(std::function<bool(cv::UMat&)> generator, float fps, bool async) :
        generator_(generator), fps_(fps), async_(async) {
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

bool Source::isAsync() {
    return async_;
}

bool Source::isThreadSafe() {
    return threadSafe_;
}

void Source::setThreadSafe(bool ts) {
    threadSafe_ = ts;
}

float Source::fps() {
    return fps_;
}

std::pair<uint64_t, cv::UMat&> Source::operator()() {
    if(threadSafe_) {
        std::unique_lock<std::mutex> lock(mtx_);
        open_ = generator_(frame_);
        return {count_++, frame_};
    } else {
        open_ = generator_(frame_);
        return {count_++, frame_};
    }
}
} /* namespace v4d */
} /* namespace kb */
