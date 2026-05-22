// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/source.hpp"

namespace cv {
namespace v4d {

Source::Source(std::function<bool(cv::UMat&)> generator, float fps) :
        generator_(generator), fps_(fps) {
}

Source::Source() :
        open_(false), fps_(0) {
}

Source::~Source() {
}

bool Source::isOpen() {
	std::lock_guard<std::mutex> guard(mtx_);
    return generator_ && open_;
}

float Source::fps() {
    return fps_;
}

std::pair<uint64_t, cv::UMat> Source::operator()() {
	std::lock_guard<std::mutex> guard(mtx_);
	static thread_local cv::UMat frame;
    if(threadSafe_) {
        static std::mutex mtx_;
        std::unique_lock<std::mutex> lock(mtx_);
        open_ = generator_(frame);
        return {count_++, frame};
    } else {
        open_ = generator_(frame);
        return {count_++, frame};
    }
}
} /* namespace v4d */
} /* namespace kb */
