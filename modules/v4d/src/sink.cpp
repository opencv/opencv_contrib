// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/sink.hpp"

namespace cv {
namespace v4d {

Sink::Sink(std::function<bool(const uint64_t&, const cv::UMat&)> consumer) :
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

bool Sink::isThreadSafe() {
    return threadSafe_;
}

void Sink::setThreadSafe(bool ts) {
    threadSafe_ = ts;
}

void Sink::operator()(const uint64_t& seq, const cv::UMat& frame) {
	if(isThreadSafe()) {
		std::unique_lock<std::mutex> lock(mtx_);
		open_ = consumer_(seq, frame);
	} else {
		open_ = consumer_(seq, frame);
	}
}
} /* namespace v4d */
} /* namespace kb */
