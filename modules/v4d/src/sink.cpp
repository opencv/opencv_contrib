// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/sink.hpp"
#include <opencv2/core/utils/logger.hpp>

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
	std::lock_guard<std::mutex> lock(mtx_);
    if (consumer_)
        return true;
    else
        return false;
}

bool Sink::isOpen() {
	std::lock_guard<std::mutex> lock(mtx_);
    return open_;
}

void Sink::operator()(const uint64_t& seq, const cv::UMat& frame) {
	std::lock_guard<std::mutex> lock(mtx_);
	if(seq == nextSeq_) {
		uint64_t currentSeq = seq;
		cv::UMat currentFrame = frame;
		buffer_[seq] = frame;
		do {
			open_ = consumer_(currentSeq, currentFrame);
			++nextSeq_;
			buffer_.erase(buffer_.begin());
			if(buffer_.empty())
				break;
			auto pair = (*buffer_.begin());
			currentSeq = pair.first;
			currentFrame = pair.second;
		} while(currentSeq == nextSeq_);
	} else {
		buffer_[seq] = frame;
	}
	if(buffer_.size() > 240) {
		CV_LOG_WARNING(nullptr, "Buffer overrun in sink.");
		buffer_.clear();
	}
}
} /* namespace v4d */
} /* namespace kb */
