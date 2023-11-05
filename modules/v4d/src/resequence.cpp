#include "../include/opencv2/v4d/detail/resequence.hpp"
#include <opencv2/core/utils/logger.hpp>

namespace cv {
namespace v4d {
	void Resequence::finish() {
		finish_ = true;
		notify();
	}

	void Resequence::notify() {
		cv_.notify_all();
	}

	void Resequence::waitFor(const uint64_t& seq) {
		while(!finish_) {
			if(seq == nextSeq_) {
				std::unique_lock<std::mutex> lock(putMtx_);
				++nextSeq_;
				break;
			} else {
				std::unique_lock<std::mutex> lock(waitMtx_);
				cv_.wait(lock, [this, seq](){return seq == nextSeq_;});
			}
		}
    }
} /* namespace v4d */
} /* namespace cv */
