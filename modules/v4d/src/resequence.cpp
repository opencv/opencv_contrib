#include "../include/opencv2/v4d/detail/resequence.hpp"
#include <opencv2/core/utils/logger.hpp>

namespace cv {
namespace v4d {
	void Resequence::finish() {
		std::unique_lock<std::mutex> lock(putMtx_);
		finish_ = true;
		notify();
	}

	void Resequence::notify() {
		cv_.notify_all();
	}

	void Resequence::waitFor(const uint64_t& seq) {
		while(true) {
			{
				std::unique_lock<std::mutex> lock(putMtx_);
				if(finish_)
					break;

				if(seq == nextSeq_) {
					++nextSeq_;
					break;
				}
			}
			std::unique_lock<std::mutex> lock(waitMtx_);
			cv_.wait(lock);
		}
    }
} /* namespace v4d */
} /* namespace cv */
