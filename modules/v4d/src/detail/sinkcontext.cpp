// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "../../include/opencv2/v4d/detail/sinkcontext.hpp"
#include "../../include/opencv2/v4d/v4d.hpp"

#include <opencv2/imgproc.hpp>

namespace cv {
namespace v4d {
namespace detail {

SinkContext::SinkContext(cv::Ptr<FrameBufferContext> mainFbContext) : mainFbContext_(mainFbContext) {
}

void SinkContext::execute(std::function<void()> fn) {
    if (hasContext()) {
        CLExecScope_t scope(getCLExecContext());
        fn();
    } else {
    	fn();
    }
	auto v4d = mainFbContext_->getV4D();
	if(v4d->hasSink()) {
		v4d->getSink()->operator ()(v4d->sourceCtx()->sequenceNumber(), sinkBuffer());
	}
}

bool SinkContext::hasContext() {
    return !context_.empty();
}

void SinkContext::copyContext() {
    context_ = CLExecContext_t::getCurrent();
}

CLExecContext_t SinkContext::getCLExecContext() {
    return context_;
}

cv::UMat& SinkContext::sinkBuffer() {
	return sinkBuffer_;
}
}
}
}
