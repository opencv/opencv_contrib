// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "../../include/opencv2/v4d/detail/sourcecontext.hpp"

#include "../../include/opencv2/v4d/v4d.hpp"

#include <opencv2/imgproc.hpp>

namespace cv {
namespace v4d {
namespace detail {

SourceContext::SourceContext(cv::Ptr<FrameBufferContext> mainFbContext) : mainFbContext_(mainFbContext) {
}

void SourceContext::execute(std::function<void()> fn) {
    run_sync_on_main<30>([this,fn](){
#ifndef __EMSCRIPTEN__
    if (hasContext()) {
        CLExecScope_t scope(getCLExecContext());
#endif
        if (mainFbContext_->getV4D()->hasSource()) {
        	auto src = mainFbContext_->getV4D()->getSource();

        	if(src->isOpen()) {
				auto p = src->operator ()();
				currentSeqNr_ = p.first;

				if(p.second.empty())
					p.second.create(mainFbContext_->size(), CV_8UC3);

				resizePreserveAspectRatio(p.second, captureBufferRGB_, mainFbContext_->size());
				cv::cvtColor(captureBufferRGB_, sourceBuffer(), cv::COLOR_RGB2BGRA);
        	}
        }
        fn();

#ifndef __EMSCRIPTEN__
    } else {
        if (mainFbContext_->getV4D()->hasSource()) {
        	auto src = mainFbContext_->getV4D()->getSource();

        	if(src->isOpen()) {
				auto p = src->operator ()();
				currentSeqNr_ = p.first;

				if(p.second.empty())
					p.second.create(mainFbContext_->size(), CV_8UC3);

				resizePreserveAspectRatio(p.second, captureBufferRGB_, mainFbContext_->size());
				cv::cvtColor(captureBufferRGB_, sourceBuffer(), cv::COLOR_RGB2BGRA);
        	}
        }
        fn();
    }
#endif
    });
}

uint64_t SourceContext::sequenceNumber() {
	return currentSeqNr_;
}

bool SourceContext::hasContext() {
    return !context_.empty();
}

void SourceContext::copyContext() {
#ifndef __EMSCRIPTEN__
    context_ = CLExecContext_t::getCurrent();
#endif
}

CLExecContext_t SourceContext::getCLExecContext() {
    return context_;
}

cv::UMat& SourceContext::sourceBuffer() {
	return captureBuffer_;
}
}
}
}
