// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "clvacontext.hpp"

#include "opencv2/v4d/v4d.hpp"

namespace cv {
namespace v4d {
namespace detail {

CLVAContext::CLVAContext(FrameBufferContext& clglContext) :
        mainFbContext_(clglContext) {
}

cv::Size CLVAContext::getVideoFrameSize() {
    assert(videoFrameSize_ == cv::Size(0, 0) || "Video frame size not initialized");
    return videoFrameSize_;
}

bool CLVAContext::capture(std::function<void(cv::UMat&)> fn, cv::UMat& frameBuffer) {
    {
        if (!context_.empty()) {
#ifndef __EMSCRIPTEN__
            CLExecScope_t scope(context_);
#endif
            fn(videoFrame_);
            videoFrameSize_ = videoFrame_.size();
        } else {
            fn(videoFrame_);
            videoFrameSize_ = videoFrame_.size();
        }
    }
    {
#ifndef __EMSCRIPTEN__
        CLExecScope_t scope(mainFbContext_.getCLExecContext());
#endif
        if (videoFrame_.empty())
            return false;

        cv::Size fbSize = mainFbContext_.getSize();
        resizeKeepAspectRatio(videoFrame_, rgbBuffer_, fbSize);
        cv::cvtColor(rgbBuffer_, frameBuffer, cv::COLOR_RGB2BGRA);

        assert(frameBuffer.size() == fbSize);
    }
    return true;
}

void CLVAContext::write(std::function<void(const cv::UMat&)> fn, const cv::UMat& frameBuffer) {
    {
#ifndef __EMSCRIPTEN__
        CLExecScope_t scope(mainFbContext_.getCLExecContext());
#endif
        cv::cvtColor(frameBuffer, rgbBuffer_, cv::COLOR_BGRA2RGB);
        if(videoFrameSize_ == cv::Size(0,0))
            videoFrameSize_ = rgbBuffer_.size();
        cv::resize(rgbBuffer_, videoFrame_, videoFrameSize_);
    }
    assert(videoFrame_.size() == videoFrameSize_);
    {
#ifndef __EMSCRIPTEN__
        CLExecScope_t scope(context_);
#endif
        fn(videoFrame_.clone());
    }
}

bool CLVAContext::hasContext() {
    return !context_.empty();
}

void CLVAContext::copyContext() {
#ifndef __EMSCRIPTEN__
    context_ = CLExecContext_t::getCurrent();
#endif
}

CLExecContext_t CLVAContext::getCLExecContext() {
    return context_;
}
}
}
}
