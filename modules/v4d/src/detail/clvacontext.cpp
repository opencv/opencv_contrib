// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "clvacontext.hpp"
#include "opencv2/v4d/v4d.hpp"

namespace cv {
namespace v4d {
namespace detail {

CLVAContext::CLVAContext(V4D& v4d, FrameBufferContext& mainFbContext) :
        mainFbContext_(mainFbContext), clvaFbContext_(v4d, "CLVA", mainFbContext) {
}

cv::Size CLVAContext::getVideoFrameSize() {
    assert(inputVideoFrameSize_ == cv::Size(0, 0) || "Video frame size not initialized");
    return inputVideoFrameSize_;
}

cv::UMat CLVAContext::capture(std::function<void(cv::UMat&)> fn) {
    cv::Size fbSize = fbCtx().getSize();
    if (!context_.empty()) {
        {
#ifndef __EMSCRIPTEN__
            CLExecScope_t scope(context_);
#endif
            fn(readFrame_);
        }
        if (readFrame_.empty())
            return {};
        inputVideoFrameSize_ = readFrame_.size();

        fbCtx().execute([this](cv::UMat& frameBuffer) {
            resizePreserveAspectRatio(readFrame_, readRGBBuffer_, frameBuffer.size());
            cv::cvtColor(readRGBBuffer_, frameBuffer, cv::COLOR_RGB2BGRA);
        });
    } else {
        fn(readFrame_);
        if (readFrame_.empty())
            return {};
        inputVideoFrameSize_ = readFrame_.size();
        fbCtx().execute([this](cv::UMat& frameBuffer) {
            resizePreserveAspectRatio(readFrame_, readRGBBuffer_, frameBuffer.size());
            cv::cvtColor(readRGBBuffer_, frameBuffer, cv::COLOR_RGB2BGRA);
        });
    }

    return readRGBBuffer_;
}

void CLVAContext::write(std::function<void(const cv::UMat&)> fn) {
        fbCtx().execute([=,this](cv::UMat& frameBuffer) {
            frameBuffer.copyTo(writeFrame_);
        });
#ifndef __EMSCRIPTEN__
        CLExecScope_t scope(context_);
#endif
        cv::cvtColor(writeFrame_, writeRGBBuffer_, cv::COLOR_BGRA2RGB);
        fn(writeRGBBuffer_);
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

FrameBufferContext& CLVAContext::fbCtx() {
    return clvaFbContext_;
}
}
}
}
