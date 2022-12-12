#include "clvacontext.hpp"

#include "glwindow.hpp"

namespace kb {

CLVAContext::CLVAContext(CLGLContext &fbContext) :
        fbContext_(fbContext) {
}

bool CLVAContext::capture(std::function<void(cv::UMat&)> fn) {
    {
        CLExecScope_t scope(context_);
        fn(videoFrame_);
    }
    {
        CLExecScope_t scope(fbContext_.getCLExecContext());
        fbContext_.acquireFromGL(frameBuffer_);
        if (videoFrame_.empty())
            return false;

        cv::cvtColor(videoFrame_, frameBuffer_, cv::COLOR_RGB2BGRA);
        cv::Size fbSize = fbContext_.getSize();
        cv::resize(frameBuffer_, frameBuffer_, fbSize);
        fbContext_.releaseToGL(frameBuffer_);
        assert(frameBuffer_.size() == fbSize);
    }
    return true;
}

void CLVAContext::write(std::function<void(const cv::UMat&)> fn) {
    cv::Size fbSize = fbContext_.getSize();
    {
        CLExecScope_t scope(fbContext_.getCLExecContext());
        fbContext_.acquireFromGL(frameBuffer_);
        cv::resize(frameBuffer_, frameBuffer_, fbSize);
        cv::cvtColor(frameBuffer_, videoFrame_, cv::COLOR_BGRA2RGB);
        fbContext_.releaseToGL(frameBuffer_);
    }
    assert(videoFrame_.size() == fbSize);
    {
        CLExecScope_t scope(context_);
        fn(videoFrame_);
    }
}

bool CLVAContext::hasContext() {
    return !context_.empty();
}

void CLVAContext::copyContext() {
    context_ = CLExecContext_t::getCurrent();
}

CLExecContext_t CLVAContext::getCLExecContext() {
    return context_;
}
} /* namespace kb */
