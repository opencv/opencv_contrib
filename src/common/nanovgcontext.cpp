#include "nanovgcontext.hpp"

#include "viz2d.hpp"

namespace kb {

NanoVGContext::NanoVGContext(Viz2D &window, NVGcontext *context, CLGLContext &fbContext) :
        window_(window), context_(context), fbContext_(fbContext) {
    nvgCreateFont(context_, "libertine", "assets/LinLibertine_RB.ttf");

    //FIXME workaround for first frame color glitch
    cv::UMat tmp;
    fbContext_.acquireFromGL(tmp);
    fbContext_.releaseToGL(tmp);
}

void NanoVGContext::render(std::function<void(NVGcontext*, const cv::Size&)> fn) {
    CLExecScope_t scope(fbContext_.getCLExecContext());
    begin();
    fn(context_, fbContext_.getSize());
    end();
}

void NanoVGContext::begin() {
    fbContext_.begin();

    float w = window_.getNativeFrameBufferSize().width;
    float h = window_.getNativeFrameBufferSize().height;
    float r = window_.getXPixelRatio();
    nvgSave(context_);
    nvgBeginFrame(context_, w, h, r);
}

void NanoVGContext::end() {
    nvgEndFrame(context_);
    nvgRestore(context_);
    fbContext_.end();
}
} /* namespace kb */
