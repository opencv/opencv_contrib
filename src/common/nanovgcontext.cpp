#include "nanovgcontext.hpp"

#include "viz2d.hpp"

namespace kb {

NanoVGContext::NanoVGContext(Viz2D &window, NVGcontext *context, CLGLContext &fbContext) :
        window_(window), context_(context), clglContext_(fbContext) {
    nvgCreateFont(context_, "libertine", "assets/LinLibertine_RB.ttf");

    //FIXME workaround for first frame color glitch
    cv::UMat tmp;
    clglContext_.acquireFromGL(tmp);
    clglContext_.releaseToGL(tmp);
}

void NanoVGContext::render(std::function<void(NVGcontext*, const cv::Size&)> fn) {
    CLExecScope_t scope(clglContext_.getCLExecContext());
    begin();
    fn(context_, clglContext_.getSize());
    end();
}

void NanoVGContext::begin() {
    clglContext_.begin();

    float w = window_.clva().getVideoFrameSize().width;
    float h = window_.clva().getVideoFrameSize().height;
    float r = window_.getXPixelRatio();

    nvgSave(context_);
    nvgBeginFrame(context_, w, h, 1);
    GL_CHECK(glViewport(0, 0,w,h));
}

void NanoVGContext::end() {
    nvgEndFrame(context_);
    nvgRestore(context_);
    clglContext_.end();
}
} /* namespace kb */
