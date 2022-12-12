#include "nanovgcontext.hpp"

#include "glwindow.hpp"

namespace kb {

NanoVGContext::NanoVGContext(GLWindow &window, NVGcontext *context, CLGLContext &fbContext) :
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

    float r = window_.getPixelRatio();
    float w = window_.getSize().width;
    float h = window_.getSize().height;
//    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, kb::gl::frame_buf));
    nvgSave(context_);
    nvgBeginFrame(context_, w, h, r);
}

void NanoVGContext::end() {
    nvgEndFrame(context_);
    nvgRestore(context_);
    fbContext_.end();
}
} /* namespace kb */
