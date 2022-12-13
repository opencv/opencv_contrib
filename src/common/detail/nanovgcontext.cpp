#include "nanovgcontext.hpp"

#include "../viz2d.hpp"

namespace kb {
namespace viz2d {
namespace detail {
NanoVGContext::NanoVGContext(Viz2D &v2d, NVGcontext *context, CLGLContext &fbContext) :
        v2d_(v2d), context_(context), clglContext_(fbContext) {
    nvgCreateFont(context_, "libertine", "assets/LinLibertine_RB.ttf");

    //FIXME workaround for first frame color glitch
    cv::UMat tmp;
    CLGLContext::FrameBufferScope fbScope(clglContext_, tmp);
}

void NanoVGContext::render(std::function<void(const cv::Size&)> fn) {
    CLExecScope_t scope(clglContext_.getCLExecContext());
    CLGLContext::GLScope glScope(clglContext_);
    NanoVGContext::Scope nvgScope(*this);
    kb::nvg::detail::set_current_context(context_),
    fn(clglContext_.getSize());
}

void NanoVGContext::begin() {
    float w = v2d_.getVideoFrameSize().width;
    float h = v2d_.getVideoFrameSize().height;
    float r = v2d_.getXPixelRatio();

    nvgSave(context_);
    nvgBeginFrame(context_, w, h, r);
    GL_CHECK(glViewport(0, 0, w, h));
}

void NanoVGContext::end() {
    //FIXME make nvgCancelFrame possible
    nvgEndFrame(context_);
    nvgRestore(context_);
}
}
}
}
