// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "nanovgcontext.hpp"

namespace cv {
namespace v4d {
namespace detail {
NanoVGContext::NanoVGContext(FrameBufferContext& fbContext) :
        mainFbContext_(fbContext), nvgFbContext_(fbContext) {
#ifndef __EMSCRIPTEN__
    CLExecScope_t scope(nvgFbContext_.getCLExecContext());
#endif
    FrameBufferContext::GLScope nvgGlScope(nvgFbContext_);
    screen_ = new nanogui::Screen();
    screen_->initialize(nvgFbContext_.getGLFWWindow(), false);
    context_ = screen_->nvg_context();
}

void NanoVGContext::render(std::function<void(const cv::Size&)> fn) {
    {
#ifndef __EMSCRIPTEN__
        CLExecScope_t scope(mainFbContext_.getCLExecContext());
#endif
        FrameBufferContext::GLScope mainGlScope(mainFbContext_);
        FrameBufferContext::FrameBufferScope fbScope(mainFbContext_, fb_);
        fb_.copyTo(preFB_);
    }
    {
#ifndef __EMSCRIPTEN__
        CLExecScope_t scope(nvgFbContext_.getCLExecContext());
#endif
        FrameBufferContext::GLScope nvgGlScope(nvgFbContext_);
        FrameBufferContext::FrameBufferScope fbScope(nvgFbContext_, fb_);
        preFB_.copyTo(fb_);
    }
    {
#ifndef __EMSCRIPTEN__
        CLExecScope_t scope(nvgFbContext_.getCLExecContext());
#endif
        FrameBufferContext::GLScope nvgGlScope(nvgFbContext_);
        NanoVGContext::Scope nvgScope(*this);
        cv::v4d::nvg::detail::NVG::initializeContext(context_);
        fn(nvgFbContext_.getSize());
    }
    {
#ifndef __EMSCRIPTEN__
        CLExecScope_t scope(nvgFbContext_.getCLExecContext());
#endif
        FrameBufferContext::GLScope nvgGlScope(nvgFbContext_);
        FrameBufferContext::FrameBufferScope fbScope(nvgFbContext_, fb_);
        fb_.copyTo(postFB_);
    }
    {
#ifndef __EMSCRIPTEN__
        CLExecScope_t scope(mainFbContext_.getCLExecContext());
#endif
        FrameBufferContext::GLScope mainGlScope(mainFbContext_);
        FrameBufferContext::FrameBufferScope fbScope(mainFbContext_, fb_);
        postFB_.copyTo(fb_);
    }

}

void NanoVGContext::begin() {
    float w = nvgFbContext_.getSize().width;
    float h = nvgFbContext_.getSize().height;
    float r = nvgFbContext_.getXPixelRatio();

    nvgSave(context_);
    nvgBeginFrame(context_, w, h, r);
//FIXME mirroring with text somehow doesn't work
//    nvgTranslate(context_, 0, h);
//    nvgScale(context_, 1, -1);
    GL_CHECK(glViewport(0, 0, w, h));
}

void NanoVGContext::end() {
    //FIXME make nvgCancelFrame possible
    nvgEndFrame(context_);
    nvgRestore(context_);
}

FrameBufferContext& NanoVGContext::fbCtx() {
    return nvgFbContext_;
}
}
}
}
