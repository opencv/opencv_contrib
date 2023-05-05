// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "nanovgcontext.hpp"
#include "opencv2/v4d/v4d.hpp"

namespace cv {
namespace v4d {
namespace detail {

NanoVGContext::NanoVGContext(V4D& v4d, FrameBufferContext& fbContext) :
        v4d_(v4d), context_(nullptr), mainFbContext_(fbContext), nvgFbContext_(v4d, "NanoVG", fbContext) {
    run_sync_on_main<13>([this](){ init(); });
}

void NanoVGContext::init() {
//    GL_CHECK(glEnable(GL_DEPTH_TEST));
//    GL_CHECK(glDepthFunc(GL_LESS));
//    GL_CHECK(glEnable(GL_STENCIL_TEST));
//    GL_CHECK(glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP));
//    GL_CHECK(glStencilFunc(GL_ALWAYS, 0, 0xffffffff));
//    GL_CHECK(glStencilMask(0x00));
    FrameBufferContext::GLScope glScope(fbCtx(), GL_DRAW_FRAMEBUFFER);
    glClear(GL_STENCIL_BUFFER_BIT);
    screen_ = new nanogui::Screen();
    screen_->initialize(fbCtx().getGLFWWindow(), false);
    fbCtx().setWindowSize(fbCtx().size());
    context_ = screen_->nvg_context();

    if (!context_)
        throw std::runtime_error("Could not initialize NanoVG!");
}

void NanoVGContext::render(std::function<void(const cv::Size&)> fn) {
    run_sync_on_main<14>([&,this](){
#ifdef __EMSCRIPTEN__
//    {
//        FrameBufferContext::GLScope mainGlScope(mainFbContext_);
//        FrameBufferContext::FrameBufferScope fbScope(mainFbContext_, fb_);
//        fb_.copyTo(preFB_);
//    }
//    {
//        FrameBufferContext::GLScope nvgGlScope(nvgFbContext_);
//        FrameBufferContext::FrameBufferScope fbScope(nvgFbContext_, fb_);
//        preFB_.copyTo(fb_);
//    }
#endif
    {
//        GL_CHECK(glEnable(GL_DEPTH_TEST));
//        GL_CHECK(glDepthFunc(GL_LESS));
//        GL_CHECK(glEnable(GL_STENCIL_TEST));
//        GL_CHECK(glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP));
//        GL_CHECK(glStencilFunc(GL_ALWAYS, 0, 0xffffffff));
//        GL_CHECK(glStencilMask(0x00));
        FrameBufferContext::GLScope glScope(fbCtx());
        glClear(GL_STENCIL_BUFFER_BIT);
//        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
//        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        NanoVGContext::Scope nvgScope(*this);
        cv::v4d::nvg::detail::NVG::initializeContext(context_);
        fn(fbCtx().size());
    }
#ifdef __EMSCRIPTEN__
//    {
//        FrameBufferContext::GLScope nvgGlScope(nvgFbContext_);
//        FrameBufferContext::FrameBufferScope fbScope(nvgFbContext_, fb_);
//        fb_.copyTo(postFB_);
//    }
//    {
//        FrameBufferContext::GLScope mainGlScope(mainFbContext_);
//        FrameBufferContext::FrameBufferScope fbScope(mainFbContext_, fb_);
//        postFB_.copyTo(fb_);
//    }
#endif
    });
}

void NanoVGContext::begin() {
    float w = fbCtx().size().width;
    float h = fbCtx().size().height;
    float r = fbCtx().getXPixelRatio();

    nvgSave(context_);
//    glClear(GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);
    nvgBeginFrame(context_, w, h, r);
//FIXME mirroring with text somehow doesn't work
//    nvgTranslate(context_, 0, h);
//    nvgScale(context_, 1, -1);
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
