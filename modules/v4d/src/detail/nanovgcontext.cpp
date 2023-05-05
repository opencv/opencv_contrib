// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "nanovgcontext.hpp"
#include "opencv2/v4d/v4d.hpp"
#include "nanovg.h"

#ifdef OPENCV_V4D_USE_ES3
#  define NANOVG_GLES3_IMPLEMENTATION 1
#  define NANOVG_GLES3 1
#else
#  define NANOVG_GL3 1
#  define NANOVG_GL3_IMPLEMENTATION 1
#endif
#define NANOVG_GL_USE_UNIFORMBUFFER 1
#include "nanovg_gl.h"

namespace cv {
namespace v4d {
namespace detail {

NanoVGContext::NanoVGContext(V4D& v4d, FrameBufferContext& fbContext) :
        v4d_(v4d), context_(nullptr), mainFbContext_(fbContext), nvgFbContext_(v4d, "NanoVG", fbContext) {
    run_sync_on_main<13>([this](){ init(); });
}

void NanoVGContext::init() {
    FrameBufferContext::GLScope glScope(fbCtx(), GL_DRAW_FRAMEBUFFER);
    screen_ = new nanogui::Screen();
    screen_->initialize(fbCtx().getGLFWWindow(), false);
    fbCtx().setWindowSize(fbCtx().size());
    context_ = screen_->nvg_context();

//    FrameBufferContext::GLScope glScope(fbCtx());
//    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//    glEnable(GL_BLEND);
//    glEnable(GL_STENCIL_TEST);
//    glEnable(GL_DEPTH_TEST);
//    glDisable(GL_SCISSOR_TEST);
//    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
//    glStencilMask(0xffffffff);
//    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
//    glStencilFunc(GL_ALWAYS, 0, 0xffffffff);
//    glStencilMask(0xFF);
//
//    int flags = NVG_ANTIALIAS | NVG_STENCIL_STROKES | NVG_DEBUG;
//#ifdef OPENCV_V4D_USE_ES3 || __EMSCRIPTEN__
//    context_ = nvgCreateGLES3(flags);
//#else
//    context_ = nvgCreateGL3(flags);
//#endif
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
        FrameBufferContext::GLScope glScope(fbCtx());
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
