// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "nanovgcontext.hpp"
#include "opencv2/v4d/util.hpp"
#include "opencv2/v4d/nvg.hpp"

namespace cv {
namespace v4d {
namespace detail {
#if !defined(GL_RGBA_FLOAT_MODE)
#  define GL_RGBA_FLOAT_MODE 0x8820
#endif


#ifdef __EMSCRIPTEN__
#include <emscripten/threading.h>

static void initmenvg(NanoVGContext* initiatenvg) {
    initiatenvg->init();
}
#endif

NanoVGContext::NanoVGContext(V4D& v4d, FrameBufferContext& fbContext) :
        v4d_(v4d), screen_(nullptr), context_(nullptr), mainFbContext_(fbContext), nvgFbContext_(v4d, "NanoVG", fbContext) {
#ifdef __EMSCRIPTEN__
    emscripten_sync_run_in_main_runtime_thread(EM_FUNC_SIG_VI, initmenvg, this);
#else
    init();
#endif
}

void NanoVGContext::init() {
    fbCtx().makeCurrent();
    screen_ = new nanogui::Screen();
    screen_->initialize(fbCtx().getGLFWWindow(), false);
    context_ = screen_->nvg_context();
    fbCtx().resizeWindow(fbCtx().getSize());
    fbCtx().makeNoneCurrent();
}

void NanoVGContext::render(std::function<void(const cv::Size&)> fn) {
#ifdef __EMSCRIPTEN__
    {
        FrameBufferContext::GLScope mainGlScope(mainFbContext_);
        FrameBufferContext::FrameBufferScope fbScope(mainFbContext_, fb_);
        fb_.copyTo(preFB_);
    }
    {
        FrameBufferContext::GLScope nvgGlScope(nvgFbContext_);
        FrameBufferContext::FrameBufferScope fbScope(nvgFbContext_, fb_);
        preFB_.copyTo(fb_);
    }
#endif
    {
        FrameBufferContext::GLScope glScope(fbCtx());
        NanoVGContext::Scope nvgScope(*this);
        cv::v4d::nvg::detail::NVG::initializeContext(context_);
        fn(fbCtx().getSize());
//        fbCtx().makeCurrent();
//        fbCtx().blitFrameBufferToScreen(cv::Rect(0,0, fbCtx().getSize().width, fbCtx().getSize().height), fbCtx().getSize(), false);
//        glfwSwapBuffers(fbCtx().getGLFWWindow());
    }
#ifdef __EMSCRIPTEN__
    {
        FrameBufferContext::GLScope nvgGlScope(nvgFbContext_);
        FrameBufferContext::FrameBufferScope fbScope(nvgFbContext_, fb_);
        fb_.copyTo(postFB_);
    }
    {
        FrameBufferContext::GLScope mainGlScope(mainFbContext_);
        FrameBufferContext::FrameBufferScope fbScope(mainFbContext_, fb_);
        postFB_.copyTo(fb_);
    }
#endif
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
