// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "nanovgcontext.hpp"
#include "opencv2/v4d/nvg.hpp"

namespace cv {
namespace v4d {
namespace detail {

NanoVGContext::NanoVGContext(V4D& v4d, FrameBufferContext& fbContext) :
        v4d_(v4d), mainFbContext_(fbContext), nvgFbContext_(v4d, "NanoVG", fbContext), context_(
                nullptr) {
    UMat tmp(fbCtx().size(), CV_8UC4);

    run_sync_on_main<13>([this, &tmp]() {
        {
            //Workaround for first frame glitch
            FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
            FrameBufferContext::FrameBufferScope fbScope(fbCtx(), tmp);
        }
        {
            FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
            screen_ = new nanogui::Screen();
            screen_->initialize(fbCtx().getGLFWWindow(), false);
            fbCtx().setWindowSize(fbCtx().size());
            context_ = screen_->nvg_context();
            form_ = new cv::v4d::FormHelper(screen_);
            if (!context_)
                throw std::runtime_error("Could not initialize NanoVG!");
       }
#ifdef __EMSCRIPTEN__
        mainFbContext_.initWebGLCopy(fbCtx());
#endif
    });

    tmp.release();
}

void NanoVGContext::render(std::function<void(const cv::Size&)> fn) {
    run_sync_on_main<14>([this, fn](){
#ifndef __EMSCRIPTEN__
        if(!fbCtx().isShared()) {
            UMat tmp;
            mainFbContext_.copyTo(tmp);
            fbCtx().copyFrom(tmp);
        }
#endif
        {
            FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
            glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
            NanoVGContext::Scope nvgScope(*this);
            cv::v4d::nvg::detail::NVG::initializeContext(context_);
            fn(fbCtx().size());
        }
        {
            if(!fbCtx().isShared()) {
#ifdef __EMSCRIPTEN__
                mainFbContext_.doWebGLCopy(fbCtx());
#else
                UMat tmp;
                fbCtx().copyTo(tmp);
                mainFbContext_.copyFrom(tmp);
#endif
            }
        }
    });
}

void NanoVGContext::begin() {
    float w = fbCtx().size().width;
    float h = fbCtx().size().height;
    float r = fbCtx().pixelRatioX();

    nvgSave(context_);
    nvgBeginFrame(context_, w, h, r);
//FIXME mirroring with text somehow doesn't work
//    nvgTranslate(context_, 0, h);
//    nvgScale(context_, 1, -1);
}

void NanoVGContext::end() {
    fbCtx().makeCurrent();
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
