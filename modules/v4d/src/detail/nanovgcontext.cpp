// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "nanovgcontext.hpp"
#include "opencv2/v4d/nvg.hpp"
#include "nanovg_gl.h"

namespace cv {
namespace v4d {
namespace detail {

NanoVGContext::NanoVGContext(FrameBufferContext& fbContext) :
        mainFbContext_(fbContext), nvgFbContext_(*fbContext.getV4D(), "NanoVG", fbContext), context_(
                nullptr) {
    run_sync_on_main<13>([this]() {
        {
            FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
            context_ = nvgCreateGL3(NVG_ANTIALIAS | NVG_STENCIL_STROKES | NVG_DEBUG);
            if (!context_)
                throw std::runtime_error("Could not initialize NanoVG!");
        }
#ifdef __EMSCRIPTEN__
        run_sync_on_main<12>([&,this](){
            mainFbContext_.initWebGLCopy(fbCtx().getIndex());
        });
#endif
    });
}

void NanoVGContext::render(std::function<void(const cv::Size&)> fn) {
    run_sync_on_main<14>([this, fn]() {
#ifndef __EMSCRIPTEN__
        if (!fbCtx().isShared()) {
            UMat tmp;
            mainFbContext_.copyTo(tmp);
            fbCtx().copyFrom(tmp);
        }
#endif
        {
            FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
            glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
#ifdef __EMSCRIPTEN__
            glClearColor(0,0,0,0);
            glClear(GL_COLOR_BUFFER_BIT);
#endif
        NanoVGContext::Scope nvgScope(*this);
        cv::v4d::nvg::detail::NVG::initializeContext(context_);
        fn(fbCtx().size());
    }
    {
        if (!fbCtx().isShared()) {
#ifdef __EMSCRIPTEN__
                mainFbContext_.doWebGLCopy(fbCtx());
#else
        UMat tmp;
        fbCtx().copyTo(tmp);
        mainFbContext_.copyFrom(tmp);
#endif
    }
}
}   );
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
