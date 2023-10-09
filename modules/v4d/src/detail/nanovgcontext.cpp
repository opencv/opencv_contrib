// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/detail/nanovgcontext.hpp"
#include "opencv2/v4d/nvg.hpp"
#include "opencv2/v4d/detail/gl.hpp"
#include "nanovg_gl.h"

namespace cv {
namespace v4d {
namespace detail {

NanoVGContext::NanoVGContext(cv::Ptr<FrameBufferContext> fbContext) :
        mainFbContext_(fbContext), nvgFbContext_(new FrameBufferContext(*fbContext->getV4D(), "NanoVG", *fbContext)), context_(
                nullptr) {
    run_sync_on_main<13>([this]() {
        {
            FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
#if defined(OPENCV_V4D_USE_ES3) || defined(EMSCRIPTEN)
            context_ = nvgCreateGLES3(NVG_ANTIALIAS | NVG_STENCIL_STROKES);
#else
            context_ = nvgCreateGL3(NVG_ANTIALIAS | NVG_STENCIL_STROKES);
#endif
            if (!context_)
                throw std::runtime_error("Could not initialize NanoVG!");
#ifdef __EMSCRIPTEN__
            nvgCreateFont(context_, "icons", "assets/fonts/entypo.ttf");
            nvgCreateFont(context_, "sans", "assets/fonts/Roboto-Regular.ttf");
            nvgCreateFont(context_, "sans-bold", "/assets/fonts/Roboto-Bold.ttf");
#else
            nvgCreateFont(context_, "icons", "modules/v4d/assets/fonts/entypo.ttf");
            nvgCreateFont(context_, "sans", "modules/v4d/assets/fonts/Roboto-Regular.ttf");
            nvgCreateFont(context_, "sans-bold", "modules/v4d/assets/fonts/Roboto-Bold.ttf");
#endif
#ifdef __EMSCRIPTEN__
            mainFbContext_.initWebGLCopy(fbCtx()->getIndex());
#endif
        }
    });
}

void NanoVGContext::execute(std::function<void()> fn) {
    run_sync_on_main<14>([this, fn]() {
#ifndef __EMSCRIPTEN__
        if (!fbCtx()->isShared()) {
            UMat tmp;
            mainFbContext_->copyTo(tmp);
            fbCtx()->copyFrom(tmp);
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
            fn();
        }
        if (!fbCtx()->isShared()) {
#ifdef __EMSCRIPTEN__
            mainFbContext_.doWebGLCopy(fbCtx());
#else
            UMat tmp;
            fbCtx()->copyTo(tmp);
            mainFbContext_->copyFrom(tmp);
#endif
        }
    });
}

void NanoVGContext::begin() {
    float w = fbCtx()->size().width;
    float h = fbCtx()->size().height;
    float r = fbCtx()->pixelRatioX();

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

cv::Ptr<FrameBufferContext> NanoVGContext::fbCtx() {
    return nvgFbContext_;
}
}
}
}
