// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "nanovgcontext.hpp"

#include "../viz2d.hpp"

namespace cv {
namespace viz {
namespace detail {
NanoVGContext::NanoVGContext(Viz2D& v2d, NVGcontext* context, FrameBufferContext& fbContext) :
        v2d_(v2d), context_(context), clglContext_(fbContext) {
    //FIXME workaround for first frame color glitch
    cv::UMat tmp;
    FrameBufferContext::FrameBufferScope fbScope(clglContext_, tmp);
}

void NanoVGContext::render(std::function<void(const cv::Size&)> fn) {
#ifndef __EMSCRIPTEN__
    CLExecScope_t scope(clglContext_.getCLExecContext());
#endif
    FrameBufferContext::GLScope glScope(clglContext_);
    NanoVGContext::Scope nvgScope(*this);
    cv::viz::nvg::detail::NVG::initializeContext(context_), fn(clglContext_.getSize());
}

void push() {
#ifndef VIZ2D_USE_ES3
    GL_CHECK(glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS));
    GL_CHECK(glPushAttrib(GL_ALL_ATTRIB_BITS));
    GL_CHECK(glMatrixMode(GL_MODELVIEW));
    GL_CHECK(glPushMatrix());
    GL_CHECK(glMatrixMode(GL_PROJECTION));
    GL_CHECK(glPushMatrix());
    GL_CHECK(glMatrixMode(GL_TEXTURE));
    GL_CHECK(glPushMatrix());
#endif
}

void pop() {
#ifndef VIZ2D_USE_ES3
    GL_CHECK(glMatrixMode(GL_TEXTURE));
    GL_CHECK(glPopMatrix());
    GL_CHECK(glMatrixMode(GL_PROJECTION));
    GL_CHECK(glPopMatrix());
    GL_CHECK(glMatrixMode(GL_MODELVIEW));
    GL_CHECK(glPopMatrix());
    GL_CHECK(glPopClientAttrib());
    GL_CHECK(glPopAttrib());
#endif
}

void NanoVGContext::begin() {
    push();
    float w = v2d_.getFrameBufferSize().width;
    float h = v2d_.getFrameBufferSize().height;
    float r = v2d_.getXPixelRatio();

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
    pop();
}
}
}
}
