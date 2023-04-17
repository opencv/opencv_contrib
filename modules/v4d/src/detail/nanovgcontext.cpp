// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "nanovgcontext.hpp"

#include "opencv2/v4d/v4d.hpp"

namespace cv {
namespace viz {
namespace detail {
NanoVGContext::NanoVGContext(V4D& v4d, NVGcontext* context, FrameBufferContext& fbContext) :
        v4d_(v4d), context_(context), clglContext_(fbContext) {
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
#ifndef OPENCV_V4D_ES_VERSION
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
#ifndef OPENCV_V4D_ES_VERSION
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
    float w = v4d_.getFrameBufferSize().width;
    float h = v4d_.getFrameBufferSize().height;
    float r = v4d_.getXPixelRatio();

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
