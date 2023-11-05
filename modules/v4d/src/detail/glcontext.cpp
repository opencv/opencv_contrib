// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/detail/glcontext.hpp"
#include "opencv2/v4d/detail/gl.hpp"

namespace cv {
namespace v4d {
namespace detail {
GLContext::GLContext(const int32_t& idx, cv::Ptr<FrameBufferContext> fbContext) :
        idx_(idx), mainFbContext_(fbContext), glFbContext_(new FrameBufferContext(*fbContext->getV4D(), "OpenGL" + std::to_string(idx), fbContext)) {
#ifdef __EMSCRIPTEN__
    run_sync_on_main<19>([&,this](){
        mainFbContext_->initWebGLCopy(fbCtx()->getIndex());
    });
#endif
}

void GLContext::execute(std::function<void()> fn) {
    run_sync_on_main<15>([this, fn](){
#ifndef __EMSCRIPTEN__
        if(!fbCtx()->hasParent()) {
            UMat tmp;
            mainFbContext_->copyTo(tmp);
            fbCtx()->copyFrom(tmp);
        }
#endif
        {
            FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
            GL_CHECK(glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
#ifdef __EMSCRIPTEN__
            //Preserve the clear color state though it is a bit costly. We don't want to interfere.
            GLfloat cColor[4];
            GL_CHECK(glGetFloatv(GL_COLOR_CLEAR_VALUE, cColor));
            GL_CHECK(glClearColor(0,0,0,0));
            GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));
            GL_CHECK(glClearColor(cColor[0], cColor[1], cColor[2], cColor[3]));
#endif
            fn();
        }
        if(!fbCtx()->hasParent()) {
#ifdef __EMSCRIPTEN__
            mainFbContext_->doWebGLCopy(fbCtx());
#else
            UMat tmp;
            fbCtx()->copyTo(tmp);
            mainFbContext_->copyFrom(tmp);
#endif
        }
    });
}

const int32_t& GLContext::getIndex() const {
	return idx_;
}
cv::Ptr<FrameBufferContext> GLContext::fbCtx() {
    return glFbContext_;
}

}
}
}
