// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "glcontext.hpp"

namespace cv {
namespace v4d {
namespace detail {
GLContext::GLContext(FrameBufferContext& fbContext) :
        mainFbContext_(fbContext), glFbContext_(fbContext.getV4D(), "OpenGL", fbContext) {
#ifdef __EMSCRIPTEN__
    run_sync_on_main<19>([&,this](){
        mainFbContext_.initWebGLCopy(fbCtx().getIndex());
    });
#endif
}

void GLContext::render(std::function<void(const cv::Size&)> fn) {
    run_sync_on_main<15>([&,this](){
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
#ifdef __EMSCRIPTEN__
            //Preserve the clear color state though it is a bit costly. We don't want to interfere.
            GLfloat cColor[4];
            glGetFloatv(GL_COLOR_CLEAR_VALUE, cColor);
            glClearColor(0,0,0,0);
            glClear(GL_COLOR_BUFFER_BIT);
            glClearColor(cColor[0], cColor[1], cColor[2], cColor[3]);
#endif
            fn(fbCtx().size());
        }
        if(!fbCtx().isShared()) {
#ifdef __EMSCRIPTEN__
            mainFbContext_.doWebGLCopy(fbCtx());
#else
            UMat tmp;
            fbCtx().copyTo(tmp);
            mainFbContext_.copyFrom(tmp);
#endif
        }
    });
}

FrameBufferContext& GLContext::fbCtx() {
    return glFbContext_;
}

}
}
}
