// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "glcontext.hpp"

namespace cv {
namespace v4d {
namespace detail {
GLContext::GLContext(V4D& v4d, FrameBufferContext& fbContext) :
        mainFbContext_(fbContext), glFbContext_(v4d, "OpenGL", fbContext) {
}

void GLContext::render(std::function<void(const cv::Size&)> fn) {
    run_sync_on_main([&,this](){
#ifdef __EMSCRIPTEN__
    fb_.create(mainFbContext_.getSize(), CV_8UC4);
    preFB_.create(mainFbContext_.getSize(), CV_8UC4);
    postFB_.create(mainFbContext_.getSize(), CV_8UC4);
    {
        FrameBufferContext::GLScope mainGlScope(mainFbContext_);
        FrameBufferContext::FrameBufferScope fbScope(mainFbContext_, fb_);
        fb_.copyTo(preFB_);
    }
    {
        FrameBufferContext::GLScope glGlScope(glFbContext_);
        FrameBufferContext::FrameBufferScope fbScope(glFbContext_, fb_);
        preFB_.copyTo(fb_);
    }
#endif
    {
            FrameBufferContext::GLScope glScope(fbCtx());
            fn(fbCtx().getSize());
    }
#ifdef __EMSCRIPTEN__
    {
        FrameBufferContext::GLScope glScope(glFbContext_);
        FrameBufferContext::FrameBufferScope fbScope(glFbContext_, fb_);
        fb_.copyTo(postFB_);
    }
    {
        FrameBufferContext::GLScope mainGlScope(mainFbContext_);
        FrameBufferContext::FrameBufferScope fbScope(mainFbContext_, fb_);
        postFB_.copyTo(fb_);
    }
#endif
    });
}

FrameBufferContext& GLContext::fbCtx() {
    return glFbContext_;
}

}
}
}
