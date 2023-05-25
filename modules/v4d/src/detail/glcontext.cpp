// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "glcontext.hpp"
namespace cv {
namespace v4d {
namespace detail {
GLContext::GLContext(V4D& v4d, FrameBufferContext& fbContext) :
        v4d_(v4d), mainFbContext_(fbContext), glFbContext_(v4d, "OpenGL", fbContext) {
}

void GLContext::render(std::function<void(const cv::Size&)> fn) {
    run_sync_on_main<15>([&,this](){
        FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
        fn(fbCtx().size());
    });
}

FrameBufferContext& GLContext::fbCtx() {
    return glFbContext_;
}

}
}
}
