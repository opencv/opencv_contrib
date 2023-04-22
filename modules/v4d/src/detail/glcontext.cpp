// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "glcontext.hpp"

namespace cv {
namespace viz {
namespace detail {
GLContext::GLContext(FrameBufferContext& fbContext) :
        mainFbContext_(fbContext), glFbContext_(fbContext) {
}

void GLContext::render(std::function<void(const cv::Size&)> fn) {
#ifndef __EMSCRIPTEN__
    CLExecScope_t scope(glFbContext_.getCLExecContext());
#endif
    FrameBufferContext::GLScope glScope(glFbContext_);
    fn(glFbContext_.getSize());
}

FrameBufferContext& GLContext::fbCtx() {
    return glFbContext_;
}

}
}
}
