// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_GLCONTEXT_HPP_
#define SRC_OPENCV_GLCONTEXT_HPP_

#include "framebuffercontext.hpp"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

namespace cv {
namespace v4d {
namespace detail {
/*!
 * Used to setup a nanovg context
 */
class GLContext {
    FrameBufferContext& mainFbContext_;
    FrameBufferContext glFbContext_;
public:
     /*!
     * Creates a OpenGL Context
     * @param fbContext The framebuffer context
     */
    GLContext(FrameBufferContext& fbContext);
    /*!
     * Execute function object fn inside a gl context.
     * The context takes care of setting up opengl states.
     * @param fn A function that is passed the size of the framebuffer
     * and performs drawing using opengl
     */
    void render(std::function<void(const cv::Size&)> fn);

    FrameBufferContext& fbCtx();
};
}
}
}

#endif /* SRC_OPENCV_GLCONTEXT_HPP_ */
