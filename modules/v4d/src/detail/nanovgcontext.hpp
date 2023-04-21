// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_NANOVGCONTEXT_HPP_
#define SRC_OPENCV_NANOVGCONTEXT_HPP_

#include "framebuffercontext.hpp"
#include <nanogui/nanogui.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#ifndef OPENCV_V4D_USE_ES3
#define NANOGUI_USE_OPENGL
#else
#define NANOGUI_USE_GLES
#define NANOGUI_GLES_VERSION 3
#endif

#include <nanogui/opengl.h>
#include "opencv2/v4d/util.hpp"
#include "opencv2/v4d/nvg.hpp"

namespace cv {
namespace viz {
namespace detail {
/*!
 * Used to setup a nanovg context
 */
class NanoVGContext {
    NVGcontext* context_;
    FrameBufferContext& mainFbContext_;
    FrameBufferContext& nvgFbContext_;
    cv::UMat preFB_;
    cv::UMat fb_;
    cv::UMat postFB_;
public:
    /*!
     * Makes sure #NanoVGContext::begin and #NanoVGContext::end are both called
     */
    class Scope {
        NanoVGContext& ctx_;
    public:
        /*!
         * Setup NanoVG rendering
         * @param ctx The corresponding #NanoVGContext
         */
        Scope(NanoVGContext& ctx) :
                ctx_(ctx) {
            ctx_.begin();
        }
        /*!
         * Tear-down NanoVG rendering
         */
        ~Scope() {
            ctx_.end();
        }
    };
    /*!
     * Creates a NanoVGContext
     * @param v4d The V4D object used in conjunction with this context
     * @param context The native NVGContext
     * @param fbContext The framebuffer context
     */
    NanoVGContext(NVGcontext* context, FrameBufferContext& fbContext);
    /*!
     * Execute function object fn inside a nanovg context.
     * The context takes care of setting up opengl and nanovg states.
     * A function object passed like that can use the functions in cv::viz::nvg.
     * @param fn A function that is passed the size of the framebuffer
     * and performs drawing using cv::viz::nvg
     */
    void render(std::function<void(const cv::Size&)> fn);
private:
    /*!
     * Setup NanoVG context
     */
    void begin();
    /*!
     * Tear down NanoVG context
     */
    void end();
};
}
}
}

#endif /* SRC_OPENCV_NANOVGCONTEXT_HPP_ */
