// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_NANOGUICONTEXT_HPP_
#define SRC_OPENCV_NANOGUICONTEXT_HPP_

#include "framebuffercontext.hpp"


#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

namespace nanogui {
    class Screen;
}
namespace cv {
namespace v4d {
class FormHelper;
namespace detail {
/*!
 * Used to setup a nanogui context
 */
class NanoguiContext {
    nanogui::Screen* screen_;
    cv::v4d::FormHelper* form_;
    FrameBufferContext& mainFbContext_;
    FrameBufferContext nguiFbContext_;
    NVGcontext* context_;
    cv::TickMeter tick_;
    float fps_ = 0;
    bool first_ = true;
    cv::UMat copyBuffer_;
public:
    NanoguiContext(FrameBufferContext& fbContext);
    void render(bool print, bool graphical);
    void build(std::function<void(cv::v4d::FormHelper&)> fn);
    nanogui::Screen& screen();
    cv::v4d::FormHelper& form();
    FrameBufferContext& fbCtx();
};
}
}
}

#endif /* SRC_OPENCV_NANOGUICONTEXT_HPP_ */
