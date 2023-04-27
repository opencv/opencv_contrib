// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "nanoguicontext.hpp"

namespace cv {
namespace v4d {
namespace detail {
NanoguiContext::NanoguiContext(V4D& v4d, FrameBufferContext& fbContext) :
        mainFbContext_(fbContext), nguiFbContext_(v4d, "NanoGUI", fbContext) {
    fbCtx().makeCurrent();
    screen_ = new nanogui::Screen();
    screen_->initialize(nguiFbContext_.getGLFWWindow(), false);
    form_ = new cv::v4d::FormHelper(screen_);
    fbCtx().resizeWindow(fbCtx().getSize());
    fbCtx().makeNoneCurrent();
}

void NanoguiContext::render() {
    screen().draw_widgets();
}

void NanoguiContext::build(std::function<void(cv::v4d::FormHelper&)> fn) {
    fbCtx().makeCurrent();
    fn(form());
    screen().perform_layout();
    fbCtx().makeNoneCurrent();
}

nanogui::Screen& NanoguiContext::screen() {
    return *screen_;
}

cv::v4d::FormHelper& NanoguiContext::form() {
    return *form_;
}

FrameBufferContext& NanoguiContext::fbCtx() {
    return nguiFbContext_;
}
}
}
}
