// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>
#include "opencv2/v4d/v4d.hpp"
#include "nanoguicontext.hpp"

namespace cv {
namespace v4d {
namespace detail {

NanoguiContext::NanoguiContext(V4D& v4d, FrameBufferContext& fbContext) :
        mainFbContext_(fbContext), nguiFbContext_(v4d, "NanoGUI", fbContext) {
    run_sync_on_main([this](){ init(); });
}

void NanoguiContext::init() {
    FrameBufferContext::GLScope glScope(fbCtx());
    screen_ = new nanogui::Screen();
    screen_->initialize(nguiFbContext_.getGLFWWindow(), false);
    form_ = new cv::v4d::FormHelper(screen_);
}

void NanoguiContext::render() {
    run_sync_on_main([&,this](){
#ifdef __EMSCRIPTEN__
    fb_.create(mainFbContext_.size(), CV_8UC4);
    preFB_.create(mainFbContext_.size(), CV_8UC4);
    postFB_.create(mainFbContext_.size(), CV_8UC4);
    {
        FrameBufferContext::GLScope mainGlScope(mainFbContext_);
        FrameBufferContext::FrameBufferScope fbScope(mainFbContext_, fb_);
        fb_.copyTo(preFB_);
    }
    {
        FrameBufferContext::GLScope glGlScope(fbCtx());
        FrameBufferContext::FrameBufferScope fbScope(fbCtx(), fb_);
        preFB_.copyTo(fb_);
    }
#endif
    {
        FrameBufferContext::GLScope glScope(fbCtx());
        screen().draw_widgets();
    }
#ifdef __EMSCRIPTEN__
    {
        FrameBufferContext::GLScope glScope(fbCtx());
        FrameBufferContext::FrameBufferScope fbScope(fbCtx(), fb_);
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

void NanoguiContext::build(std::function<void(cv::v4d::FormHelper&)> fn) {
    run_sync_on_main([fn,this](){
        FrameBufferContext::GLScope glScope(fbCtx());
        fn(form());
        screen().perform_layout();
    });
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
