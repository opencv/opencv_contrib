// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/v4d.hpp"
#include "nanoguicontext.hpp"
#include "nanovgcontext.hpp"

namespace cv {
namespace v4d {
namespace detail {

NanoguiContext::NanoguiContext(FrameBufferContext& fbContext) :
        mainFbContext_(fbContext), context_(nullptr) {
    UMat tmp(fbContext.size(), CV_8UC4);

    run_sync_on_main<21>([&, this]() {
        fbContext.makeCurrent();
        GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0))
        screen_ = new nanogui::Screen();
        screen_->initialize(fbContext.getGLFWWindow(), false);
        context_ = screen_->nvg_context();
        form_ = new cv::v4d::FormHelper(screen_);
        if (!context_)
            throw std::runtime_error("Could not initialize NanoVG!");
        GL_CHECK(glFlush());
        GL_CHECK(glFinish());
    });

    tmp.release();
}

void NanoguiContext::render() {
        GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0))
        GL_CHECK(glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
        screen().draw_widgets();
        GL_CHECK(glFlush());
        GL_CHECK(glFinish());
}

void NanoguiContext::updateFps(bool print, bool graphical) {
    if (!first_) {
        tick_.stop();

        if (tick_.getTimeMilli() > 50) {
            if(print) {
                cerr << "FPS : " << (fps_ = tick_.getFPS());
#ifndef __EMSCRIPTEN__
                cerr << '\r';
#else
                cerr << endl;
#endif
            }
            tick_.reset();
        }

        if (graphical) {
            //FIXME use proper scopes
            GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0))
            GL_CHECK(glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
            float w = mainFbContext_.size().width;
            float h = mainFbContext_.size().height;
            float r = mainFbContext_.pixelRatioX();

            nvgSave(context_);
            nvgBeginFrame(context_, w, h, r);
            cv::v4d::nvg::detail::NVG::initializeContext(context_);
            using namespace cv::v4d::nvg;
            string txt = "FPS: " + std::to_string(fps_);
            beginPath();
            roundedRect(5, 5, 15 * txt.size() + 5, 30, 5);
            fillColor(cv::Scalar(255, 255, 255, 180));
            fill();

            fontSize(30.0f);
            fontFace("mono");
            fillColor(cv::Scalar(90, 90, 90, 255));
            textAlign(NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
            text(10, 20, txt.c_str(), nullptr);

            nvgEndFrame(context_);
            nvgRestore(context_);
        }
    }
    first_ = false;
    tick_.start();
}

void NanoguiContext::build(std::function<void(cv::v4d::FormHelper&)> fn) {
    run_sync_on_main<5>([fn,this](){
        GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0))
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

}
}
}
