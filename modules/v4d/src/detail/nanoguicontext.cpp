// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/v4d.hpp"
#include "nanoguicontext.hpp"

namespace cv {
namespace v4d {
namespace detail {

NanoguiContext::NanoguiContext(FrameBufferContext& fbContext) :
        NanoVGContext(fbContext) {
}

void NanoguiContext::render() {
    run_sync_on_main<4>([this](){
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
            GLfloat cColor[4];
            glGetFloatv(GL_COLOR_CLEAR_VALUE, cColor);
            glClearColor(0,0,0,0);
            glClear(GL_COLOR_BUFFER_BIT);
            glClearColor(cColor[0], cColor[1], cColor[2], cColor[3]);
#endif
            screen().draw_widgets();
        }
        {
            if(!fbCtx().isShared()) {
#ifdef __EMSCRIPTEN__
                mainFbContext_.doWebGLCopy(fbCtx());
#else
                UMat tmp;
                fbCtx().copyTo(tmp);
                mainFbContext_.copyFrom(tmp);
#endif
            }
        }
    });
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
            NanoVGContext::render([this](const Size sz){
                CV_UNUSED(sz);
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
            });
        }
    }
    first_ = false;
    tick_.start();
}

void NanoguiContext::build(std::function<void(cv::v4d::FormHelper&)> fn) {
    run_sync_on_main<5>([fn,this](){
        FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
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
    return NanoVGContext::nvgFbContext_;
}
}
}
}
