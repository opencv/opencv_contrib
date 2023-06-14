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
        mainFbContext_(fbContext), nguiFbContext_("NanoGUI", fbContext), context_(
                nullptr), copyBuffer_(mainFbContext_.size(), CV_8UC4) {
    run_sync_on_main<25>([this]() {
        {
            //Workaround for first frame glitch
            FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
            FrameBufferContext::FrameBufferScope fbScope(fbCtx(), copyBuffer_);
        }
        {
#ifndef __EMSCRIPTEN__
            mainFbContext_.makeCurrent();
            GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
#else
            FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
#endif
            screen_ = new nanogui::Screen();
            screen_->initialize(fbCtx().getGLFWWindow(), false);
            fbCtx().setWindowSize(fbCtx().size());
            context_ = screen_->nvg_context();
            form_ = new cv::v4d::FormHelper(screen_);
            if (!context_)
                throw std::runtime_error("Could not initialize NanoVG!");
       }
#ifdef __EMSCRIPTEN__
        mainFbContext_.initWebGLCopy(fbCtx());
#endif
    });
}

void NanoguiContext::render(bool print, bool graphical) {
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
            run_sync_on_main<4>([this](){
                string txt = "FPS: " + std::to_string(fps_);
#ifndef __EMSCRIPTEN__
                if(!fbCtx().isShared()) {
                    mainFbContext_.copyTo(copyBuffer_);
                    fbCtx().copyFrom(copyBuffer_);
                }
#endif
                {
#ifndef __EMSCRIPTEN__
                    mainFbContext_.makeCurrent();
                    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
                    GL_CHECK(glViewport(0, 0, mainFbContext_.getWindowSize().width, mainFbContext_.getWindowSize().height));
                    glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
#else
                    FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
                    GL_CHECK(glViewport(0, 0, mainFbContext_.getWindowSize().width, mainFbContext_.getWindowSize().height));
//                    GLfloat cColor[4];
//                    glGetFloatv(GL_COLOR_CLEAR_VALUE, cColor);
//                    glClearColor(0,0,1,1);
//                    glClearColor(0,0,0,0);
//                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
//                    glClearColor(cColor[0], cColor[1], cColor[2], cColor[3]);
                    glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
                }
                {
                    FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
#endif
                    float w = mainFbContext_.size().width;
                    float h = mainFbContext_.size().height;
                    float r = mainFbContext_.pixelRatioX();

                    nvgSave(context_);
                    nvgBeginFrame(context_, w, h, r);
                    cv::v4d::nvg::detail::NVG::initializeContext(context_);

                    using namespace cv::v4d::nvg;
                    beginPath();
                    roundedRect(5, 5, 15 * txt.size() + 5, 30, 5);
                    fillColor(cv::Scalar(255, 255, 255, 180));
                    fill();
                }
                {
#ifdef __EMSCRIPTEN__
                    FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
#endif
                    using namespace cv::v4d::nvg;
                    fontSize(30.0f);
                    fontFace("mono");
                    fillColor(cv::Scalar(90, 90, 90, 255));
                    textAlign(NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
                    text(10, 20, txt.c_str(), nullptr);

                    nvgEndFrame(context_);
                    nvgRestore(context_);
                    screen().draw_widgets();
                }

                if(!fbCtx().isShared()) {
#ifdef __EMSCRIPTEN__
                    mainFbContext_.doWebGLCopy(fbCtx());
#else
                    fbCtx().copyTo(copyBuffer_);
                    mainFbContext_.copyFrom(copyBuffer_);
#endif
                }
            });
        }
    }
    first_ = false;
    tick_.start();
}

void NanoguiContext::build(std::function<void(cv::v4d::FormHelper&)> fn) {
    run_sync_on_main<5>([fn,this](){
#ifndef __EMSCRIPTEN__
                    mainFbContext_.makeCurrent();
                    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
                    GL_CHECK(glViewport(0, 0, mainFbContext_.getWindowSize().width, mainFbContext_.getWindowSize().height));
#else
                    FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
                    GL_CHECK(glViewport(0, 0, mainFbContext_.getWindowSize().width, mainFbContext_.getWindowSize().height));

#endif
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
