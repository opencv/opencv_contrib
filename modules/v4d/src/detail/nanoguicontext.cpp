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
        mainFbContext_(fbContext), nguiFbContext_(fbContext.getV4D(), "NanoGUI", fbContext), context_(
                nullptr), copyBuffer_(mainFbContext_.size(), CV_8UC4) {
    run_sync_on_main<25>([&, this]() {
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
            screen_ = new nanogui::Screen(true, true, false);
            screen_->initialize(fbCtx().getGLFWWindow(), false);
            Size winSize = fbContext.getV4D().getWindowSize();
#ifndef __EMSCRIPTEN__
            screen_->set_size({int(winSize.width / fbContext.getV4D().pixelRatioX()), int(winSize.height / fbContext.getV4D().pixelRatioY())});
#endif
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

        if (tick_.getTimeMilli() > 100) {
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

        run_sync_on_main<4>([this, graphical](){
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
                GL_CHECK(glFinish());
                GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
                GL_CHECK(glViewport(0, 0, mainFbContext_.getWindowSize().width, mainFbContext_.getWindowSize().height));
                glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
#else
                FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
                GL_CHECK(glClearColor(0,0,0,0));
                GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
                GL_CHECK(glViewport(0, 0, mainFbContext_.getWindowSize().width, mainFbContext_.getWindowSize().height));
#endif
                if (graphical) {
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
#ifdef __EMSCRIPTEN__
                    fbCtx().makeCurrent();
#endif
                    fontSize(30.0f);
                    fontFace("mono");
                    fillColor(cv::Scalar(90, 90, 90, 255));
                    textAlign(NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
                    text(10, 20, txt.c_str(), nullptr);

                    nvgEndFrame(context_);
                    nvgRestore(context_);
                }
            }
            {
#ifdef __EMSCRIPTEN__
                FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
#else
                GL_CHECK(glFinish());
#endif
                screen().draw_widgets();
#ifndef __EMSCRIPTEN__
                GL_CHECK(glFlush());
#endif
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
