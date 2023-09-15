// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/v4d.hpp"
#include "detail/framebuffercontext.hpp"
#include "detail/clvacontext.hpp"
#include "detail/nanovgcontext.hpp"
#include "detail/nanoguicontext.hpp"
#include "detail/glcontext.hpp"
#include "detail/timetracker.hpp"
#include "opencv2/v4d/dialog.hpp"
#include "opencv2/v4d/formhelper.hpp"
#include <sstream>

namespace cv {
namespace v4d {

cv::Ptr<V4D> V4D::make(const cv::Size& size, const cv::Size& fbsize, const string& title, bool offscreen, bool debug, bool compat, int samples) {
    V4D* v4d = new V4D(size, fbsize, title, offscreen, debug, compat, samples);
    v4d->setVisible(!offscreen);
    return v4d->self();
}

V4D::V4D(const cv::Size& size, const cv::Size& fbsize, const string& title, bool offscreen, bool debug, bool compat, int samples) :
        initialSize_(size), title_(title), compat_(
                compat), samples_(samples), debug_(debug), viewport_(0, 0, size.width, size.height), scaling_(true), pool_(2) {
#ifdef __EMSCRIPTEN__
    printf(""); //makes sure we have FS as a dependency
#endif
        mainFbContext_ = new detail::FrameBufferContext(*this, fbsize.empty() ? size : fbsize, offscreen, title_, 3,
                2, compat_, samples_, debug_, nullptr, nullptr);
#ifndef __EMSCRIPTEN__
        CLExecScope_t scope(mainFbContext_->getCLExecContext());
#endif
        nvgContext_ = new detail::NanoVGContext(*mainFbContext_);
        nguiContext_ = new detail::NanoguiContext(*mainFbContext_);
        clvaContext_ = new detail::CLVAContext(*mainFbContext_);
        self_ = cv::Ptr<V4D>(this);
}

V4D::~V4D() {
    //don't delete form_. it is autmatically cleaned up by the base class (nanogui::Screen)
    if (nvgContext_)
        delete nvgContext_;
    if (nguiContext_)
        delete nguiContext_;
    if (clvaContext_)
        delete clvaContext_;
    if (mainFbContext_)
        delete mainFbContext_;

    for(auto& it : glContexts_) {
        delete it.second;
    }
}

cv::ogl::Texture2D& V4D::texture() {
    return mainFbContext_->getTexture2D();
}

void V4D::setMouseButtonEventCallback(
        std::function<void(int button, int action, int modifiers)> fn) {
    mouseEventCb_ = fn;
}

void V4D::setKeyboardEventCallback(
        std::function<bool(int key, int scancode, int action, int modifiers)> fn) {
    keyEventCb_ = fn;
}

void V4D::mouse_button_event(int button, int action, int modifiers) {
    if (mouseEventCb_)
        return mouseEventCb_(button, action, modifiers);

    return nguiCtx().screen().mouse_button_callback_event(button, action, modifiers);
}

bool V4D::keyboard_event(int key, int scancode, int action, int modifiers) {
    if (keyEventCb_)
        return keyEventCb_(key, scancode, action, modifiers);

    return nguiCtx().screen().keyboard_event(key, scancode, action, modifiers);
}

cv::Point2f V4D::getMousePosition() {
    return mousePos_;
}

void V4D::setMousePosition(const cv::Point2f& pt) {
    mousePos_ = pt;
}

FrameBufferContext& V4D::fbCtx() {
    assert(mainFbContext_ != nullptr);
    return *mainFbContext_;
}

CLVAContext& V4D::clvaCtx() {
    assert(clvaContext_ != nullptr);
    return *clvaContext_;
}

NanoVGContext& V4D::nvgCtx() {
    assert(nvgContext_ != nullptr);
    return *nvgContext_;
}

NanoguiContext& V4D::nguiCtx() {
    assert(nguiContext_ != nullptr);
    return *nguiContext_;
}

GLContext& V4D::glCtx(int32_t idx) {
    auto it = glContexts_.find(idx);
    if(it != glContexts_.end())
        return *(*it).second;
    else {
        GLContext* ctx = new GLContext(*mainFbContext_);
        glContexts_.insert({idx, ctx});
        return *ctx;
    }
}

bool V4D::hasFbCtx() {
    return mainFbContext_ != nullptr;
}

bool V4D::hasClvaCtx() {
    return clvaContext_ != nullptr;
}

bool V4D::hasNvgCtx() {
    return nvgContext_ != nullptr;
}

bool V4D::hasNguiCtx() {
    return nguiContext_ != nullptr;
}

bool V4D::hasGlCtx(uint32_t idx) {
    return glContexts_.find(idx) != glContexts_.end();
}

size_t V4D::numGlCtx() {
    return std::max(off_t(0), off_t(glContexts_.size()) - 1);
}

void V4D::gl(std::function<void()> fn) {
    TimeTracker::getInstance()->execute("gl(" + detail::func_id(fn) + ")/" + std::to_string(-1), [=](){
        glCtx(-1).render([=](const cv::Size& sz) {
            CV_UNUSED(sz);
            fn();
        });
    });
}


void V4D::gl(std::function<void(const cv::Size&)> fn) {
    TimeTracker::getInstance()->execute("gl(" + detail::func_id(fn) + ")/" + std::to_string(-1), [=](){
        glCtx(-1).render(fn);
    });
}

void V4D::gl(std::function<void()> fn, const uint32_t& idx) {
    TimeTracker::getInstance()->execute("gl(" + detail::func_id(fn) + ")/" + std::to_string(idx), [=](){
        glCtx(idx).render([=](const cv::Size& sz) {
            CV_UNUSED(sz);
            fn();
        });
    });
}


void V4D::gl(std::function<void(const cv::Size&)> fn, const uint32_t& idx) {
    TimeTracker::getInstance()->execute("gl(" + detail::func_id(fn) + ")/" + std::to_string(idx), [=](){
        glCtx(idx).render(fn);
    });
}


void V4D::fb(std::function<void(cv::UMat&)> fn) {
    TimeTracker::getInstance()->execute("fb(" + detail::func_id(fn) + ")", [&](){
        fbCtx().execute(fn);
    });
}

void V4D::nvg(std::function<void()> fn) {
    TimeTracker::getInstance()->execute("nvg(" + detail::func_id(fn) + ")", [&](){
        nvgCtx().render([fn](const cv::Size& sz) {
            CV_UNUSED(sz);
            fn();
        });
    });
}

void V4D::nvg(std::function<void(const cv::Size&)> fn) {
    TimeTracker::getInstance()->execute("nvg(" + detail::func_id(fn) + ")", [&](){
        nvgCtx().render(fn);
    });
}

void V4D::nanogui(std::function<void(cv::v4d::FormHelper& form)> fn) {
    nguiCtx().build(fn);
}

void V4D::copyTo(cv::UMat& m) {
    TimeTracker::getInstance()->execute("copyTo", [&](){
        fbCtx().copyTo(m);
    });
}

void V4D::copyFrom(const cv::UMat& m) {
    TimeTracker::getInstance()->execute("copyTo", [&](){
        fbCtx().copyFrom(m);
    });
}

#ifdef __EMSCRIPTEN__
bool first = true;
static void do_frame(void* void_fn_ptr) {
     if(first) {
         glfwSwapInterval(1);
         first = false;
     }
     auto* fn_ptr = reinterpret_cast<std::function<bool()>*>(void_fn_ptr);
     if (fn_ptr) {
         auto& fn = *fn_ptr;
         //FIXME cancel main loop
         fn();
     }
 }
#endif

void V4D::run(std::function<bool(cv::Ptr<V4D>)> fn) {
#ifndef __EMSCRIPTEN__
    bool success = true;
    while (keepRunning() && success) {
        CLExecScope_t scope(fbCtx().getCLExecContext());
        success = fn(self());
    }
    pool_.finish();
#else
    std::function<bool()> fnFrame([=,this](){
        return fn(self());
    });
    emscripten_set_main_loop_arg(do_frame, &fnFrame, -1, true);
#endif
}

void V4D::setSource(const Source& src) {
    if (!clvaCtx().hasContext()) {
        if(isIntelVaSupported()) {
            clvaCtx().copyContext();
        }
    }

    source_ = src;
}

void V4D::feed(const cv::UMat& in) {
    TimeTracker::getInstance()->execute("feed", [&](){
        cv::UMat frame;
        clvaCtx().capture([&](cv::UMat& videoFrame) {
            in.copyTo(videoFrame);
        }, frame);

        fb([frame](cv::UMat& frameBuffer){
            frame.copyTo(frameBuffer);
        });
    });
}

cv::UMat V4D::fetch() {
    cv::UMat frame;
    TimeTracker::getInstance()->execute("copyTo", [&](){
        fb([frame](cv::UMat& framebuffer){
            framebuffer.copyTo(frame);
        });
    });
    return frame;
}

bool V4D::capture() {
    return this->capture([&](cv::UMat& videoFrame) {
        if (source_.isReady())
            source_().second.copyTo(videoFrame);
    });
}

bool V4D::capture(std::function<void(cv::UMat&)> fn) {
    bool res = true;
    TimeTracker::getInstance()->execute("capture", [&, this](){
        if (!source_.isReady() || !source_.isOpen()) {
#ifndef __EMSCRIPTEN__
            res = false;
#endif
            return;
        }
        if (futureReader_.valid()) {
            if (!futureReader_.get()) {
#ifndef __EMSCRIPTEN__
                res = false;
#endif
                return;
            }
        }

        if(nextReaderFrame_.empty()) {
            if (!clvaCtx().capture(fn, nextReaderFrame_)) {
#ifndef __EMSCRIPTEN__
                res = false;
#endif
                return;
            }
        }
        nextReaderFrame_.copyTo(currentReaderFrame_);
        futureReader_ = pool_.enqueue(
            [](V4D* v, std::function<void(UMat&)> func, cv::UMat& frame) {
                return v->clvaCtx().capture(func, frame);
            }, this, fn, nextReaderFrame_);

        fb([this](cv::UMat& frameBuffer){
            currentReaderFrame_.copyTo(frameBuffer);
        });
    });
    return res;
}

bool V4D::isSourceReady() {
    return source_.isReady();
}

void V4D::setSink(const Sink& sink) {
    if (!clvaCtx().hasContext()) {
        if(isIntelVaSupported()) {
            clvaCtx().copyContext();
        }
    }
    sink_ = sink;
}

void V4D::write() {
    this->write([&](const cv::UMat& videoFrame) {
        if (sink_.isReady())
            sink_(videoFrame);
    });
}

void V4D::write(std::function<void(const cv::UMat&)> fn) {
    TimeTracker::getInstance()->execute("write", [&, this](){
        if (!sink_.isReady() || !sink_.isOpen())
            return;

        if (futureWriter_.valid())
            futureWriter_.get();

        fb([this](cv::UMat& frameBuffer){
            frameBuffer.copyTo(currentWriterFrame_);
            //            currentWriterFrame_ = frameBuffer;
        });

        futureWriter_ = pool_.enqueue([](V4D* v, std::function<void(const UMat&)> func, cv::UMat& frame) {
            v->clvaCtx().write(func, frame);
        }, this, fn, currentWriterFrame_);
    });
}

bool V4D::isSinkReady() {
    return sink_.isReady();
}

void V4D::showGui(bool s) {
    auto children = nguiCtx().screen().children();
    for (auto* child : children) {
        child->set_visible(s);
    }
}

cv::Vec2f V4D::position() {
    return fbCtx().position();
}

cv::Rect& V4D::viewport() {
    return viewport_;
}

float V4D::pixelRatioX() {
    return fbCtx().pixelRatioX();
}

float V4D::pixelRatioY() {
    return fbCtx().pixelRatioY();
}

cv::Size V4D::framebufferSize() {
    return fbCtx().size();
}

cv::Size V4D::initialSize() {
    return initialSize_;
}

cv::Size V4D::size() {
    return fbCtx().getWindowSize();
}

void V4D::setWindowSize(const cv::Size& sz) {
    fbCtx().setWindowSize(sz);
}

bool V4D::getShowFPS() {
    return showFPS_;
}

bool V4D::getPrintFPS() {
    return printFPS_;
}

bool V4D::getShowTracking() {
    return showTracking_;
}

void V4D::setShowFPS(bool s) {
    showFPS_ = s;
}

void V4D::setPrintFPS(bool p) {
    printFPS_ = p;
}

void V4D::setShowTracking(bool st) {
    showTracking_ = st;
}

bool V4D::isFullscreen() {
    return fbCtx().isFullscreen();
}

void V4D::setFullscreen(bool f) {
    fbCtx().setFullscreen(f);
}

bool V4D::isResizable() {
    return fbCtx().isResizable();
}

void V4D::setResizable(bool r) {
    fbCtx().setResizable(r);
}

bool V4D::isVisible() {
    return fbCtx().isVisible();
}

void V4D::setVisible(bool v) {
    fbCtx().setVisible(v);
    nguiCtx().screen().perform_layout();
}

void V4D::setScaling(bool s) {
    scaling_ = s;
}

bool V4D::isScaling() {
    return scaling_;
}

void V4D::setDefaultKeyboardEventCallback() {
    setKeyboardEventCallback([&](int key, int scancode, int action, int modifiers) {
        CV_UNUSED(scancode);
        CV_UNUSED(modifiers);
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            setVisible(!isVisible());
            return true;
        } else if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
            auto children = nguiCtx().screen().children();
            for (auto* child : children) {
                child->set_visible(!child->visible());
            }

            return true;
        }
        return false;
    });
}

void V4D::swapContextBuffers() {
    run_sync_on_main<10>([this]() {
        for(size_t i = 0; i < numGlCtx(); ++i) {
            FrameBufferContext::GLScope glScope(glCtx(i).fbCtx(), GL_READ_FRAMEBUFFER);
            glCtx(i).fbCtx().blitFrameBufferToScreen(viewport(), glCtx(i).fbCtx().getWindowSize(), isScaling());
//            GL_CHECK(glFinish());
#ifndef __EMSCRIPTEN__
            glfwSwapBuffers(glCtx(i).fbCtx().getGLFWWindow());
#else
            emscripten_webgl_commit_frame();
#endif
        }
    });

    run_sync_on_main<11>([this]() {
        FrameBufferContext::GLScope glScope(nvgCtx().fbCtx(), GL_READ_FRAMEBUFFER);
        nvgCtx().fbCtx().blitFrameBufferToScreen(viewport(), nvgCtx().fbCtx().getWindowSize(), isScaling());
//        GL_CHECK(glFinish());
#ifndef __EMSCRIPTEN__
        glfwSwapBuffers(nvgCtx().fbCtx().getGLFWWindow());
#else
        emscripten_webgl_commit_frame();
#endif
    });

    run_sync_on_main<12>([this]() {
        FrameBufferContext::GLScope glScope(nguiCtx().fbCtx(), GL_READ_FRAMEBUFFER);
        nguiCtx().fbCtx().blitFrameBufferToScreen(viewport(), nguiCtx().fbCtx().getWindowSize(), isScaling());
//        GL_CHECK(glFinish());
#ifndef __EMSCRIPTEN__
        glfwSwapBuffers(nguiCtx().fbCtx().getGLFWWindow());
#else
        emscripten_webgl_commit_frame();
#endif
    });
}

bool V4D::display() {
    bool result = true;
#ifndef __EMSCRIPTEN__
    if (isVisible()) {
#else
    if (true) {
#endif
        if(debug)
            swapContextBuffers();


#ifdef __EMSCRIPTEN__
        nguiCtx().render(printFPS_, showFPS_, showTracking_);
#endif
        run_sync_on_main<6>([&, this](){
            {
               FrameBufferContext::GLScope glScope(fbCtx(), GL_READ_FRAMEBUFFER);
               fbCtx().blitFrameBufferToScreen(viewport(), fbCtx().getWindowSize(), isScaling());
            }
#ifndef __EMSCRIPTEN__
            nguiCtx().render(printFPS_, showFPS_, showTracking_);
#endif

//            {
//                FrameBufferContext::GLScope glScope(nvgCtx().fbCtx(), GL_FRAMEBUFFER);
//                GL_CHECK(glFinish());
//            }
//            {
//                FrameBufferContext::GLScope glScope(nguiCtx().fbCtx(), GL_FRAMEBUFFER);
//                GL_CHECK(glFinish());
//            }
//            for(size_t i = 0; i < numGlCtx(); ++i) {
//                {
//                    FrameBufferContext::GLScope glScope(glCtx(i).fbCtx(), GL_FRAMEBUFFER);
//                    GL_CHECK(glFinish());
//                }
//            }
//            {
//                FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
//                GL_CHECK(glFinish());
//            }

            fbCtx().makeCurrent();
#ifndef __EMSCRIPTEN__
            glfwSwapBuffers(fbCtx().getGLFWWindow());
#else
            emscripten_webgl_commit_frame();
#endif
            glfwPollEvents();
            result = !glfwWindowShouldClose(getGLFWWindow());
        });
    }
    if(frameCnt_ == (std::numeric_limits<uint64_t>().max() - 1))
        frameCnt_ = 0;
    else
        ++frameCnt_;

    return result;
}

uint64_t V4D::frameCount() {
    return frameCnt_;
}

bool V4D::isClosed() {
    return fbCtx().isClosed();
}

void V4D::close() {
    fbCtx().close();
}

GLFWwindow* V4D::getGLFWWindow() {
    return fbCtx().getGLFWWindow();
}

void V4D::printSystemInfo() {
    run_sync_on_main<8>([this](){
        fbCtx().makeCurrent();
        cerr << "OpenGL: " << getGlInfo() << endl;
        cerr << "OpenCL Platforms: " << getClInfo() << endl;
    });
}

void V4D::makeCurrent() {
    fbCtx().makeCurrent();
}

cv::Ptr<V4D> V4D::self() {
       return self_;
}

}
}
