// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/v4d.hpp"
#include "opencv2/v4d/detail/framebuffercontext.hpp"
#include "detail/clvacontext.hpp"
#include "detail/nanovgcontext.hpp"
#include "detail/glcontext.hpp"
#include "detail/timetracker.hpp"
#include <sstream>

namespace cv {
namespace v4d {

cv::Ptr<V4D> V4D::make(int w, int h, const string& title, bool offscreen, bool debug, int samples) {
    V4D* v4d = new V4D(cv::Size(w,h), cv::Size(), title, offscreen, debug, samples);
    v4d->setVisible(!offscreen);
    return v4d->self();
}

cv::Ptr<V4D> V4D::make(const cv::Size& size, const cv::Size& fbsize, const string& title, bool offscreen, bool debug, int samples) {
    V4D* v4d = new V4D(size, fbsize, title, offscreen, debug, samples);
    v4d->setVisible(!offscreen);
    return v4d->self();
}

V4D::V4D(const cv::Size& size, const cv::Size& fbsize, const string& title, bool offscreen, bool debug, int samples) :
        initialSize_(size), title_(title), samples_(samples), debug_(debug), viewport_(0, 0, size.width, size.height), scaling_(true), pool_(2) {
#ifdef __EMSCRIPTEN__
    printf(""); //makes sure we have FS as a dependency
#endif
    self_ = cv::Ptr<V4D>(this);
    mainFbContext_ = new detail::FrameBufferContext(*this, fbsize.empty() ? size : fbsize, offscreen, title_, 3,
                2, samples_, debug_, nullptr, nullptr);
#ifndef __EMSCRIPTEN__
    CLExecScope_t scope(mainFbContext_->getCLExecContext());
#endif
    nvgContext_ = new detail::NanoVGContext(*mainFbContext_);
    clvaContext_ = new detail::CLVAContext(*mainFbContext_);
    imguiContext_ = new detail::ImGuiContextImpl(*mainFbContext_);
}

V4D::~V4D() {
    if (imguiContext_)
        delete imguiContext_;
    if (nvgContext_)
        delete nvgContext_;
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

ImGuiContextImpl& V4D::imguiCtx() {
    assert(imguiContext_ != nullptr);
    return *imguiContext_;
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

bool V4D::hasImguiCtx() {
    return imguiContext_ != nullptr;
}


bool V4D::hasGlCtx(uint32_t idx) {
    return glContexts_.find(idx) != glContexts_.end();
}

size_t V4D::numGlCtx() {
    return std::max(off_t(0), off_t(glContexts_.size()) - 1);
}

void V4D::gl(std::function<void()> fn) {
//    TimeTracker::getInstance()->execute("gl(" + detail::func_id(fn) + ")/" + std::to_string(-1), [this, fn](){
        glCtx(-1).render([=](const cv::Size& sz) {
            CV_UNUSED(sz);
            fn();
        });
//    });
}


void V4D::gl(std::function<void(const cv::Size&)> fn) {
//    TimeTracker::getInstance()->execute("gl(" + detail::func_id(fn) + ")/" + std::to_string(-1), [this, fn](){
        glCtx(-1).render(fn);
//    });
}

void V4D::gl(std::function<void()> fn, const uint32_t& idx) {
//    TimeTracker::getInstance()->execute("gl(" + detail::func_id(fn) + ")/" + std::to_string(idx), [this, fn, idx](){
        glCtx(idx).render([=](const cv::Size& sz) {
            CV_UNUSED(sz);
            fn();
        });
//    });
}


void V4D::gl(std::function<void(const cv::Size&)> fn, const uint32_t& idx) {
//    TimeTracker::getInstance()->execute("gl(" + detail::func_id(fn) + ")/" + std::to_string(idx), [this, fn, idx](){
        glCtx(idx).render(fn);
//    });
}


void V4D::fb(std::function<void(cv::UMat&)> fn) {
    TimeTracker::getInstance()->execute("fb(" + detail::func_id(fn) + ")", [this, fn](){
        fbCtx().execute(fn);
    });
}

void V4D::nvg(std::function<void()> fn) {
    TimeTracker::getInstance()->execute("nvg(" + detail::func_id(fn) + ")", [this, fn](){
        nvgCtx().render([fn](const cv::Size& sz) {
            CV_UNUSED(sz);
            fn();
        });
    });
}

void V4D::nvg(std::function<void(const cv::Size&)> fn) {
    TimeTracker::getInstance()->execute("nvg(" + detail::func_id(fn) + ")", [this, fn](){
        nvgCtx().render(fn);
    });
}

void V4D::imgui(std::function<void(ImGuiContext* ctx)> fn) {
    TimeTracker::getInstance()->execute("imgui(" + detail::func_id(fn) + ")", [this, fn](){
        imguiCtx().build([fn](ImGuiContext* ctx) {
            fn(ctx);
        });
    });
}

void V4D::copyTo(cv::UMat& m) {
    TimeTracker::getInstance()->execute("copyTo", [this, &m](){
        fbCtx().copyTo(m);
    });
}

void V4D::copyFrom(const cv::UMat& m) {
    TimeTracker::getInstance()->execute("copyTo", [this, &m](){
        fbCtx().copyFrom(m);
    });
}

#ifdef __EMSCRIPTEN__
bool first = true;
static void do_frame(void* void_fn_ptr) {
     if(first) {
         glfwSwapInterval(0);
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

void V4D::feed(cv::InputArray in) {
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

cv::_InputArray V4D::fetch() {
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
        });

        futureWriter_ = pool_.enqueue([](V4D* v, std::function<void(const UMat&)> func, cv::UMat& frame) {
            v->clvaCtx().write(func, frame);
        }, this, fn, currentWriterFrame_);
    });
}

bool V4D::isSinkReady() {
    return sink_.isReady();
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
}

void V4D::setScaling(bool s) {
    scaling_ = s;
}

bool V4D::isScaling() {
    return scaling_;
}

void V4D::swapContextBuffers() {
    run_sync_on_main<10>([this]() {
        for(size_t i = 0; i < numGlCtx(); ++i) {
            FrameBufferContext::GLScope glScope(glCtx(i).fbCtx(), GL_READ_FRAMEBUFFER);
            glCtx(i).fbCtx().blitFrameBufferToScreen(viewport(), glCtx(i).fbCtx().getWindowSize(), isScaling());
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
#ifndef __EMSCRIPTEN__
        glfwSwapBuffers(nvgCtx().fbCtx().getGLFWWindow());
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
        run_sync_on_main<6>([&, this]() {
            {
                FrameBufferContext::GLScope glScope(fbCtx(), GL_READ_FRAMEBUFFER);
                fbCtx().blitFrameBufferToScreen(viewport(), fbCtx().getWindowSize(), isScaling());
            }
            imguiCtx().render();
#ifndef __EMSCRIPTEN__
            if(debug_)
                swapContextBuffers();
#endif
            fbCtx().makeCurrent();
#ifndef __EMSCRIPTEN__
            glfwSwapBuffers(fbCtx().getGLFWWindow());
#else
            GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
            GL_CHECK(glViewport(0, 0, size().width, size().height));
            GL_CHECK(glFinish());
            emscripten_webgl_commit_frame();
#endif
            glfwPollEvents();
            result = !glfwWindowShouldClose(getGLFWWindow());
        });
    }
    if (frameCnt_ == (std::numeric_limits<uint64_t>().max() - 1))
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
    imguiCtx().makeCurrent();
}

cv::Ptr<V4D> V4D::self() {
       return self_;
}

}
}
