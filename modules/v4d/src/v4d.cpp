// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/v4d.hpp"
#include "opencv2/v4d/detail/framebuffercontext.hpp"
#include <sstream>
#include <algorithm>
#include <opencv2/core.hpp>
#include <vector>

namespace cv {
namespace v4d {
const std::thread::id V4D::default_thread_id_;
std::thread::id V4D::main_thread_id_;
bool V4D::first_run_;

cv::Ptr<V4D> V4D::make(int w, int h, const string& title, AllocateFlags flags, bool offscreen, bool debug, int samples) {
    V4D* v4d = new V4D(cv::Size(w,h), cv::Size(), title, flags, offscreen, debug, samples);
    v4d->setVisible(!offscreen);
    return v4d->self();
}

cv::Ptr<V4D> V4D::make(const cv::Size& size, const cv::Size& fbsize, const string& title, AllocateFlags flags, bool offscreen, bool debug, int samples) {
    V4D* v4d = new V4D(size, fbsize, title, flags, offscreen, debug, samples);
    v4d->setVisible(!offscreen);
    return v4d->self();
}

V4D::V4D(const cv::Size& size, const cv::Size& fbsize, const string& title, AllocateFlags flags, bool offscreen, bool debug, int samples) :
        initialSize_(size), debug_(debug), viewport_(0, 0, size.width, size.height), stretching_(true) {
#ifdef __EMSCRIPTEN__
    printf(""); //makes sure we have FS as a dependency
#endif
    self_ = cv::Ptr<V4D>(this);
    mainFbContext_ = new detail::FrameBufferContext(*this, fbsize.empty() ? size : fbsize, offscreen, title, 3,
                2, samples, debug, nullptr, nullptr);
#ifndef __EMSCRIPTEN__
    CLExecScope_t scope(mainFbContext_->getCLExecContext());
#endif
    if(flags & NANOVG)
        nvgContext_ = new detail::NanoVGContext(mainFbContext_);
    sourceContext_ = new detail::SourceContext(mainFbContext_);
    sinkContext_ = new detail::SinkContext(mainFbContext_);
    singleContext_ = new detail::SingleContext();
    parallelContext_ = new detail::ParallelContext();
    if(flags & IMGUI)
        imguiContext_ = new detail::ImGuiContextImpl(mainFbContext_);
}

V4D::~V4D() {

}

cv::Ptr<cv::UMat> V4D::get(const string& name) {
    return umat_pool_[name];
}


cv::Ptr<cv::UMat> V4D::get(const string& name, cv::Size sz, int type) {
    cv::Ptr<cv::UMat> u = umat_pool_[name];
    u->create(sz, type);
    return u;
}

size_t V4D::workers() {
        return numWorkers_;
}

bool V4D::isMain() const {
        return main_thread_id_ == default_thread_id_ || main_thread_id_ == std::this_thread::get_id();
}

cv::ogl::Texture2D& V4D::texture() {
    return mainFbContext_->getTexture2D();
}

std::string V4D::title() {
    return fbCtx()->title_;
}


cv::Point2f V4D::getMousePosition() {
    return mousePos_;
}

void V4D::setMousePosition(const cv::Point2f& pt) {
    mousePos_ = pt;
}

cv::Ptr<FrameBufferContext> V4D::fbCtx() {
    assert(mainFbContext_ != nullptr);
    return mainFbContext_;
}

cv::Ptr<SourceContext> V4D::sourceCtx() {
    assert(sourceContext_ != nullptr);
    return sourceContext_;
}

cv::Ptr<SinkContext> V4D::sinkCtx() {
    assert(sinkContext_ != nullptr);
    return sinkContext_;
}

cv::Ptr<NanoVGContext> V4D::nvgCtx() {
    assert(nvgContext_ != nullptr);
    return nvgContext_;
}

cv::Ptr<SingleContext> V4D::singleCtx() {
    assert(singleContext_ != nullptr);
    return singleContext_;
}

cv::Ptr<ParallelContext> V4D::parallelCtx() {
    assert(parallelContext_ != nullptr);
    return parallelContext_;
}

cv::Ptr<ImGuiContextImpl> V4D::imguiCtx() {
    assert(imguiContext_ != nullptr);
    return imguiContext_;
}

cv::Ptr<GLContext> V4D::glCtx(int32_t idx) {
    auto it = glContexts_.find(idx);
    if(it != glContexts_.end())
        return (*it).second;
    else {
        cv::Ptr<GLContext> ctx = new GLContext(mainFbContext_);
        glContexts_.insert({idx, ctx});
        return ctx;
    }
}

bool V4D::hasFbCtx() {
    return mainFbContext_ != nullptr;
}

bool V4D::hasSourceCtx() {
    return sourceContext_ != nullptr;
}

bool V4D::hasSinkCtx() {
    return sinkContext_ != nullptr;
}

bool V4D::hasNvgCtx() {
    return nvgContext_ != nullptr;
}

bool V4D::hasSingleCtx() {
    return singleContext_ != nullptr;
}

bool V4D::hasParallelCtx() {
    return parallelContext_ != nullptr;
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


void V4D::imgui(std::function<void(ImGuiContext* ctx)> fn) {
    TimeTracker::getInstance()->execute("imgui", [this, fn](){
        imguiCtx()->build([fn](ImGuiContext* ctx) {
            fn(ctx);
        });
    });
}

void V4D::copyTo(cv::UMat& m) {
    TimeTracker::getInstance()->execute("copyTo", [this, &m](){
        fbCtx()->copyTo(m);
    });
}

void V4D::copyFrom(const cv::UMat& m) {
    TimeTracker::getInstance()->execute("copyTo", [this, &m](){
        fbCtx()->copyFrom(m);
    });
}

void V4D::setSource(cv::Ptr<Source> src) {
    source_ = src;
}
cv::Ptr<Source> V4D::getSource() {
    return source_;
}

bool V4D::hasSource() {
    return source_ != nullptr;
}

void V4D::feed(cv::InputArray in) {
	CV_Assert(false);
//#ifndef __EMSCRIPTEN__
//    CLExecScope_t scope(fbCtx()->getCLExecContext());
//#endif
//    TimeTracker::getInstance()->execute("feed", [this, &in](){
//        cv::UMat frame;
//        captureCtx().capture([&](cv::UMat& videoFrame) {
//            in.copyTo(videoFrame);
//        }, frame);
//
//        fb([](cv::UMat& fb, const cv::UMat& f) {
//            f.copyTo(fb);
//        }, frame);
//    });
}

cv::_InputArray V4D::fetch() {
	CV_Assert(false);
   cv::UMat frame;
//    TimeTracker::getInstance()->execute("copyTo", [this, &frame](){
//        fb([](const cv::UMat& fb, cv::UMat& f) {
//            fb.copyTo(f);
//        }, frame);
//    });
    return frame;
}


bool V4D::isSourceReady() {
    return source_ && source_->isReady();
}

void V4D::setSink(Sink& sink) {
    sink_ = &sink;
}

Sink& V4D::getSink() {
    CV_Assert(sink_ != nullptr);
    return *sink_;
}

bool V4D::hasSink() {
    return sink_ != nullptr;
}

bool V4D::isSinkReady() {
    return sink_ && sink_->isReady();
}

cv::Vec2f V4D::position() {
    return fbCtx()->position();
}

cv::Rect& V4D::viewport() {
    return viewport_;
}

float V4D::pixelRatioX() {
    return fbCtx()->pixelRatioX();
}

float V4D::pixelRatioY() {
    return fbCtx()->pixelRatioY();
}

cv::Size V4D::fbSize() {
    return fbCtx()->size();
}

cv::Size V4D::initialSize() {
    return initialSize_;
}

cv::Size V4D::size() {
    return fbCtx()->getWindowSize();
}

void V4D::setSize(const cv::Size& sz) {
    fbCtx()->setWindowSize(sz);
}

void V4D::setShowFPS(bool s) {
    showFPS_ = s;
}

bool V4D::getShowFPS() {
    return showFPS_;
}

void V4D::setPrintFPS(bool p) {
    printFPS_ = p;
}

bool V4D::getPrintFPS() {
    return printFPS_;
}

void V4D::setShowTracking(bool st) {
    showTracking_ = st;
}

bool V4D::getShowTracking() {
    return showTracking_;
}

bool V4D::isFullscreen() {
    return fbCtx()->isFullscreen();
}

void V4D::setFullscreen(bool f) {
    fbCtx()->setFullscreen(f);
}

bool V4D::isResizable() {
    return fbCtx()->isResizable();
}

void V4D::setResizable(bool r) {
    fbCtx()->setResizable(r);
}

bool V4D::isVisible() {
    return fbCtx()->isVisible();
}

void V4D::setVisible(bool v) {
    fbCtx()->setVisible(v);
}

void V4D::setStretching(bool s) {
    stretching_ = s;
}

bool V4D::isStretching() {
    return stretching_;
}

void V4D::setFocused(bool f) {
    focused_ = f;
}

bool V4D::isFocused() {
    return focused_;
}

void V4D::swapContextBuffers() {
    for(size_t i = 0; i < numGlCtx(); ++i) {
        FrameBufferContext::GLScope glScope(glCtx(i)->fbCtx(), GL_READ_FRAMEBUFFER);
        glCtx(i)->fbCtx()->blitFrameBufferToFrameBuffer(viewport(), glCtx(i)->fbCtx()->getWindowSize(), 0, isStretching());
#ifndef __EMSCRIPTEN__
        glfwSwapBuffers(glCtx(i)->fbCtx()->getGLFWWindow());
#else
        emscripten_webgl_commit_frame();
#endif
    }

    FrameBufferContext::GLScope glScope(nvgCtx()->fbCtx(), GL_READ_FRAMEBUFFER);
    nvgCtx()->fbCtx()->blitFrameBufferToFrameBuffer(viewport(), nvgCtx()->fbCtx()->getWindowSize(), 0, isStretching());
#ifndef __EMSCRIPTEN__
    glfwSwapBuffers(nvgCtx()->fbCtx()->getGLFWWindow());
#else
    emscripten_webgl_commit_frame();
#endif
}

bool V4D::display() {
    bool result = true;
    int fcnt = ++Global::frame_cnt();

#ifndef __EMSCRIPTEN__
    if (isVisible()) {
#else
    if (true) {
#endif
        run_sync_on_main<6>([&, this]() {
            if(this->isMain()) {
            	auto start = Global::start_time();
            	auto now = get_epoch_nanos();

            	double diff_seconds = (now - start) / 1000000000.0;
            	if(diff_seconds > 2.0) {
            		Global::start_time() = now;
            		Global::frame_cnt() = 1;
            	}

            	Global::fps() = (fcnt / diff_seconds);
            	cerr << "\rFPS:" << Global::fps() << endl;
            }
            {
                FrameBufferContext::GLScope glScope(fbCtx(), GL_READ_FRAMEBUFFER);
                fbCtx()->blitFrameBufferToFrameBuffer(viewport(), fbCtx()->getWindowSize(), 0, isStretching());
            }
            if(hasImguiCtx())
                imguiCtx()->render(getShowFPS());
#ifndef __EMSCRIPTEN__
            if(debug_)
                swapContextBuffers();
#endif
#ifndef __EMSCRIPTEN__
            glfwSwapBuffers(fbCtx()->getGLFWWindow());
#else
            emscripten_webgl_commit_frame();
#endif
            glfwPollEvents();
            result = !glfwWindowShouldClose(getGLFWWindow());

            {
                FrameBufferContext::GLScope glScope(fbCtx(), GL_DRAW_FRAMEBUFFER);
                GL_CHECK(glViewport(0, 0, fbCtx()->size().width, fbCtx()->size().height));
                GL_CHECK(glClearColor(0,0,0,255));
                GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));
            }
#ifndef __EMSCRIPTEN__
            {
                GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
                GL_CHECK(glViewport(0, 0, size().width, size().height));
                GL_CHECK(glClearColor(0,0,0,255));
                GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));
            }
#endif
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
    return fbCtx()->isClosed();
}

void V4D::close() {
    fbCtx()->close();
}

GLFWwindow* V4D::getGLFWWindow() {
    return fbCtx()->getGLFWWindow();
}

void V4D::printSystemInfo() {
    run_sync_on_main<8>([this](){
        cerr << "OpenGL: " << getGlInfo() << endl;
        cerr << "OpenCL Platforms: " << getClInfo() << endl;
    });
}

//void V4D::makeCurrent() {
//    fbCtx()->makeCurrent();
//}

cv::Ptr<V4D> V4D::self() {
       return self_;
}


}
}
