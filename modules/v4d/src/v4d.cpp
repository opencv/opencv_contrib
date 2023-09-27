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
        initialSize_(size), debug_(debug), viewport_(0, 0, size.width, size.height), stretching_(true), pool_(2) {
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
        nvgContext_ = new detail::NanoVGContext(*mainFbContext_);
    clvaContext_ = new detail::CLVAContext(*mainFbContext_);
    if(flags & IMGUI)
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

std::string V4D::title() {
    return fbCtx().title_;
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
    TimeTracker::getInstance()->execute("gl(" + detail::func_id(fn) + ")/" + std::to_string(-1), [this, fn](){
        glCtx(-1).render([=](const cv::Size& sz) {
            CV_UNUSED(sz);
            fn();
        });
    });
}


void V4D::gl(std::function<void(const cv::Size&)> fn) {
    TimeTracker::getInstance()->execute("gl(" + detail::func_id(fn) + ")/" + std::to_string(-1), [this, fn](){
        glCtx(-1).render(fn);
    });
}

void V4D::gl(std::function<void()> fn, const uint32_t& idx) {
    TimeTracker::getInstance()->execute("gl(" + detail::func_id(fn) + ")/" + std::to_string(idx), [this, fn, idx](){
        glCtx(idx).render([=](const cv::Size& sz) {
            CV_UNUSED(sz);
            fn();
        });
    });
}


void V4D::gl(std::function<void(const cv::Size&)> fn, const uint32_t& idx) {
    TimeTracker::getInstance()->execute("gl(" + detail::func_id(fn) + ")/" + std::to_string(idx), [this, fn, idx](){
        glCtx(idx).render(fn);
    });
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

void V4D::run(std::function<bool(cv::Ptr<V4D>)> fn, size_t workers) {
#ifndef __EMSCRIPTEN__
    std::vector<std::thread*> threads;
    for (size_t i = 0; i < workers; ++i) {
        threads.push_back(
                new std::thread(
                        [this, fn, i] {
                            cv::Ptr<cv::v4d::V4D> worker = V4D::make(
                                    this->initialSize().width,
                                    this->initialSize().height,
                                    this->title() + "-worker-" + std::to_string(i),
                                    NANOVG,
                                    !this->debug_,
                                    this->debug_,
                                    0);
                            if (this->hasSource()) {
                                Source& src = this->getSource();
                                src.setThreadSafe(true);
                                worker->setSource(src);
                            }
                            if (this->hasSink()) {
                                Sink& sink = this->getSink();
                                sink.setThreadSafe(true);
                                worker->setSink(sink);
                            }
                            worker->run(fn, 0);
                        }
                )
        );
    }

    this->makeCurrent();
    bool success = true;
    while (keepRunning() && success) {
        CLExecScope_t scope(fbCtx().getCLExecContext());
        success = fn(self());
    }
    pool_.finish();

    for(auto& t : threads)
        t->join();
#else
    std::function<bool()> fnFrame([=,this](){
        return fn(self());
    });
    emscripten_set_main_loop_arg(do_frame, &fnFrame, -1, true);
#endif
}

void V4D::setSource(Source& src) {
    source_ = &src;
}

Source& V4D::getSource() {
    CV_Assert(source_ != nullptr);
    return *source_;
}

bool V4D::hasSource() {
    return source_ != nullptr;
}

void V4D::feed(cv::InputArray in) {
    CLExecScope_t scope(fbCtx().getCLExecContext());
    TimeTracker::getInstance()->execute("feed", [this, &in](){
        cv::Mat source;
        in.getUMat().getMat(ACCESS_READ).copyTo(source);
        cv::UMat frame;
        clvaCtx().capture([&](cv::UMat& videoFrame) {
            source.copyTo(videoFrame);
        }, frame);

        fb([&frame](cv::UMat& framebuffer){
            frame.copyTo(framebuffer);
        });
    });
}

cv::_InputArray V4D::fetch() {
    cv::UMat frame;
    TimeTracker::getInstance()->execute("copyTo", [this, &frame](){
        fb([frame](cv::UMat& framebuffer){
            framebuffer.copyTo(frame);
        });
    });
    return frame;
}

bool V4D::capture() {
    CLExecScope_t scope(fbCtx().getCLExecContext());
    if (source_) {
        return this->capture([this](cv::UMat& videoFrame) {
            if (source_->isReady()) {
                auto p = source_->operator()();
                currentSeqNr_ = p.first;
                if (source_->isThreadSafe()) {
                    p.second.getMat(cv::ACCESS_READ).copyTo(videoFrame);
                } else {
                    p.second.copyTo(videoFrame);
                }
                return true;
            }
            return false;
        });
#ifndef __EMSCRIPTEN__
        return false;
#else
        return true;
#endif
    }
    return false;
}

bool V4D::capture(std::function<void(cv::UMat&)> fn) {
    bool res = true;
    TimeTracker::getInstance()->execute("capture", [this, fn, &res](){
        CV_UNUSED(res);
        if (!source_ || !source_->isReady() || !source_->isOpen()) {
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
#ifndef __EMSCRIPTEN__
            return v->clvaCtx().capture(func, frame);
#else
            v->clvaCtx().capture(func, frame);
            return true;
#endif
            }, this, fn, nextReaderFrame_);

        fb([this](cv::UMat& frameBuffer){
            currentReaderFrame_.copyTo(frameBuffer);
        });
    });
    return res;
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

void V4D::write() {
    this->write([&](const cv::UMat& videoFrame) {
        if (sink_ && sink_->isReady())
            sink_->operator()(currentSeqNr_, videoFrame);
    });
}

void V4D::write(std::function<void(const cv::UMat&)> fn) {
    TimeTracker::getInstance()->execute("write", [this, fn](){
        if (!sink_ || !sink_->isReady() || !sink_->isOpen())
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
    return sink_ && sink_->isReady();
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

void V4D::setSize(const cv::Size& sz) {
    fbCtx().setWindowSize(sz);
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
    run_sync_on_main<10>([this]() {
        for(size_t i = 0; i < numGlCtx(); ++i) {
            FrameBufferContext::GLScope glScope(glCtx(i).fbCtx(), GL_READ_FRAMEBUFFER);
            glCtx(i).fbCtx().blitFrameBufferToFrameBuffer(viewport(), glCtx(i).fbCtx().getWindowSize(), 0, isStretching());
#ifndef __EMSCRIPTEN__
            glfwSwapBuffers(glCtx(i).fbCtx().getGLFWWindow());
#else
            emscripten_webgl_commit_frame();
#endif
        }
    });

    run_sync_on_main<11>([this]() {
        FrameBufferContext::GLScope glScope(nvgCtx().fbCtx(), GL_READ_FRAMEBUFFER);
        nvgCtx().fbCtx().blitFrameBufferToFrameBuffer(viewport(), nvgCtx().fbCtx().getWindowSize(), 0, isStretching());
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
                fbCtx().blitFrameBufferToFrameBuffer(viewport(), fbCtx().getWindowSize(), 0, isStretching());
            }
            if(hasImguiCtx())
                imguiCtx().render(getShowFPS());
//#ifndef __EMSCRIPTEN__
//            if(debug_)
//                swapContextBuffers();
//#endif
            fbCtx().makeCurrent();
#ifndef __EMSCRIPTEN__
            glfwSwapBuffers(fbCtx().getGLFWWindow());
#else
            emscripten_webgl_commit_frame();
#endif
            glfwPollEvents();
            result = !glfwWindowShouldClose(getGLFWWindow());

            {
                FrameBufferContext::GLScope glScope(fbCtx(), GL_DRAW_FRAMEBUFFER);
                GL_CHECK(glViewport(0, 0, fbCtx().size().width, fbCtx().size().height));
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
