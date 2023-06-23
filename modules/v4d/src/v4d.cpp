// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/v4d.hpp"
#include "detail/clvacontext.hpp"
#include "detail/framebuffercontext.hpp"
#include "detail/glcontext.hpp"
#include "detail/nanovgcontext.hpp"
#include "detail/nanoguicontext.hpp"
#include "opencv2/v4d/dialog.hpp"
#include "opencv2/v4d/formhelper.hpp"
#include <sstream>

namespace cv {
namespace v4d {
namespace detail {
void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error: (%d) %s\n", error, description);
}
}

cv::Scalar colorConvert(const cv::Scalar& src, cv::ColorConversionCodes code) {
    cv::Mat tmpIn(1, 1, CV_8UC3);
    cv::Mat tmpOut(1, 1, CV_8UC3);

    tmpIn.at<cv::Vec3b>(0, 0) = cv::Vec3b(src[0], src[1], src[2]);
    cvtColor(tmpIn, tmpOut, code);
    const cv::Vec3b& vdst = tmpOut.at<cv::Vec3b>(0, 0);
    cv::Scalar dst(vdst[0], vdst[1], vdst[2], src[3]);
    return dst;
}

cv::Ptr<V4D> V4D::make(const cv::Size& size, const cv::Size& fbsize, const string& title, bool offscreen, bool debug, bool compat, int samples) {
    cv::Ptr<V4D> v4d = new V4D(size, fbsize, title, offscreen, debug, compat, samples);
    v4d->setVisible(!offscreen);
    return v4d;
}

V4D::V4D(const cv::Size& size, const cv::Size& fbsize, const string& title, bool offscreen, bool debug, bool compat, int samples) :
        initialSize_(size), title_(title), compat_(
                compat), samples_(samples), debug_(debug), viewport_(0, 0, size.width, size.height), zoomScale_(
                1), mousePos_(0, 0), stretch_(true), pool_(2) {
#ifdef __EMSCRIPTEN__
    printf(""); //makes sure we have FS as a dependency
#endif
        mainFbContext_ = new detail::FrameBufferContext(*this, fbsize.empty() ? size : fbsize, offscreen, title_, 3,
                2, compat_, samples_, debug_, nullptr, nullptr);

        nvgContext_ = new detail::NanoVGContext(*mainFbContext_);
        nguiContext_ = new detail::NanoguiContext(*mainFbContext_);
        clvaContext_ = new detail::CLVAContext(*mainFbContext_);
        glContext_ = new detail::GLContext(*mainFbContext_);
}

V4D::~V4D() {
    //don't delete form_. it is autmatically cleaned up by the base class (nanogui::Screen)
    if (glContext_)
        delete glContext_;
    if (nvgContext_)
        delete nvgContext_;
    if (nguiContext_)
        delete nguiContext_;
    if (clvaContext_)
        delete clvaContext_;
    if (mainFbContext_)
        delete mainFbContext_;
}

cv::ogl::Texture2D& V4D::texture() {
    return mainFbContext_->getTexture2D();
}

void V4D::setKeyboardEventCallback(
        std::function<bool(int key, int scancode, int action, int modifiers)> fn) {
    keyEventCb_ = fn;
}

bool V4D::keyboard_event(int key, int scancode, int action, int modifiers) {
    if (keyEventCb_)
        return keyEventCb_(key, scancode, action, modifiers);

    return nguiCtx().screen().keyboard_event(key, scancode, action, modifiers);
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

GLContext& V4D::glCtx() {
    assert(glContext_ != nullptr);
    return *glContext_;
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

bool V4D::hasGlCtx() {
    return glContext_ != nullptr;
}

void V4D::gl(std::function<void()> fn) {
    glCtx().render([=](const cv::Size& sz) {
        CV_UNUSED(sz);
        fn();
    });
}

void V4D::gl(std::function<void(const cv::Size&)> fn) {
    glCtx().render(fn);
}

void V4D::fb(std::function<void(cv::UMat&)> fn) {
    fbCtx().execute(fn);
}

void V4D::nvg(std::function<void()> fn) {
    nvgCtx().render([fn](const cv::Size& sz) {
        CV_UNUSED(sz);
        fn();
    });
}

void V4D::nvg(std::function<void(const cv::Size&)> fn) {
    nvgCtx().render(fn);
}

void V4D::nanogui(std::function<void(cv::v4d::FormHelper& form)> fn) {
    nguiCtx().build(fn);
}

void V4D::copyTo(cv::OutputArray m) {
    UMat um = m.getUMat();
    fbCtx().copyTo(um);
}

void V4D::copyFrom(cv::InputArray m) {
    UMat um = m.getUMat();
    fbCtx().copyFrom(um);
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

void V4D::run(std::function<bool()> fn) {
#ifndef __EMSCRIPTEN__
    while (keepRunning() && fn()) {
    }
#else
    emscripten_set_main_loop_arg(do_frame, &fn, -1, true);
#endif
}

void V4D::setSource(const Source& src) {
    if (!clvaCtx().hasContext())
        clvaCtx().copyContext();
    source_ = src;
}

void V4D::feed(cv::InputArray& in) {
        cv::UMat frame;
        clvaCtx().capture([&](cv::UMat& videoFrame) {
            in.copyTo(videoFrame);
        }, frame);

        fb([frame](cv::UMat& frameBuffer){
            frame.copyTo(frameBuffer);
        });
}

bool V4D::capture() {
    return this->capture([&](cv::UMat& videoFrame) {
        if (source_.isReady())
            source_().second.copyTo(videoFrame);
    });
}

bool V4D::capture(std::function<void(cv::UMat&)> fn) {
    if (!source_.isReady() || !source_.isOpen()) {
#ifndef __EMSCRIPTEN__
        return false;
#else
        return true;
#endif
    }
    if (futureReader_.valid()) {
        if (!futureReader_.get()) {
#ifndef __EMSCRIPTEN__
            return false;
#else
            return true;
#endif
        }
    }

    if(nextReaderFrame_.empty()) {
        if (!clvaCtx().capture(fn, nextReaderFrame_)) {
#ifndef __EMSCRIPTEN__
            return false;
#else
            return true;
#endif
        }
    }
    currentReaderFrame_ = nextReaderFrame_.clone();
    futureReader_ = pool_.enqueue(
        [](V4D* v, std::function<void(UMat&)> func, cv::UMat& frame) {
            return v->clvaCtx().capture(func, frame);
        }, this, fn, nextReaderFrame_);

    fb([this](cv::UMat& frameBuffer){
        currentReaderFrame_.copyTo(frameBuffer);
    });
    return true;
}

bool V4D::isSourceReady() {
    return source_.isReady();
}

void V4D::setSink(const Sink& sink) {
    if (!clvaCtx().hasContext())
        clvaCtx().copyContext();
    sink_ = sink;
}

void V4D::write() {
    this->write([&](const cv::UMat& videoFrame) {
        if (sink_.isReady())
            sink_(videoFrame);
    });
}

void V4D::write(std::function<void(const cv::UMat&)> fn) {
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

void V4D::setMouseDrag(bool d) {
    mouseDrag_ = d;
}

bool V4D::isMouseDrag() {
    return mouseDrag_;
}

void V4D::pan(int x, int y) {
    viewport_.x += x * zoomScale_;
    viewport_.y += y * zoomScale_;
}

void V4D::zoom(float factor) {
    if (zoomScale_ == 1 && viewport_.x == 0 && viewport_.y == 0 && factor > 1)
        return;

    double oldScale = zoomScale_;
    double origW = framebufferSize().width;
    double origH = framebufferSize().height;

    zoomScale_ *= factor;
    if (zoomScale_ <= 0.025) {
        zoomScale_ = 0.025;
        return;
    } else if (zoomScale_ > 1) {
        zoomScale_ = 1;
        viewport_.width = origW;
        viewport_.height = origH;
        if (factor > 1) {
            viewport_.x += log10(((viewport_.x * (1.0 - factor)) / viewport_.width) * 9 + 1.0)
                    * viewport_.width;
            viewport_.y += log10(((viewport_.y * (1.0 - factor)) / viewport_.height) * 9 + 1.0)
                    * viewport_.height;
        } else {
            viewport_.x += log10(((-viewport_.x * (1.0 - factor)) / viewport_.width) * 9 + 1.0)
                    * viewport_.width;
            viewport_.y += log10(((-viewport_.y * (1.0 - factor)) / viewport_.height) * 9 + 1.0)
                    * viewport_.height;
        }
        return;
    }

    cv::Vec2f offset;
    double oldW = (origW * oldScale);
    double oldH = (origH * oldScale);
    viewport_.width = std::min(zoomScale_ * origW, origW);
    viewport_.height = std::min(zoomScale_ * origH, origH);

    float delta_x;
    float delta_y;

    if (factor < 1.0) {
        offset = cv::Vec2f(viewport_.x, viewport_.y)
                - cv::Vec2f(mousePos_[0], origH - mousePos_[1]);
        delta_x = offset[0] / oldW;
        delta_y = offset[1] / oldH;
    } else {
        offset = cv::Vec2f(viewport_.x - (viewport_.width / 2.0),
                viewport_.y - (viewport_.height / 2.0)) - cv::Vec2f(viewport_.x, viewport_.y);
        delta_x = offset[0] / oldW;
        delta_y = offset[1] / oldH;
    }

    float x_offset;
    float y_offset;
    x_offset = delta_x * (viewport_.width - oldW);
    y_offset = delta_y * (viewport_.height - oldH);

    if (factor < 1.0) {
        viewport_.x += x_offset;
        viewport_.y += y_offset;
    } else {
        viewport_.x += x_offset;
        viewport_.y += y_offset;
    }
}

cv::Vec2f V4D::position() {
    return fbCtx().position();
}

cv::Vec2f V4D::getMousePosition() {
    return mousePos_;
}

void V4D::setMousePosition(int x, int y) {
    mousePos_ = { float(x), float(y) };
}

float V4D::zoomScale() {
    return zoomScale_;
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

cv::Size V4D::getWindowSize() {
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

void V4D::setShowFPS(bool s) {
    showFPS_ = s;
}

void V4D::setPrintFPS(bool p) {
    printFPS_ = p;
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

void V4D::setFrameBufferScaling(bool s) {
    stretch_ = s;
}

bool V4D::isFrameBufferScaling() {
    return stretch_;
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
        FrameBufferContext::GLScope glScope(glCtx().fbCtx(), GL_READ_FRAMEBUFFER);
        glCtx().fbCtx().blitFrameBufferToScreen(viewport(), glCtx().fbCtx().getWindowSize(), isFrameBufferScaling());
#ifndef __EMSCRIPTEN__
        glfwSwapBuffers(glCtx().fbCtx().getGLFWWindow());
#else
        emscripten_webgl_commit_frame();
#endif
    });

    run_sync_on_main<11>([this]() {
        FrameBufferContext::GLScope glScope(nvgCtx().fbCtx(), GL_READ_FRAMEBUFFER);
        nvgCtx().fbCtx().blitFrameBufferToScreen(viewport(), nvgCtx().fbCtx().getWindowSize(), isFrameBufferScaling());
#ifndef __EMSCRIPTEN__
        glfwSwapBuffers(nvgCtx().fbCtx().getGLFWWindow());
#else
        emscripten_webgl_commit_frame();
#endif
    });

    run_sync_on_main<12>([this]() {
        FrameBufferContext::GLScope glScope(nguiCtx().fbCtx(), GL_READ_FRAMEBUFFER);
        nguiCtx().fbCtx().blitFrameBufferToScreen(viewport(), nguiCtx().fbCtx().getWindowSize(), isFrameBufferScaling());
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

//        swapContextBuffers();

#ifdef __EMSCRIPTEN__
        nguiCtx().render(printFPS_, showFPS_);
#endif
        run_sync_on_main<6>([&, this](){
            {
               FrameBufferContext::GLScope glScope(fbCtx(), GL_READ_FRAMEBUFFER);
               fbCtx().blitFrameBufferToScreen(viewport(), fbCtx().getWindowSize(), isFrameBufferScaling());
            }
#ifndef __EMSCRIPTEN__
            nguiCtx().render(printFPS_, showFPS_);
#endif
            fbCtx().makeCurrent();
#ifndef __EMSCRIPTEN__
            glfwSwapBuffers(fbCtx().getGLFWWindow());
#else
            emscripten_webgl_commit_frame();
#endif
            {
#ifndef __EMSCRIPTEN__
                fbCtx().makeCurrent();
                GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
//#else
//                FrameBufferContext::GLScope glScope(fbCtx(), GL_FRAMEBUFFER);
//#endif
                GL_CHECK(glClearColor(0, 0, 0, 1));
                GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
#endif
            }

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

}
}
