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

void gl_check_error(const std::filesystem::path& file, unsigned int line, const char* expression) {
    int errorCode = glGetError();

    if (errorCode != 0) {
        std::stringstream ss;
        ss << "GL failed in " << file.filename() << " (" << line << ") : " << "\nExpression:\n   "
                << expression << "\nError code:\n   " << errorCode;
        throw std::runtime_error(ss.str());
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

cv::Ptr<V4D> V4D::make(const cv::Size& size, const string& title, bool debug) {
    cv::Ptr<V4D> v4d = new V4D(size, false, title, 4, 6, true, 0, debug);
    v4d->setVisible(true);
    return v4d;
}

cv::Ptr<V4D> V4D::make(const cv::Size& initialSize, bool offscreen, const string& title, int major,
        int minor, bool compat, int samples, bool debug) {
    return new V4D(initialSize, offscreen, title, major, minor, compat, samples, debug);
}

V4D::V4D(const cv::Size& size, bool offscreen, const string& title, int major, int minor,
        bool compat, int samples, bool debug) :
        initialSize_(size), offscreen_(offscreen), title_(title), major_(major), minor_(minor), compat_(
                compat), samples_(samples), debug_(debug), viewport_(0, 0, size.width, size.height), scale_(
                1), mousePos_(0, 0), stretch_(true), pool_(2) {
#ifdef __EMSCRIPTEN__
    printf(""); //makes sure we have FS as a dependency
#endif
        mainFbContext_ = new detail::FrameBufferContext(*this, initialSize_, offscreen_, title_, major_,
                minor_, compat_, samples_, debug_, nullptr, nullptr);

        nvgContext_ = new detail::NanoVGContext(*this, *mainFbContext_);
        nguiContext_ = new detail::NanoguiContext(*this, *mainFbContext_);
        clvaContext_ = new detail::CLVAContext(*this, *mainFbContext_);
        glContext_ = new detail::GLContext(*this, *mainFbContext_);
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

cv::Size V4D::getVideoFrameSize() {
    return clvaCtx().getVideoFrameSize();
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

#ifdef __EMSCRIPTEN__
static void do_frame(void* void_fn_ptr) {
     auto* fn_ptr = reinterpret_cast<std::function<bool()>*>(void_fn_ptr);
     if (fn_ptr) {
         auto& fn = *fn_ptr;
         fn();
     }
 }
#endif

void V4D::run(std::function<bool()> fn) {
#ifndef __EMSCRIPTEN__
    while (keepRunning() && fn())
        ;
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
            cvtColor(frame,frameBuffer, cv::COLOR_RGB2BGRA);
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
        [](V4D* v, std::function<void(UMat&)> fn, cv::UMat& frame) {
            return v->clvaCtx().capture(fn, frame);
        }, this, fn, nextReaderFrame_);

    fb([this](cv::UMat& frameBuffer){
        cvtColor(currentReaderFrame_,frameBuffer, cv::COLOR_RGB2BGRA);
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

    futureWriter_ = pool_.enqueue([](V4D* v, std::function<void(const UMat&)> fn, cv::UMat& frame) {
        v->clvaCtx().write(fn, frame);
    }, this, fn, currentWriterFrame_);
}

bool V4D::isSinkReady() {
    return sink_.isReady();
}

void V4D::clear(const cv::Scalar& bgra) {
    this->gl([&]() {
        const float& b = bgra[0] / 255.0f;
        const float& g = bgra[1] / 255.0f;
        const float& r = bgra[2] / 255.0f;
        const float& a = bgra[3] / 255.0f;
        GL_CHECK(glClearColor(r, g, b, a));
        GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));
    });
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
    viewport_.x += x * scale_;
    viewport_.y += y * scale_;
}

void V4D::zoom(float factor) {
    if (scale_ == 1 && viewport_.x == 0 && viewport_.y == 0 && factor > 1)
        return;

    double oldScale = scale_;
    double origW = getFrameBufferSize().width;
    double origH = getFrameBufferSize().height;

    scale_ *= factor;
    if (scale_ <= 0.025) {
        scale_ = 0.025;
        return;
    } else if (scale_ > 1) {
        scale_ = 1;
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
    viewport_.width = std::min(scale_ * origW, origW);
    viewport_.height = std::min(scale_ * origH, origH);

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

cv::Vec2f V4D::getPosition() {
    fbCtx().makeCurrent();
    int x, y;
    glfwGetWindowPos(getGLFWWindow(), &x, &y);
    return cv::Vec2f(x, y);
}

cv::Vec2f V4D::getMousePosition() {
    return mousePos_;
}

void V4D::setMousePosition(int x, int y) {
    mousePos_ = { float(x), float(y) };
}

float V4D::getScale() {
    return scale_;
}

cv::Rect& V4D::viewport() {
    return viewport_;
}

float V4D::getXPixelRatio() {
    fbCtx().makeCurrent();
#ifdef __EMSCRIPTEN__
    float r = emscripten_get_device_pixel_ratio();

    return r;
#else
    float xscale, yscale;
    glfwGetWindowContentScale(getGLFWWindow(), &xscale, &yscale);

    return xscale;
#endif
}

float V4D::getYPixelRatio() {
    fbCtx().makeCurrent();
#ifdef __EMSCRIPTEN__
    float r = emscripten_get_device_pixel_ratio();

    return r;
#else
    float xscale, yscale;
    glfwGetWindowContentScale(getGLFWWindow(), &xscale, &yscale);

    return yscale;
#endif
}

cv::Size V4D::getNativeFrameBufferSize() {
    fbCtx().makeCurrent();
        int w, h;
        glfwGetFramebufferSize(getGLFWWindow(), &w, &h);
        return cv::Size{w, h};
}

cv::Size V4D::getFrameBufferSize() {
    return fbCtx().getSize();
}

cv::Size V4D::getInitialSize() {
    return initialSize_;
}

void V4D::resizeWindow(const cv::Size& sz) {
    fbCtx().resizeWindow(sz);
}

bool V4D::isFullscreen() {
    fbCtx().makeCurrent();
    return glfwGetWindowMonitor(getGLFWWindow()) != nullptr;
}

void V4D::setFullscreen(bool f) {
    fbCtx().makeCurrent();
        auto monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        if (f) {
            glfwSetWindowMonitor(getGLFWWindow(), monitor, 0, 0, mode->width, mode->height,
                    mode->refreshRate);
            resizeWindow(getNativeFrameBufferSize());
        } else {
            glfwSetWindowMonitor(getGLFWWindow(), nullptr, 0, 0, getInitialSize().width,
                    getInitialSize().height, 0);
            resizeWindow(getInitialSize());
        }
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

bool V4D::isOffscreen() {
    return offscreen_;
}

void V4D::setOffscreen(bool o) {
    offscreen_ = o;
    setVisible(!o);
}

void V4D::setStretching(bool s) {
    stretch_ = s;
}

bool V4D::isStretching() {
    return stretch_;
}

void V4D::setDefaultKeyboardEventCallback() {
    setKeyboardEventCallback([&](int key, int scancode, int action, int modifiers) {
        CV_UNUSED(scancode);
        CV_UNUSED(modifiers);
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            setOffscreen(!isOffscreen());
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

bool V4D::display() {
    bool result = true;
    if (!offscreen_) {
//        {
//            FrameBufferContext::GLScope glScope(clvaCtx().fbCtx());
//            clvaCtx().fbCtx().blitFrameBufferToScreen(viewport(), getWindowSize(), isStretching());
//            clvaCtx().fbCtx().makeCurrent();
//            glfwSwapBuffers(clvaCtx().fbCtx().getGLFWWindow());
//        }
//        {
//            FrameBufferContext::GLScope glScope(glCtx().fbCtx());
//            glCtx().fbCtx().blitFrameBufferToScreen(viewport(), getWindowSize(), isStretching());
//            glCtx().fbCtx().makeCurrent();
//            glfwSwapBuffers(glCtx().fbCtx().getGLFWWindow());
//        }
//        {
//            FrameBufferContext::GLScope glScope(nvgCtx().fbCtx());
//            nvgCtx().fbCtx().blitFrameBufferToScreen(viewport(), getWindowSize(), isStretching());
//            nvgCtx().fbCtx().makeCurrent();
//            glfwSwapBuffers(nvgCtx().fbCtx().getGLFWWindow());
//        }
//        run_sync_on_main([this](){
//            FrameBufferContext::GLScope glScope(nguiCtx().fbCtx());
//            nguiCtx().fbCtx().blitFrameBufferToScreen(viewport(), getWindowSize(), isStretching());
//            nguiCtx().fbCtx().makeCurrent();
//            glfwSwapBuffers(nguiCtx().fbCtx().getGLFWWindow());
//        });

        nguiCtx().render();

        run_sync_on_main([&, this](){
            FrameBufferContext::GLScope glScope(fbCtx());
            fbCtx().blitFrameBufferToScreen(viewport(), fbCtx().getWindowSize(), isStretching());
            glfwSwapBuffers(fbCtx().getGLFWWindow());
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
    return closed_;
}

void V4D::close() {
    setVisible(false);
    closed_ = true;
}

GLFWwindow* V4D::getGLFWWindow() {
    return fbCtx().getGLFWWindow();
}

void V4D::printSystemInfo() {
    run_sync_on_main([this](){
        fbCtx().makeCurrent();
        cerr << "OpenGL Version: " << getGlInfo() << endl;
        cerr << "OpenCL Platforms: " << getClInfo() << endl;
    });
}

void V4D::updateFps(bool graphical) {
    if (frameCount() > 0) {
        tick_.stop();

        if (tick_.getTimeMilli() > 50) {
            cerr << "FPS : " << (fps_ = tick_.getFPS());
#ifndef __EMSCRIPTEN__
            cerr << '\r';
#else
            cerr << endl;
#endif
            tick_.reset();
        }

        if (graphical) {
            this->nvg([this]() {
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

    tick_.start();
}
}
}
