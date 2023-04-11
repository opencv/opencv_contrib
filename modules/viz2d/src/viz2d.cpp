// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/viz2d/viz2d.hpp"
#include "detail/clvacontext.hpp"
#include "detail/framebuffercontext.hpp"
#include "detail/nanovgcontext.hpp"

#ifdef __EMSCRIPTEN__
#  include <emscripten.h>
#endif

namespace cv {
namespace viz {
namespace detail {
void gl_check_error(const std::filesystem::path& file, unsigned int line, const char* expression) {
    int errorCode = glGetError();

    if (errorCode != 0) {
        std::cerr << "GL failed in " << file.filename() << " (" << line << ") : "
                << "\nExpression:\n   " << expression << "\nError code:\n   " << errorCode
                << "\n   " << std::endl;
        assert(false);
    }
}

void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error: %s\n", description);
}

bool contains_absolute(nanogui::Widget* w, const nanogui::Vector2i& p) {
    nanogui::Vector2i d = p - w->absolute_position();
    return d.x() >= 0 && d.y() >= 0 && d.x() < w->size().x() && d.y() < w->size().y();
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

void resizeKeepAspectRatio(const cv::UMat& src, cv::UMat& output, const cv::Size& dstSize,
        const cv::Scalar& bgcolor) {
    double h1 = dstSize.width * (src.rows / (double) src.cols);
    double w2 = dstSize.height * (src.cols / (double) src.rows);
    if (h1 <= dstSize.height) {
        cv::resize(src, output, cv::Size(dstSize.width, h1));
    } else {
        cv::resize(src, output, cv::Size(w2, dstSize.height));
    }

    int top = (dstSize.height - output.rows) / 2;
    int down = (dstSize.height - output.rows + 1) / 2;
    int left = (dstSize.width - output.cols) / 2;
    int right = (dstSize.width - output.cols + 1) / 2;

    cv::copyMakeBorder(output, output, top, down, left, right, cv::BORDER_CONSTANT, bgcolor);
}

cv::Ptr<Viz2D> Viz2D::make(const cv::Size& size, const string& title, bool debug) {
    cv::Ptr<Viz2D> v2d = new Viz2D(size, size, false, title, 4, 6, 0, debug);
    return v2d;
}

cv::Ptr<Viz2D> Viz2D::make(const cv::Size& initialSize, const cv::Size& frameBufferSize,
        bool offscreen, const string& title, int major, int minor, int samples, bool debug) {
    return new Viz2D(initialSize, frameBufferSize, offscreen, title, major, minor, samples, debug);
}

Viz2D::Viz2D(const cv::Size& size, const cv::Size& frameBufferSize, bool offscreen,
        const string& title, int major, int minor, int samples, bool debug) :
        initialSize_(size), frameBufferSize_(frameBufferSize), viewport_(0, 0,
                frameBufferSize.width, frameBufferSize.height), scale_(1), mousePos_(0, 0), offscreen_(
                offscreen), stretch_(false), title_(title), major_(major), minor_(minor), samples_(
                samples), debug_(debug) {
    assert(
            frameBufferSize_.width >= initialSize_.width
                    && frameBufferSize_.height >= initialSize_.height);

    initializeWindowing();
}

Viz2D::~Viz2D() {
    //don't delete form_. it is autmatically cleaned up by the base class (nanogui::Screen)
    if (screen_)
        delete screen_;
    if (writer_)
        delete writer_;
    if (capture_)
        delete capture_;
    if (nvgContext_)
        delete nvgContext_;
    if (clvaContext_)
        delete clvaContext_;
    if (clglContext_)
        delete clglContext_;
}

bool Viz2D::initializeWindowing() {
    if (glfwInit() != GLFW_TRUE)
        return false;

    glfwSetErrorCallback(cv::viz::glfw_error_callback);

    if (debug_)
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);

    if (offscreen_)
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    glfwSetTime(0);

#ifdef __APPLE__
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#elif defined(VIZ2D_USE_ES3)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API) ;
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major_);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor_);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
#endif
    glfwWindowHint(GLFW_SAMPLES, samples_);
    glfwWindowHint(GLFW_RED_BITS, 8);
    glfwWindowHint(GLFW_GREEN_BITS, 8);
    glfwWindowHint(GLFW_BLUE_BITS, 8);
    glfwWindowHint(GLFW_ALPHA_BITS, 8);
    glfwWindowHint(GLFW_STENCIL_BITS, 8);
    glfwWindowHint(GLFW_DEPTH_BITS, 24);
#ifndef __EMSCRIPTEN__
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
#else
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
#endif
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    /* I figure we don't need double buffering because the FBO (and the bound texture) is our backbuffer that
     * we blit to the front on every iteration.
     * On X11, wayland and in WASM it works and boosts performance a bit.
     */
    glfwWindowHint(GLFW_DOUBLEBUFFER, GL_FALSE);

    glfwWindow_ = glfwCreateWindow(initialSize_.width, initialSize_.height, title_.c_str(), nullptr,
            nullptr);
    if (glfwWindow_ == NULL) {
        return false;
    }
    glfwMakeContextCurrent(getGLFWWindow());

    screen_ = new nanogui::Screen();
    screen().initialize(getGLFWWindow(), false);
    form_ = new FormHelper(&screen());

    this->setWindowSize(initialSize_);

    glfwSetWindowUserPointer(getGLFWWindow(), this);

    glfwSetCursorPosCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, double x, double y) {
        Viz2D* v2d = reinterpret_cast<Viz2D*>(glfwGetWindowUserPointer(glfwWin));
        v2d->screen().cursor_pos_callback_event(x, y);
        auto cursor = v2d->getMousePosition();
        auto diff = cursor - cv::Vec2f(x, y);
        if (v2d->isMouseDrag()) {
            v2d->pan(diff[0], -diff[1]);
        }
        v2d->setMousePosition(x, y);
    }
    );
    glfwSetMouseButtonCallback(getGLFWWindow(),
            [](GLFWwindow* glfwWin, int button, int action, int modifiers) {
                Viz2D* v2d = reinterpret_cast<Viz2D*>(glfwGetWindowUserPointer(glfwWin));
                v2d->screen().mouse_button_callback_event(button, action, modifiers);
                if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                    v2d->setMouseDrag(action == GLFW_PRESS);
                }
            }
    );
    glfwSetKeyCallback(getGLFWWindow(),
            [](GLFWwindow* glfwWin, int key, int scancode, int action, int mods) {
                Viz2D* v2d = reinterpret_cast<Viz2D*>(glfwGetWindowUserPointer(glfwWin));
                v2d->screen().key_callback_event(key, scancode, action, mods);
            }
    );
    glfwSetCharCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, unsigned int codepoint) {
        Viz2D* v2d = reinterpret_cast<Viz2D*>(glfwGetWindowUserPointer(glfwWin));
        v2d->screen().char_callback_event(codepoint);
    }
    );
    glfwSetDropCallback(getGLFWWindow(),
            [](GLFWwindow* glfwWin, int count, const char** filenames) {
                Viz2D* v2d = reinterpret_cast<Viz2D*>(glfwGetWindowUserPointer(glfwWin));
                v2d->screen().drop_callback_event(count, filenames);
            }
    );
    glfwSetScrollCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, double x, double y) {
        Viz2D* v2d = reinterpret_cast<Viz2D*>(glfwGetWindowUserPointer(glfwWin));
        std::vector<nanogui::Widget*> widgets;
        find_widgets(&v2d->screen(), widgets);
        for (auto* w : widgets) {
            auto mousePos = nanogui::Vector2i(v2d->getMousePosition()[0] / v2d->getXPixelRatio(), v2d->getMousePosition()[1] / v2d->getYPixelRatio());
    if(contains_absolute(w, mousePos)) {
        v2d->screen().scroll_callback_event(x, y);
        return;
    }
}

        v2d->zoom(y < 0 ? 1.1 : 0.9);
    }
    );

//FIXME resize internal buffers?
//    glfwSetWindowContentScaleCallback(getGLFWWindow(),
//        [](GLFWwindow* glfwWin, float xscale, float yscale) {
//        }
//    );

    glfwSetFramebufferSizeCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, int width, int height) {
        Viz2D* v2d = reinterpret_cast<Viz2D*>(glfwGetWindowUserPointer(glfwWin));
        v2d->screen().resize_callback_event(width, height);
    });

    clglContext_ = new detail::FrameBufferContext(this->getFrameBufferSize());
    clvaContext_ = new detail::CLVAContext(*clglContext_);
    nvgContext_ = new detail::NanoVGContext(*this, getNVGcontext(), *clglContext_);
    return true;
}

cv::ogl::Texture2D& Viz2D::texture() {
    return clglContext_->getTexture2D();
}

FormHelper& Viz2D::form() {
    return *form_;
}

void Viz2D::setKeyboardEventCallback(
        std::function<bool(int key, int scancode, int action, int modifiers)> fn) {
    keyEventCb_ = fn;
}

bool Viz2D::keyboard_event(int key, int scancode, int action, int modifiers) {
    if (keyEventCb_)
        return keyEventCb_(key, scancode, action, modifiers);

    if (screen().keyboard_event(key, scancode, action, modifiers))
        return true;
    return false;
}

FrameBufferContext& Viz2D::fb() {
    assert(clglContext_ != nullptr);
    makeCurrent();
    return *clglContext_;
}

CLVAContext& Viz2D::clva() {
    assert(clvaContext_ != nullptr);
    makeCurrent();
    return *clvaContext_;
}

NanoVGContext& Viz2D::nvg() {
    assert(nvgContext_ != nullptr);
    makeCurrent();
    return *nvgContext_;
}

nanogui::Screen& Viz2D::screen() {
    assert(screen_ != nullptr);
    makeCurrent();
    return *screen_;
}

cv::Size Viz2D::getVideoFrameSize() {
    return clva().getVideoFrameSize();
}

void Viz2D::gl(std::function<void(const cv::Size&)> fn) {
    auto fbSize = getFrameBufferSize();
#ifndef __EMSCRIPTEN__
    detail::CLExecScope_t scope(fb().getCLExecContext());
#endif
    detail::FrameBufferContext::GLScope glScope(fb());
    fn(fbSize);
}

void Viz2D::fb(std::function<void(cv::UMat&)> fn) {
    fb().execute(fn);
}

void Viz2D::nvg(std::function<void(const cv::Size&)> fn) {
    nvg().render(fn);
}

void Viz2D::nanogui(std::function<void(FormHelper& form)> fn) {
    fn(form());
}

void Viz2D::run(std::function<bool()> fn) {
#ifndef __EMSCRIPTEN__
    while (keepRunning() && fn())
        ;
#else
    emscripten_set_main_loop(fn, -1, true);
#endif
}

void Viz2D::setSource(const Source& src) {
    if (!clva().hasContext())
        clva().copyContext();
    source_ = src;
}

void Viz2D::feed(cv::InputArray& in) {
    this->capture([&](cv::OutputArray& videoFrame) {
        in.getUMat().copyTo(videoFrame);
    });
}

bool Viz2D::capture() {
    return this->capture([&](cv::UMat& videoFrame) {
        if (source_.isReady())
            source_().second.copyTo(videoFrame);
    });
}

bool Viz2D::capture(std::function<void(cv::UMat&)> fn) {
    return clva().capture(fn);
}

bool Viz2D::isSourceReady() {
    return source_.isReady();
}

void Viz2D::setSink(const Sink& sink) {
    if (!clva().hasContext())
        clva().copyContext();
    sink_ = sink;
}

void Viz2D::write() {
    this->write([&](const cv::UMat& videoFrame) {
        if (sink_.isReady())
            sink_(videoFrame);
    });
}

void Viz2D::write(std::function<void(const cv::UMat&)> fn) {
    clva().write(fn);
}

bool Viz2D::isSinkReady() {
    return sink_.isReady();
}

void Viz2D::makeCurrent() {
    glfwMakeContextCurrent(getGLFWWindow());
}

void Viz2D::makeNoneCurrent() {
    glfwMakeContextCurrent(nullptr);
}

void Viz2D::clear(const cv::Scalar& bgra) {
    const float& b = bgra[0] / 255.0f;
    const float& g = bgra[1] / 255.0f;
    const float& r = bgra[2] / 255.0f;
    const float& a = bgra[3] / 255.0f;
    GL_CHECK(glClearColor(r, g, b, a));
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));
}

void Viz2D::showGui(bool s) {
    auto children = screen().children();
    for (auto* child : children) {
        child->set_visible(s);
    }
}

void Viz2D::setMouseDrag(bool d) {
    mouseDrag_ = d;
}

bool Viz2D::isMouseDrag() {
    return mouseDrag_;
}

void Viz2D::pan(int x, int y) {
    viewport_.x += x * scale_;
    viewport_.y += y * scale_;
}

void Viz2D::zoom(float factor) {
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

cv::Vec2f Viz2D::getPosition() {
    makeCurrent();
    int x, y;
    glfwGetWindowPos(getGLFWWindow(), &x, &y);
    return {float(x), float(y)};
}

cv::Vec2f Viz2D::getMousePosition() {
    return mousePos_;
}

void Viz2D::setMousePosition(int x, int y) {
    mousePos_ = { float(x), float(y) };
}

float Viz2D::getScale() {
    return scale_;
}

cv::Rect Viz2D::getViewport() {
    return viewport_;
}

cv::Size Viz2D::getNativeFrameBufferSize() {
    makeCurrent();
    int w, h;
    glfwGetFramebufferSize(getGLFWWindow(), &w, &h);
    return {w, h};
}

cv::Size Viz2D::getFrameBufferSize() {
    return frameBufferSize_;
}

cv::Size Viz2D::getWindowSize() {
    makeCurrent();
    int w, h;
    glfwGetWindowSize(getGLFWWindow(), &w, &h);
    return {w, h};
}

cv::Size Viz2D::getInitialSize() {
    return initialSize_;
}

float Viz2D::getXPixelRatio() {
    makeCurrent();
#ifdef __EMSCRIPTEN__
    return emscripten_get_device_pixel_ratio();
#else
    float xscale, yscale;
    glfwGetWindowContentScale(getGLFWWindow(), &xscale, &yscale);
    return xscale;
#endif
}

float Viz2D::getYPixelRatio() {
    makeCurrent();
#ifdef __EMSCRIPTEN__
    return emscripten_get_device_pixel_ratio();
#else
    float xscale, yscale;
    glfwGetWindowContentScale(getGLFWWindow(), &xscale, &yscale);
    return yscale;
#endif
}

void Viz2D::setWindowSize(const cv::Size& sz) {
    makeCurrent();
    screen().set_size(nanogui::Vector2i(sz.width / getXPixelRatio(), sz.height / getYPixelRatio()));
}

bool Viz2D::isFullscreen() {
    makeCurrent();
    return glfwGetWindowMonitor(getGLFWWindow()) != nullptr;
}

void Viz2D::setFullscreen(bool f) {
    makeCurrent();
    auto monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    if (f) {
        glfwSetWindowMonitor(getGLFWWindow(), monitor, 0, 0, mode->width, mode->height,
                mode->refreshRate);
        setWindowSize(getNativeFrameBufferSize());
    } else {
        glfwSetWindowMonitor(getGLFWWindow(), nullptr, 0, 0, getInitialSize().width,
                getInitialSize().height, 0);
        setWindowSize(getInitialSize());
    }
}

bool Viz2D::isResizable() {
    makeCurrent();
    return glfwGetWindowAttrib(getGLFWWindow(), GLFW_RESIZABLE) == GLFW_TRUE;
}

void Viz2D::setResizable(bool r) {
    makeCurrent();
    glfwWindowHint(GLFW_RESIZABLE, r ? GLFW_TRUE : GLFW_FALSE);
}

bool Viz2D::isVisible() {
    makeCurrent();
    return glfwGetWindowAttrib(getGLFWWindow(), GLFW_VISIBLE) == GLFW_TRUE;
}

void Viz2D::setVisible(bool v) {
    makeCurrent();
    glfwWindowHint(GLFW_VISIBLE, v ? GLFW_TRUE : GLFW_FALSE);
    screen().set_visible(v);
    screen().perform_layout();
}

bool Viz2D::isOffscreen() {
    return offscreen_;
}

void Viz2D::setOffscreen(bool o) {
    offscreen_ = o;
    setVisible(!o);
}

void Viz2D::setStretching(bool s) {
    stretch_ = s;
}

bool Viz2D::isStretching() {
    return stretch_;
}

void Viz2D::setDefaultKeyboardEventCallback() {
    setKeyboardEventCallback([&](int key, int scancode, int action, int modifiers) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            setOffscreen(!isOffscreen());
            return true;
        } else if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
            auto children = screen().children();
            for (auto* child : children) {
                child->set_visible(!child->visible());
            }

            return true;
        }
        return false;
    });
}

bool Viz2D::display() {
    bool result = true;
    if (!offscreen_) {
        makeCurrent();
        screen().draw_contents();
#ifndef __EMSCRIPTEN__
        clglContext_->blitFrameBufferToScreen(getViewport(), getWindowSize(), isStretching());
#else
        clglContext_->blitFrameBufferToScreen(getViewport(), getInitialSize(), isStretching());
#endif
        screen().draw_widgets();
        glfwSwapBuffers(glfwWindow_);
        glfwPollEvents();
        result = !glfwWindowShouldClose(glfwWindow_);
    }

    return result;
}

bool Viz2D::isClosed() {
    return closed_;

}
void Viz2D::close() {
    setVisible(false);
    closed_ = true;
}

GLFWwindow* Viz2D::getGLFWWindow() {
    return glfwWindow_;
}

NVGcontext* Viz2D::getNVGcontext() {
    return screen().nvg_context();
}
}
}
