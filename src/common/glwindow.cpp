#include "glwindow.hpp"

#include "util.hpp"

namespace kb {
void gl_check_error(const std::filesystem::path &file, unsigned int line, const char *expression) {
    int errorCode = glGetError();

    if (errorCode != 0) {
        std::cerr << "GL failed in " << file.filename() << " (" << line << ") : " << "\nExpression:\n   " << expression << "\nError code:\n   " << errorCode << "\n   " << std::endl;
        assert(false);
    }
}

GLWindow::GLWindow(const cv::Size &size, bool offscreen, const string &title, int major, int minor, int samples, bool debug) :
        size_(size), offscreen_(offscreen), title_(title), major_(major), minor_(minor), samples_(samples), debug_(debug) {
}

GLWindow::~GLWindow() {
    //don't delete form_. it is autmatically cleaned up by screen_
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
    glfwDestroyWindow(getGLFWWindow());
    glfwTerminate();
}

void GLWindow::initialize() {
    assert(glfwInit() == GLFW_TRUE);
    glfwSetErrorCallback(kb::error_callback);

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
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    glfwWindow_ = glfwCreateWindow(size_.width, size_.height, title_.c_str(), nullptr, nullptr);
    if (glfwWindow_ == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(-1);
    }
    glfwMakeContextCurrent(getGLFWWindow());
//        glfwSetFramebufferSizeCallback(getGLFWWindow(), frame_buffer_size_callback);

    screen_ = new nanogui::Screen();
    screen_->initialize(getGLFWWindow(), false);
    screen_->set_size(nanogui::Vector2i(size_.width, size_.height));
    form_ = new nanogui::FormHelper(screen_);

    glfwSetWindowUserPointer(getGLFWWindow(), this);

    glfwSetCursorPosCallback(getGLFWWindow(), [](GLFWwindow *glfwWin, double x, double y) {
        GLWindow *win = (GLWindow*) glfwGetWindowUserPointer(glfwWin);
        win->screen_->cursor_pos_callback_event(x, y);
    }
    );
    glfwSetMouseButtonCallback(getGLFWWindow(), [](GLFWwindow *glfwWin, int button, int action, int modifiers) {
        GLWindow *win = (GLWindow*) glfwGetWindowUserPointer(glfwWin);
        win->screen_->mouse_button_callback_event(button, action, modifiers);
    }
    );
    glfwSetKeyCallback(getGLFWWindow(), [](GLFWwindow *glfwWin, int key, int scancode, int action, int mods) {
        GLWindow *win = (GLWindow*) glfwGetWindowUserPointer(glfwWin);
        win->screen_->key_callback_event(key, scancode, action, mods);
    }
    );
    glfwSetCharCallback(getGLFWWindow(), [](GLFWwindow *glfwWin, unsigned int codepoint) {
        GLWindow *win = (GLWindow*) glfwGetWindowUserPointer(glfwWin);
        win->screen_->char_callback_event(codepoint);
    }
    );
    glfwSetDropCallback(getGLFWWindow(), [](GLFWwindow *glfwWin, int count, const char **filenames) {
        GLWindow *win = (GLWindow*) glfwGetWindowUserPointer(glfwWin);
        win->screen_->drop_callback_event(count, filenames);
    }
    );
    glfwSetScrollCallback(getGLFWWindow(), [](GLFWwindow *glfwWin, double x, double y) {
        GLWindow *win = (GLWindow*) glfwGetWindowUserPointer(glfwWin);
        win->screen_->scroll_callback_event(x, y);
    }
    );
    glfwSetFramebufferSizeCallback(getGLFWWindow(), [](GLFWwindow *glfwWin, int width, int height) {
        GLWindow *win = (GLWindow*) glfwGetWindowUserPointer(glfwWin);
        win->screen_->resize_callback_event(width, height);
    }
    );
    clglContext_ = new CLGLContext(getSize());
    clvaContext_ = new CLVAContext(*clglContext_);
    nvgContext_ = new NanoVGContext(*this, getNVGcontext(), *clglContext_);
}

nanogui::FormHelper* GLWindow::form() {
    return form_;
}

CLGLContext& GLWindow::clgl() {
    return *clglContext_;
}

CLVAContext& GLWindow::clva() {
    return *clvaContext_;
}

NanoVGContext& GLWindow::nvg() {
    return *nvgContext_;
}

void GLWindow::render(std::function<void(const cv::Size&)> fn) {
    clgl().render(fn);
}

void GLWindow::compute(std::function<void(cv::UMat&)> fn) {
    clgl().compute(fn);
}

void GLWindow::renderNVG(std::function<void(NVGcontext*, const cv::Size&)> fn) {
    nvg().render(fn);
}

bool GLWindow::captureVA() {
    return clva().capture([=, this](cv::UMat &videoFrame) {
        *(this->capture_) >> videoFrame;
    });
}

void GLWindow::writeVA() {
    clva().write([=, this](const cv::UMat &videoFrame) {
        *(this->writer_) << videoFrame;
    });
}

void GLWindow::makeGLFWContextCurrent() {
    glfwMakeContextCurrent(getGLFWWindow());
}

cv::VideoWriter& GLWindow::makeVAWriter(const string &outputFilename, const int fourcc, const float fps, const cv::Size &frameSize, const int vaDeviceIndex) {
    writer_ = new cv::VideoWriter(outputFilename, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, frameSize, { cv::VIDEOWRITER_PROP_HW_DEVICE, vaDeviceIndex, cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI, cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1 });

    if (!clva().hasContext()) {
        clva().copyContext();
    }
    return *writer_;
}

cv::VideoCapture& GLWindow::makeVACapture(const string &intputFilename, const int vaDeviceIndex) {
    //Initialize MJPEG HW decoding using VAAPI
    capture_ = new cv::VideoCapture(intputFilename, cv::CAP_FFMPEG, { cv::CAP_PROP_HW_DEVICE, vaDeviceIndex, cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI, cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL, 1 });

    if (!clva().hasContext()) {
        clva().copyContext();
    }

    return *capture_;
}

void GLWindow::clear(const cv::Scalar &rgba) {
    const float &r = rgba[0] / 255.0f;
    const float &g = rgba[1] / 255.0f;
    const float &b = rgba[2] / 255.0f;
    const float &a = rgba[3] / 255.0f;
    GL_CHECK(glClearColor(r, g, b, a));
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
}

cv::Size GLWindow::getFrameBufferSize() {
    int fbW, fbH;
    glfwGetFramebufferSize(getGLFWWindow(), &fbW, &fbH);
    return {fbW, fbH};
}

cv::Size GLWindow::getSize() {
    return size_;
}

float GLWindow::getPixelRatio() {
#if defined(EMSCRIPTEN)
        return emscripten_get_device_pixel_ratio();
#else
    float xscale, yscale;
    glfwGetWindowContentScale(getGLFWWindow(), &xscale, &yscale);
    return xscale;
#endif
}

void GLWindow::setSize(const cv::Size &sz) {
    screen_->set_size(nanogui::Vector2i(sz.width / getPixelRatio(), sz.height / getPixelRatio()));
    glfwSetWindowSize(getGLFWWindow(), sz.width, sz.height);
}

bool GLWindow::isFullscreen() {
    return glfwGetWindowMonitor(getGLFWWindow()) != nullptr;
}

void GLWindow::setFullscreen(bool f) {
    auto monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode *mode = glfwGetVideoMode(monitor);
    if (f) {
        glfwSetWindowMonitor(getGLFWWindow(), monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
    } else {
        glfwSetWindowMonitor(getGLFWWindow(), nullptr, 0, 0, getSize().width, getSize().width, mode->refreshRate);
    }
    setSize(getSize());
}

bool GLWindow::isResizable() {
    return glfwGetWindowAttrib(getGLFWWindow(), GLFW_RESIZABLE) == GLFW_TRUE;
}

void GLWindow::setResizable(bool r) {
    glfwWindowHint(GLFW_RESIZABLE, r ? GLFW_TRUE : GLFW_FALSE);
}

bool GLWindow::isVisible() {
    return glfwGetWindowAttrib(getGLFWWindow(), GLFW_VISIBLE) == GLFW_TRUE;
}

void GLWindow::setVisible(bool v) {
    setSize(getSize());
    screen_->set_visible(v);
    screen_->perform_layout();
    glfwWindowHint(GLFW_VISIBLE, v ? GLFW_TRUE : GLFW_FALSE);
}

bool GLWindow::isOffscreen() {
    return offscreen_;
}

nanogui::Window* GLWindow::makeWindow(int x, int y, const string &title) {
    return form()->add_window(nanogui::Vector2i(x, y), title);
}

nanogui::Label* GLWindow::makeGroup(const string &label) {
    return form()->add_group(label);
}

nanogui::detail::FormWidget<bool>* GLWindow::makeFormVariable(const string &name, bool &v, const string &tooltip) {
    auto var = form()->add_variable(name, v);
    if (!tooltip.empty())
        var->set_tooltip(tooltip);
    return var;
}

void GLWindow::setUseOpenCL(bool u) {
    clglContext_->getCLExecContext().setUseOpenCL(u);
    clvaContext_->getCLExecContext().setUseOpenCL(u);
    cv::ocl::setUseOpenCL(u);
}

bool GLWindow::display() {
    bool result = true;
    if (!offscreen_) {
        glfwPollEvents();
        screen_->draw_contents();
        clglContext_->blitFrameBufferToScreen();
        screen_->draw_widgets();
        glfwSwapBuffers(glfwWindow_);
        result = !glfwWindowShouldClose(glfwWindow_);
    }

    return result;
}

bool GLWindow::isClosed() {
    return closed_;

}
void GLWindow::close() {
    setVisible(false);
    closed_ = true;
}

GLFWwindow* GLWindow::getGLFWWindow() {
    return glfwWindow_;
}

NVGcontext* GLWindow::getNVGcontext() {
    return screen_->nvg_context();
}
} /* namespace kb */
