#ifndef SRC_SUBSYSTEMS_HPP_
#define SRC_SUBSYSTEMS_HPP_

#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <filesystem>
#include <thread>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#define NANOGUI_USE_OPENGL
#include <nanogui/nanogui.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/opengl.hpp>
#define GLFW_INCLUDE_GLCOREARB
#include <GLFW/glfw3.h>
#include <nanogui/opengl.h>

using std::cout;
using std::cerr;
using std::endl;
using std::string;

namespace kb {

typedef cv::ocl::OpenCLExecutionContext CLExecContext_t;
typedef cv::ocl::OpenCLExecutionContextScope CLExecScope_t;

void gl_check_error(const std::filesystem::path &file, unsigned int line, const char *expression) {
    GLint errorCode = glGetError();

    if (errorCode != GL_NO_ERROR) {
        cerr << "GL failed in " << file.filename() << " (" << line << ") : " << "\nExpression:\n   " << expression << "\nError code:\n   " << errorCode << "\n   " << endl;
        assert(false);
    }
}
#define GL_CHECK(expr)                            \
    expr;                                        \
    kb::gl_check_error(__FILE__, __LINE__, #expr);

class CLGLContext {
    friend class CLVAContext;
    friend class NanoVGContext;
    friend class Window;

    cv::UMat frameBuffer_;
    cv::ogl::Texture2D *frameBufferTex_;
    GLuint frameBufferID;
    GLuint renderBufferID;
    CLExecContext_t context_;
    cv::Size windowSize_;
    cv::Size frameBufferSize_;
public:
    CLGLContext(cv::Size windowSize, cv::Size frameBufferSize) :
            windowSize_(windowSize), frameBufferSize_(frameBufferSize) {
        glewExperimental = true;
        glewInit();
        cv::ogl::ocl::initializeContextFromGL();
        frameBufferID = 0;
        GL_CHECK(glGenFramebuffers(1, &frameBufferID));
        GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frameBufferID));
        GL_CHECK(glGenRenderbuffers(1, &renderBufferID));

        frameBufferTex_ = new cv::ogl::Texture2D(frameBufferSize_, cv::ogl::Texture2D::RGBA, false);
        frameBufferTex_->bind();

        GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, renderBufferID));
        GL_CHECK(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, windowSize.width, windowSize.height));
        GL_CHECK(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderBufferID));

        GL_CHECK(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, frameBufferTex_->texId(), 0));
        assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

        context_ = CLExecContext_t::getCurrentRef();
    }

    cv::ogl::Texture2D& getFrameBufferTexture() {
        return *frameBufferTex_;
    }

    cv::Size getSize() {
        return frameBufferSize_;
    }

    void render(std::function<void(cv::Size&)> fn) {
        CLExecScope_t scope(context_);
        begin();
        fn(frameBufferSize_);
        end();
    }


    void compute(std::function<void(cv::UMat&)> fn) {
        CLExecScope_t scope(getCLExecContext());
        acquireFromGL(frameBuffer_);
        fn(frameBuffer_);
        releaseToGL(frameBuffer_);
    }
private:
    cv::ogl::Texture2D& getTexture2D() {
        return *frameBufferTex_;
    }

    CLExecContext_t& getCLExecContext() {
        return context_;
    }

    void blitFrameBufferToScreen(int x = 0, int y = 0) {
        GL_CHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, frameBufferID));
        GL_CHECK(glReadBuffer(GL_COLOR_ATTACHMENT0));
        GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
        GL_CHECK(glBlitFramebuffer(0, 0, frameBufferSize_.width, frameBufferSize_.height, x, y, x + frameBufferSize_.width, y + frameBufferSize_.height, GL_COLOR_BUFFER_BIT, GL_NEAREST));
    }

    void begin() {
        GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID));
        GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, renderBufferID));
        GL_CHECK(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderBufferID));
        frameBufferTex_->bind();
    }

    void end() {
        GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));
        GL_CHECK(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, 0));
        GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, 0));
        GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));

        //glFlush seems enough but i wanna make sure that there won't be race conditions.
        //At least on TigerLake/Iris it doesn't make a difference in performance.
        GL_CHECK(glFlush());
        GL_CHECK(glFinish());
    }

    void acquireFromGL(cv::UMat &m) {
        begin();
        GL_CHECK(cv::ogl::convertFromGLTexture2D(getTexture2D(), m));
        //FIXME
        cv::flip(m, m, 0);
    }

    void releaseToGL(cv::UMat &m) {
        //FIXME
        cv::flip(m, m, 0);
        GL_CHECK(cv::ogl::convertToGLTexture2D(m, getTexture2D()));
        end();
    }
};

class Window;
class CLVAContext {
    friend class Window;
    CLExecContext_t context_;
    CLGLContext &fbContext_;
    cv::UMat frameBuffer_;
    cv::UMat videoFrame_;
    bool hasContext_ = false;
public:
    CLVAContext(CLGLContext &fbContext) :
        fbContext_(fbContext) {
    }

    bool capture(std::function<void(cv::UMat&)> fn) {
        {
            CLExecScope_t scope(context_);
            fn(videoFrame_);
        }
        {
            CLExecScope_t scope(fbContext_.getCLExecContext());
            fbContext_.acquireFromGL(frameBuffer_);
            if (videoFrame_.empty())
                return false;

            cv::cvtColor(videoFrame_, frameBuffer_, cv::COLOR_RGB2BGRA);
            cv::Size fbSize = fbContext_.getSize();
            cv::resize(frameBuffer_, frameBuffer_, fbSize);
            fbContext_.releaseToGL(frameBuffer_);
            assert(frameBuffer_.size() == fbSize);
        }
        return true;
    }

    void write(std::function<void(const cv::UMat&)> fn) {
        cv::Size fbSize = fbContext_.getSize();
        {
            CLExecScope_t scope(fbContext_.getCLExecContext());
            fbContext_.acquireFromGL(frameBuffer_);
            cv::resize(frameBuffer_, frameBuffer_, fbSize);
            cv::cvtColor(frameBuffer_, videoFrame_, cv::COLOR_BGRA2RGB);
            fbContext_.releaseToGL(frameBuffer_);
        }
        assert(videoFrame_.size() == fbSize);
        {
            CLExecScope_t scope(context_);
            fn(videoFrame_);
        }
    }

private:
    bool hasContext() {
        return !context_.empty();
    }

    void copyContext() {
        context_ = CLExecContext_t::getCurrent();
    }

    CLExecContext_t getCLExecContext() {
        return context_;
    }
};
// class CLVAContext

class NanoVGContext {
    NVGcontext *context_;
    CLGLContext &fbContext_;
    float pixelRatio_;
public:
    NanoVGContext(NVGcontext *context, CLGLContext &fbContext, float pixelRatio) :
            context_(context), fbContext_(fbContext), pixelRatio_(pixelRatio) {
        nvgCreateFont(context_, "libertine", "assets/LinLibertine_RB.ttf");

        //FIXME workaround for first frame color glitch
        cv::UMat tmp;
        fbContext_.acquireFromGL(tmp);
        fbContext_.releaseToGL(tmp);
    }

    void render(std::function<void(NVGcontext*, const cv::Size&)> fn) {
        CLExecScope_t scope(fbContext_.getCLExecContext());
        begin();
        fn(context_, fbContext_.getSize());
        end();
    }
private:
    void begin() {
        fbContext_.begin();

        float r = pixelRatio_;
        float w = fbContext_.getSize().width;
        float h = fbContext_.getSize().height;
//    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, kb::gl::frame_buf));
        nvgSave (context_);
        nvgBeginFrame(context_, w, h, r);
    }

    void end() {
        nvgEndFrame (context_);
        nvgRestore(context_);
        fbContext_.end();
    }
};

static void error_callback(int error, const char *description) {
    fprintf(stderr, "GLFW Error: %s\n", description);
}

class Window {
    cv::Size size_;
    bool offscreen_;
    int major_;
    int minor_;
    int samples_;
    bool debug_;
    string title_;
    GLFWwindow *glfwWindow_ = nullptr;
    CLGLContext* clglContext_ = nullptr;
    CLVAContext* clvaContext_ = nullptr;
    NanoVGContext* nvgContext_ = nullptr;
    cv::VideoCapture* capture_ = nullptr;
    cv::VideoWriter* writer_ = nullptr;
    nanogui::Screen* screen_ = nullptr;
    nanogui::FormHelper* form_ = nullptr;
    cv::TickMeter tickMeter_;
    std::mutex pollMutex_;
    bool startPolling_ = false;
    bool closed_ = false;
public:

    Window(const cv::Size &size, bool offscreen, const string &title, int major = 4, int minor = 6, int samples = 0, bool debug = false) :
            size_(size), offscreen_(offscreen), title_(title), major_(major), minor_(minor), samples_(samples), debug_(debug) {
    }

    ~Window() {
        //don't delete form_. it is autmatically cleaned up by screen_
        if(screen_)
            delete screen_;
        if(writer_)
            delete writer_;
        if(capture_)
            delete capture_;
        if(nvgContext_)
            delete nvgContext_;
        if(clvaContext_)
            delete clvaContext_;
        if(clglContext_)
            delete clglContext_;
        glfwDestroyWindow(getGLFWWindow());
        glfwTerminate();
    }

    void initialize() {
        assert(glfwInit() == GLFW_TRUE);
        glfwSetErrorCallback(error_callback);

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
            Window *win = (Window*) glfwGetWindowUserPointer(glfwWin);
            win->screen_->cursor_pos_callback_event(x, y);
        }
        );
        glfwSetMouseButtonCallback(getGLFWWindow(), [](GLFWwindow *glfwWin, int button, int action, int modifiers) {
            Window *win = (Window*) glfwGetWindowUserPointer(glfwWin);
            win->screen_->mouse_button_callback_event(button, action, modifiers);
        }
        );
        glfwSetKeyCallback(getGLFWWindow(), [](GLFWwindow *glfwWin, int key, int scancode, int action, int mods) {
            Window *win = (Window*) glfwGetWindowUserPointer(glfwWin);
            win->screen_->key_callback_event(key, scancode, action, mods);
        }
        );
        glfwSetCharCallback(getGLFWWindow(), [](GLFWwindow *glfwWin, unsigned int codepoint) {
            Window *win = (Window*) glfwGetWindowUserPointer(glfwWin);
            win->screen_->char_callback_event(codepoint);
        }
        );
        glfwSetDropCallback(getGLFWWindow(), [](GLFWwindow *glfwWin, int count, const char **filenames) {
            Window *win = (Window*) glfwGetWindowUserPointer(glfwWin);
            win->screen_->drop_callback_event(count, filenames);
        }
        );
        glfwSetScrollCallback(getGLFWWindow(), [](GLFWwindow *glfwWin, double x, double y) {
            Window *win = (Window*) glfwGetWindowUserPointer(glfwWin);
            win->screen_->scroll_callback_event(x, y);
        }
        );
        glfwSetFramebufferSizeCallback(getGLFWWindow(), [](GLFWwindow *glfwWin, int width, int height) {
            Window *win = (Window*) glfwGetWindowUserPointer(glfwWin);
            win->screen_->resize_callback_event(width, height);
        }
        );
        clglContext_ = new CLGLContext(getSize(), getSize());
        clvaContext_ = new CLVAContext(*clglContext_);
        nvgContext_ = new NanoVGContext(getNVGcontext(), *clglContext_, getPixelRatio());
    }
    nanogui::FormHelper* form() {
        return form_;
    }

    CLGLContext& clgl() {
        return *clglContext_;
    }

    CLVAContext& clva() {
        return *clvaContext_;
    }

    NanoVGContext& nvg() {
        return *nvgContext_;
    }

    cv::TickMeter& getTickMeter() {
        return tickMeter_;
    }

    void render(std::function<void(const cv::Size&)> fn) {
        clgl().render(fn);
    }

    void compute(std::function<void(cv::UMat&)> fn) {
        clgl().compute(fn);
    }

    void renderNVG(std::function<void(NVGcontext*, const cv::Size&)> fn) {
        nvg().render(fn);
    }

    bool captureVA() {
        return clva().capture([=,this](cv::UMat& videoFrame){
            *(this->capture_) >> videoFrame;
        });
    }

    void writeVA() {
        clva().write([=,this](const cv::UMat& videoFrame){
            *(this->writer_) << videoFrame;
        });
    }

    void makeGLFWContextCurrent() {
        glfwMakeContextCurrent(getGLFWWindow());
    }

    cv::VideoWriter& makeVAWriter(const string& outputFilename, const int fourcc, const float fps, const cv::Size& frameSize, const int vaDeviceIndex) {
        writer_ = new cv::VideoWriter(outputFilename, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, frameSize, {
                cv::VIDEOWRITER_PROP_HW_DEVICE, vaDeviceIndex,
                cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
                cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1
        });

        if(!clva().hasContext()) {
            clva().copyContext();
        }
        return *writer_;
    }

    cv::VideoCapture& makeVACapture(const string& intputFilename, const int vaDeviceIndex) {
        //Initialize MJPEG HW decoding using VAAPI
        capture_ = new cv::VideoCapture(intputFilename, cv::CAP_FFMPEG, {
                cv::CAP_PROP_HW_DEVICE, vaDeviceIndex,
                cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
                cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL, 1
        });

        if(!clva().hasContext()) {
            clva().copyContext();
        }

        return *capture_;
    }

    void clear(const cv::Scalar& rgba = cv::Scalar(0,0,0,255)) {
        const float &r = rgba[0] / 255.0f;
        const float &g = rgba[1] / 255.0f;
        const float &b = rgba[2] / 255.0f;
        const float &a = rgba[3] / 255.0f;
        GL_CHECK(glClearColor(r, g, b, a));
        GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
    }

    cv::Size getFrameBufferSize() {
        int fbW, fbH;
        glfwGetFramebufferSize(getGLFWWindow(), &fbW, &fbH);
        return {fbW, fbH};
    }

    cv::Size getSize() {
        return size_;
    }

    float getPixelRatio() {
#if defined(EMSCRIPTEN)
        return emscripten_get_device_pixel_ratio();
#else
        float xscale, yscale;
        glfwGetWindowContentScale(getGLFWWindow(), &xscale, &yscale);
        return xscale;
#endif
    }

    void setSize(const cv::Size& sz) {
        screen_->set_size(nanogui::Vector2i(sz.width / getPixelRatio(), sz.height / getPixelRatio()));
        glfwSetWindowSize(getGLFWWindow(), sz.width, sz.height);
    }

    bool isFullscreen() {
        return glfwGetWindowMonitor(getGLFWWindow()) != nullptr;
    }

    void setFullscreen(bool f) {
        auto monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode *mode = glfwGetVideoMode(monitor);
        if (f) {
            glfwSetWindowMonitor(getGLFWWindow(), monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
        } else {
            glfwSetWindowMonitor(getGLFWWindow(), nullptr, 0, 0, getSize().width, getSize().width, mode->refreshRate);
        }
        setSize(getSize());
    }

    bool isResizable() {
        return glfwGetWindowAttrib(getGLFWWindow(), GLFW_RESIZABLE) == GLFW_TRUE;
    }

    void setResizable(bool r) {
        glfwWindowHint(GLFW_RESIZABLE, r ? GLFW_TRUE : GLFW_FALSE);
    }

    bool isVisible() {
        return glfwGetWindowAttrib(getGLFWWindow(), GLFW_VISIBLE) == GLFW_TRUE;
    }

    void setVisible(bool v) {
        screen_->set_visible(v);
        screen_->perform_layout();

        glfwWindowHint(GLFW_VISIBLE, v ? GLFW_TRUE : GLFW_FALSE);
        setSize(getSize());
    }

    bool isOffscreen() {
        return offscreen_;
    }

    void setOffscreen(bool o) {
        offscreen_ = o;
    }

    nanogui::Window* makeWindow(int x, int y, const string& title) {
        return form()->add_window(nanogui::Vector2i(x, y), title);
    }

    nanogui::Label* makeGroup(const string& label) {
        return form()->add_group(label);
    }

    nanogui::detail::FormWidget<bool>* makeFormVariable(const string &name, bool &v, const string &tooltip = "") {
        auto var = form()->add_variable(name, v);
        if (!tooltip.empty())
            var->set_tooltip(tooltip);
        return var;
    }

    template<typename T> nanogui::detail::FormWidget<T>* makeFormVariable(const string &name, T &v, const T &min, const T &max, bool spinnable = true, const string &unit = "", const string tooltip = "") {
        auto var = form()->add_variable(name, v);
        var->set_spinnable(spinnable);
        var->set_min_value(min);
        var->set_max_value(max);
        if (!unit.empty())
            var->set_units(unit);
        if (!tooltip.empty())
            var->set_tooltip(tooltip);
        return var;
    }

    void setUseOpenCL(bool u) {
        tickMeter_.reset();
        clglContext_->getCLExecContext().setUseOpenCL(u);
        clvaContext_->getCLExecContext().setUseOpenCL(u);
        cv::ocl::setUseOpenCL(u);
    }

    bool display() {
        std::scoped_lock<std::mutex> lock(pollMutex_);
        bool result = true;
        if (!offscreen_) {
            screen_->draw_contents();
            clglContext_->blitFrameBufferToScreen();
            screen_->draw_widgets();
            glfwSwapBuffers(glfwWindow_);
            result = !glfwWindowShouldClose(glfwWindow_);
        }

        startPolling_ = true;
        return result;
    }

    void pollEvents() {
        std::scoped_lock<std::mutex> lock(pollMutex_);
        if(startPolling_)
            glfwPollEvents();
    }

    bool isClosed() {
        return closed_;

    }
    void close() {
        setVisible(false);
        closed_ = true;
    }
private:
    GLFWwindow* getGLFWWindow() {
        return glfwWindow_;
    }

    NVGcontext* getNVGcontext() {
        return screen_->nvg_context();
    }
};
// class Window

static std::string get_gl_info() {
    return reinterpret_cast<const char*>(glGetString(GL_VERSION));
}

static std::string get_cl_info() {
    std::stringstream ss;
    std::vector<cv::ocl::PlatformInfo> plt_info;
    cv::ocl::getPlatfomsInfo(plt_info);
    const cv::ocl::Device &defaultDevice = cv::ocl::Device::getDefault();
    cv::ocl::Device current;
    ss << endl;
    for (const auto &info : plt_info) {
        for (int i = 0; i < info.deviceNumber(); ++i) {
            ss << "\t";
            info.getDevice(current, i);
            if (defaultDevice.name() == current.name())
                ss << "* ";
            else
                ss << "  ";
            ss << info.version() << " = " << info.name() << endl;
            ss << "\t\t  GL sharing: " << (current.isExtensionSupported("cl_khr_gl_sharing") ? "true" : "false") << endl;
            ss << "\t\t  VAAPI media sharing: " << (current.isExtensionSupported("cl_intel_va_api_media_sharing") ? "true" : "false") << endl;
        }
    }

    return ss.str();
}

static void print_system_info() {
    cerr << "OpenGL Version: " << get_gl_info() << endl;
    cerr << "OpenCL Platforms: " << get_cl_info() << endl;
}

static void update_fps(cv::Ptr<Window> window, bool graphical = false) {
        static uint64_t cnt = 0;
        float fps;
        if (cnt > 0) {
            window->getTickMeter().stop();

            if (window->getTickMeter().getTimeMilli() > 1000) {
                cerr << "FPS : " << (fps = window->getTickMeter().getFPS()) << '\r';
                if (graphical) {
                    window->renderNVG([&](NVGcontext *vg, const cv::Size &size) {
                        string text = "FPS: " + std::to_string(fps);
                        nvgBeginPath(vg);
                        nvgRoundedRect(vg, 10, 10, 30 * text.size() + 10, 60, 10);
                        nvgFillColor(vg, nvgRGBA(255, 255, 255, 180));
                        nvgFill(vg);

                        nvgBeginPath(vg);
                        nvgFontSize(vg, 60.0f);
                        nvgFontFace(vg, "mono");
                        nvgFillColor(vg, nvgRGBA(90, 90, 90, 255));
                        nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
                        nvgText(vg, 22, 37, text.c_str(), nullptr);
                    });
                }
                cnt = 0;
            }
        }

        window->getTickMeter().start();
        ++cnt;
    }
} //namespace kb

#endif /* SRC_SUBSYSTEMS_HPP_ */
