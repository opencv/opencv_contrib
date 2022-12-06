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

namespace app {
unsigned int window_width;
unsigned int window_height;
bool offscreen;
} //app

namespace display {
GLFWwindow *window;

static float get_pixel_ratio(GLFWwindow *window) {
#if defined(_WIN32)
    HWND hWnd = glfwGetWin32Window(window);
    HMONITOR monitor = MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST);
    /* The following function only exists on Windows 8.1+, but we don't want to make that a dependency */
    static HRESULT (WINAPI *GetDpiForMonitor_)(HMONITOR, UINT, UINT*, UINT*) = nullptr;
    static bool GetDpiForMonitor_tried = false;

    if (!GetDpiForMonitor_tried) {
        auto shcore = LoadLibrary(TEXT("shcore"));
        if (shcore)
            GetDpiForMonitor_ = (decltype(GetDpiForMonitor_)) GetProcAddress(shcore, "GetDpiForMonitor");
        GetDpiForMonitor_tried = true;
    }

    if (GetDpiForMonitor_) {
        uint32_t dpiX, dpiY;
        if (GetDpiForMonitor_(monitor, 0 /* effective DPI */, &dpiX, &dpiY) == S_OK)
            return std::round(dpiX / 96.0);
    }
    return 1.f;
#else
    int fbW, fbH;
    int w, h;
    glfwGetFramebufferSize(window, &fbW, &fbH);
    glfwGetWindowSize(window, &w, &h);
    return (float)fbW / (float)w;
#endif
}
void update_size(float pixelRatio = display::get_pixel_ratio(display::window)) {
    glfwSetWindowSize(display::window, app::window_width * pixelRatio, app::window_height * pixelRatio);
    glViewport(0, 0, app::window_width * pixelRatio, app::window_height * pixelRatio);
}

bool is_fullscreen() {
    return glfwGetWindowMonitor(display::window) != nullptr;
}
void set_fullscreen(bool f) {
    auto monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    if(f) {
        glfwSetWindowMonitor(display::window, monitor, 0,0, mode->width, mode->height, mode->refreshRate);
    } else {
        glfwSetWindowMonitor(display::window, nullptr, 0,0,app::window_width, app::window_height,mode->refreshRate);
    }
    display::update_size();
}

void framebuffer_size_callback(GLFWwindow *win, int width, int height) {
    display::update_size();
}

void error_callback(int error, const char *description) {
    fprintf(stderr, "Error: %s\n", description);
}

void init(const string &title, int major, int minor, int samples = 4, bool debug = false) {
    assert(glfwInit() == GLFW_TRUE);
    glfwSetErrorCallback(error_callback);

    if(debug)
        glfwWindowHint (GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);

    if(app::offscreen)
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    glfwSetTime(0);

#ifdef __APPLE__
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
#endif
    glfwWindowHint(GLFW_SAMPLES, samples);
    glfwWindowHint(GLFW_RED_BITS, 8);
    glfwWindowHint(GLFW_GREEN_BITS, 8);
    glfwWindowHint(GLFW_BLUE_BITS, 8);
    glfwWindowHint(GLFW_ALPHA_BITS, 8);
    glfwWindowHint(GLFW_STENCIL_BITS, 8);
    glfwWindowHint(GLFW_DEPTH_BITS, 24);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(app::window_width, app::window_height, title.c_str(), nullptr, nullptr);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(11);
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
}
} // namespace display

namespace gl {
//code in the kb::gl namespace deals with OpenGL (and OpenCV/GL) internals
cv::ogl::Texture2D *frame_buf_tex;
GLuint frame_buf;
GLuint render_buf;
cv::ocl::OpenCLExecutionContext context;

void bind() {
    gl::context.bind();
}

void begin() {
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, frame_buf));
    GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, render_buf));
    GL_CHECK(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, render_buf));
    frame_buf_tex->bind();
}

void end() {
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));
    GL_CHECK(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, 0));
    GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, 0));
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));

    //glFlush seems enough but i wanna make sure...
    GL_CHECK(glFlush());
    GL_CHECK(glFinish());
}

void render(std::function<void(int,int)> fn) {
    gl::bind();
    gl::begin();
    fn(app::window_width, app::window_height);
    gl::end();
}

void init() {
    glewExperimental = true;
    glewInit();

    cv::ogl::ocl::initializeContextFromGL();

    frame_buf = 0;
    GL_CHECK(glGenFramebuffers(1, &frame_buf));
    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frame_buf));
    GL_CHECK(glGenRenderbuffers(1, &render_buf));
    GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, render_buf));
    GL_CHECK(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, app::window_width, app::window_height));

    GL_CHECK(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, render_buf));
    frame_buf_tex = new cv::ogl::Texture2D(cv::Size(app::window_width, app::window_height), cv::ogl::Texture2D::RGBA, false);
    frame_buf_tex->bind();

    GL_CHECK(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, frame_buf_tex->texId(), 0));

    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

    gl::context = cv::ocl::OpenCLExecutionContext::getCurrent();
}

std::string get_info() {
    return reinterpret_cast<const char*>(glGetString(GL_VERSION));
}

void blit_frame_buffer_to_screen() {
    GL_CHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, kb::gl::frame_buf));
    GL_CHECK(glReadBuffer(GL_COLOR_ATTACHMENT0));
    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
    GL_CHECK(glBlitFramebuffer(0, 0, app::window_width, app::window_height, 0, 0, app::window_width, app::window_height, GL_COLOR_BUFFER_BIT, GL_NEAREST));
}
} // namespace gl

namespace cl {
cv::UMat frameBuffer;

void acquire_from_gl(cv::UMat& m) {
    gl::begin();
    GL_CHECK(cv::ogl::convertFromGLTexture2D(*gl::frame_buf_tex, m));
    //The OpenGL frameBuffer is upside-down. Flip it. (OpenCL)
    cv::flip(m, m, 0);
}

void release_to_gl(cv::UMat& m) {
    //The OpenGL frameBuffer is upside-down. Flip it back. (OpenCL)
    cv::flip(m, m, 0);
    GL_CHECK(cv::ogl::convertToGLTexture2D(m, *gl::frame_buf_tex));
    gl::end();
}

void compute(std::function<void(cv::UMat& m)> fn) {
    gl::bind();
    acquire_from_gl(frameBuffer);
    fn(frameBuffer);
    release_to_gl(frameBuffer);
}

std::string get_info() {
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
} //namespace cl

namespace nvg {
NVGcontext *vg;

void clear(const float& r = 0.0f, const float& g = 0.0f, const float& b = 0.0f, const float& a = 1.0f) {
    GL_CHECK(glClearColor(r, g, b, a));
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
}

void begin() {
    gl::begin();

    float w;
    float h;
    w = app::window_width;
    h = app::window_height;
    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, kb::gl::frame_buf));
    nvgSave(vg);
    GL_CHECK(glViewport(0, 0,w, h));
    nvgBeginFrame(vg, w, h, std::fmax(app::window_width/w, app::window_height/h));
}

void end() {
    nvgEndFrame(vg);
    nvgRestore(vg);
    gl::end();
}

void render(std::function<void(int,int)> fn) {
    gl::bind();
    nvg::begin();
    fn(app::window_width, app::window_height);
    nvg::end();
}

void init(NVGcontext *context) {
    if (context == nullptr) {
        cerr << "Couldn't init nanovg." << endl;
        exit(24);
    }
    vg = context;
    nvgCreateFont(vg, "serif", "assets/LinLibertine_RB.ttf");

    //FIXME workaround for color glitch in first frame. I don't know why yet but acquiring and releasing the framebuffer fixes it.
    gl::bind();
    cv::UMat fb;
    cl::acquire_from_gl(fb);
    cl::release_to_gl(fb);
}
} //namespace nvg
namespace va {
cv::ocl::OpenCLExecutionContext context;
cv::UMat videoFrame;

void copy() {
    va::context = cv::ocl::OpenCLExecutionContext::getCurrent();
}

void bind() {
    va::context.bind();
}

bool read(std::function<void(cv::UMat&)> fn) {
    va::bind();
    fn(va::videoFrame);
    gl::bind();
    cl::acquire_from_gl(cl::frameBuffer);
    if(va::videoFrame.empty())
        return false;
    //Color-conversion from RGB to BGRA (OpenCL)
    cv::cvtColor(va::videoFrame, cl::frameBuffer, cv::COLOR_RGB2BGRA);
    cl::release_to_gl(cl::frameBuffer);
    return true;
}

void write(std::function<void(const cv::UMat&)> fn) {
    va::bind();
    //Color-conversion from BGRA to RGB. (OpenCL)
    cv::cvtColor(cl::frameBuffer, va::videoFrame, cv::COLOR_BGRA2RGB);
    cv::flip(va::videoFrame, va::videoFrame, 0);
    fn(va::videoFrame);
}
} // namespace va

namespace gui {
using namespace nanogui;
ref<nanogui::Screen> screen;
FormHelper* form;

template <typename T> nanogui::detail::FormWidget<T> * make_gui_variable(const string& name, T& v, const T& min, const T& max, bool spinnable = true, const string& unit = "", const string tooltip = "") {
    using kb::gui::form;
    auto var = form->add_variable(name, v);
    var->set_spinnable(spinnable);
    var->set_min_value(min);
    var->set_max_value(max);
    if(!unit.empty())
        var->set_units(unit);
    if(!tooltip.empty())
        var->set_tooltip(tooltip);
    return var;
}

void init(int w, int h) {
    screen = new nanogui::Screen();
    screen->initialize(display::window, false);
    screen->set_size(nanogui::Vector2i(w, h));
    form = new FormHelper(screen);

    glfwSetCursorPosCallback(display::window,
            [](GLFWwindow *, double x, double y) {
        gui::screen->cursor_pos_callback_event(x, y);
        }
    );
    glfwSetMouseButtonCallback(display::window,
        [](GLFWwindow *, int button, int action, int modifiers) {
        gui::screen->mouse_button_callback_event(button, action, modifiers);
        }
    );
    glfwSetKeyCallback(display::window,
        [](GLFWwindow *, int key, int scancode, int action, int mods) {
        gui::screen->key_callback_event(key, scancode, action, mods);
        }
    );
    glfwSetCharCallback(display::window,
        [](GLFWwindow *, unsigned int codepoint) {
        gui::screen->char_callback_event(codepoint);
        }
    );
    glfwSetDropCallback(display::window,
        [](GLFWwindow *, int count, const char **filenames) {
        gui::screen->drop_callback_event(count, filenames);
        }
    );
    glfwSetScrollCallback(display::window,
        [](GLFWwindow *, double x, double y) {
        gui::screen->scroll_callback_event(x, y);
       }
    );
    glfwSetFramebufferSizeCallback(display::window,
        [](GLFWwindow *, int width, int height) {
            gui::screen->resize_callback_event(width, height);
        }
    );
}

void set_visible(bool v) {
    gui::screen->set_visible(v);
    gui::screen->perform_layout();
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
}

void update_size(float pixelRatio = display::get_pixel_ratio(display::window)) {
    gui::screen->set_size(nanogui::Vector2i(app::window_width * pixelRatio, app::window_height * pixelRatio));
    display::update_size(pixelRatio);
}
} //namespace gui

namespace app {
void print_system_info() {
    cerr << "OpenGL Version: " << gl::get_info() << endl;
    cerr << "OpenCL Platforms: " << cl::get_info() << endl;
}

void init(const string &windowTitle, unsigned int width, unsigned int height, bool offscreen = false, bool fullscreen = false, int major = 4, int minor = 6, int samples = 4, bool debugContext = false) {
    using namespace kb::gui;
    app::window_width = width;
    app::window_height = height;
    app::offscreen = offscreen;

    display::init(windowTitle, samples, debugContext);
    gui::init(width, height);
    gl::init();
    nvg::init(screen->nvg_context());
}


void run(std::function<void()> fn) {
    if(!app::offscreen)
        gui::set_visible(true);
    gui::update_size();

    fn();
}

bool display() {
    if(!app::offscreen) {
        glfwPollEvents();
        gui::screen->draw_contents();
        gl::blit_frame_buffer_to_screen();
        gui::screen->draw_widgets();
        glfwSwapBuffers(display::window);
        return !glfwWindowShouldClose(display::window);
    }
    return true;
}

void print_fps() {
    static uint64_t cnt = 0;
    static double fps = 1;
    static cv::TickMeter meter;

    if (cnt > 0) {
        meter.stop();

        if (cnt % uint64(ceil(fps)) == 0) {
            fps = meter.getFPS();
            cerr << "FPS : " << fps << '\r';
            cnt = 0;
        }
    }

    meter.start();
    ++cnt;
}

void terminate() {
    glfwDestroyWindow(display::window);
    glfwTerminate();
    exit(0);
}
} //namespace app
} //namespace kb

#endif /* SRC_SUBSYSTEMS_HPP_ */
