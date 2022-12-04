#ifndef SRC_SUBSYSTEMS_HPP_
#define SRC_SUBSYSTEMS_HPP_

#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "opencv2/core/va_intel.hpp"
#include <opencv2/videoio.hpp>
#include <GL/glew.h>

#if !defined(__APPLE__)
#  include <X11/Xlib.h>
#  include <X11/Xatom.h>
#  include <X11/Xutil.h>
#  include <X11/extensions/Xrandr.h>
#  include <EGL/egl.h>
#  include <EGL/eglext.h>
#endif

#if !defined(_GCV_ONLY_X11)
#  define GLFW_INCLUDE_NONE
#  include <GLFW/glfw3.h>
#endif

#include <GL/gl.h>
#include "nanovg.h"
#define NANOVG_GL3_IMPLEMENTATION
#include "nanovg_gl.h"
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/opengl.hpp>

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

#ifdef _GCV_ONLY_X11
//adapted from https://github.com/glfw/glfw/blob/master/src/egl_context.c
const char* get_egl_error_string(EGLint error)
{
    switch (error)
    {
        case EGL_SUCCESS:
            return "Success";
        case EGL_NOT_INITIALIZED:
            return "EGL is not or could not be initialized";
        case EGL_BAD_ACCESS:
            return "EGL cannot access a requested resource";
        case EGL_BAD_ALLOC:
            return "EGL failed to allocate resources for the requested operation";
        case EGL_BAD_ATTRIBUTE:
            return "An unrecognized attribute or attribute value was passed in the attribute list";
        case EGL_BAD_CONTEXT:
            return "An EGLContext argument does not name a valid EGL rendering context";
        case EGL_BAD_CONFIG:
            return "An EGLConfig argument does not name a valid EGL frame buffer configuration";
        case EGL_BAD_CURRENT_SURFACE:
            return "The current surface of the calling thread is a window, pixel buffer or pixmap that is no longer valid";
        case EGL_BAD_DISPLAY:
            return "An EGLDisplay argument does not name a valid EGL display connection";
        case EGL_BAD_SURFACE:
            return "An EGLSurface argument does not name a valid surface configured for GL rendering";
        case EGL_BAD_MATCH:
            return "Arguments are inconsistent";
        case EGL_BAD_PARAMETER:
            return "One or more argument values are invalid";
        case EGL_BAD_NATIVE_PIXMAP:
            return "A NativePixmapType argument does not refer to a valid native pixmap";
        case EGL_BAD_NATIVE_WINDOW:
            return "A NativeWindowType argument does not refer to a valid native window";
        case EGL_CONTEXT_LOST:
            return "The application must destroy all contexts and reinitialise";
        default:
            return "ERROR: UNKNOWN EGL ERROR";
    }
}
void egl_check_error(const std::filesystem::path &file, unsigned int line, const char *expression) {
    EGLint errorCode = eglGetError();

    if (errorCode != EGL_SUCCESS) {
        cerr << "EGL failed in " << file.filename() << " (" << line << ") : " << "\nExpression:\n   " << expression << "\nError code:\n   " << get_egl_error_string(errorCode) << "\n   " << endl;
        assert(false);
    }
}
#define EGL_CHECK(expr)                                 \
        expr;                                          \
        kb::egl_check_error(__FILE__, __LINE__, #expr);
#endif

namespace app {
unsigned int window_width;
unsigned int window_height;
bool offscreen;
} //app

namespace va {
cv::ocl::OpenCLExecutionContext context;

void copy() {
    va::context = cv::ocl::OpenCLExecutionContext::getCurrent();
}

void bind() {
    va::context.bind();
}
} // namespace va

#ifdef _GCV_ONLY_X11
namespace x11 {
Display *xdisplay;
Window xroot;
Window xwin;
Atom wmDeleteMessage;

bool initialized = false;

std::pair<unsigned int, unsigned int> get_window_size() {
    std::pair<unsigned int, unsigned int> ret;
    int x, y;
    unsigned int border, depth;
    XGetGeometry(xdisplay, xwin, &xroot, &x, &y, &ret.first, &ret.second, &border, &depth);
    return ret;
}

bool window_closed() {
    if (XPending(xdisplay) == 0)
        return false;

    XEvent event;
    XNextEvent(xdisplay, &event);

    switch (event.type) {
    case ClientMessage:
        if (event.xclient.data.l[0] == static_cast<long int>(wmDeleteMessage))
            return true;
        break;

    default:
        break;
    }
    return false;
}

void init(const std::string& title) {
    xdisplay = XOpenDisplay(nullptr);
    if (xdisplay == nullptr) {
        cerr << "Unable to open X11 display" << endl;
        exit(3);
    }

    xroot = DefaultRootWindow(xdisplay);
    XSetWindowAttributes swa;
    swa.event_mask = ClientMessage;
    xwin = XCreateWindow(xdisplay, xroot, 0, 0, app::window_width, app::window_height, 0,
    CopyFromParent, InputOutput, CopyFromParent, CWEventMask, &swa);

    XSetWindowAttributes xattr;
    xattr.override_redirect = False;
    XChangeWindowAttributes(xdisplay, xwin, CWOverrideRedirect, &xattr);

    int one = 1;
    XChangeProperty(xdisplay, xwin, XInternAtom(xdisplay, "_HILDON_NON_COMPOSITED_WINDOW", False),
    XA_INTEGER, 32, PropModeReplace, (unsigned char*) &one, 1);

    XWMHints hints;
    hints.input = True;
    hints.flags = InputHint;
    XSetWMHints(xdisplay, xwin, &hints);

    XMapWindow(xdisplay, xwin);
    XStoreName(xdisplay, xwin, title.c_str());
    wmDeleteMessage = XInternAtom(xdisplay, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(xdisplay, xwin, &wmDeleteMessage, 1);

    initialized = true;
}
} // namespace x11
#else

namespace glfw {
GLFWwindow *window;
bool initialized = false;
int framebuffer_width;
int framebuffer_height;

void processInput(GLFWwindow *window){
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
    framebuffer_width = width;
    framebuffer_height = height;
    glViewport(0, 0, width, height);
}

void error_callback(int error, const char *description) {
    fprintf(stderr, "Error: %s\n", description);
}

void init(const string &title, int major = 4, int minor = 6, int samples = 4, bool debug = false) {
    assert(glfwInit() == GLFW_TRUE);
    framebuffer_width = app::window_width;
    framebuffer_height = app::window_height;
    glfwSetErrorCallback(error_callback);

    if(debug)
        glfwWindowHint (GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);

#ifdef __APPLE__
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
#endif
    glfwWindowHint(GLFW_SAMPLES, samples);

    if (app::offscreen) {
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    }

    window = glfwCreateWindow(app::window_width, app::window_height, title.c_str(), nullptr, nullptr);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(11);
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    initialized = true;
}

void terminate() {
    initialized = false;
    glfwDestroyWindow(window);
    glfwTerminate();
}
} // namespace glfw

#endif

#ifdef _GCV_ONLY_X11
namespace egl {
//code in the kb::egl namespace deals with setting up EGL
EGLDisplay display;
EGLSurface surface;
EGLContext context;

void debugMessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *msg, const void *data) {
    std::string _source;
    std::string _type;
    std::string _severity;

    switch (source) {
    case GL_DEBUG_SOURCE_API:
        _source = "API";
        break;

    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
        _source = "WINDOW SYSTEM";
        break;

    case GL_DEBUG_SOURCE_SHADER_COMPILER:
        _source = "SHADER COMPILER";
        break;

    case GL_DEBUG_SOURCE_THIRD_PARTY:
        _source = "THIRD PARTY";
        break;

    case GL_DEBUG_SOURCE_APPLICATION:
        _source = "APPLICATION";
        break;

    case GL_DEBUG_SOURCE_OTHER:
        _source = "UNKNOWN";
        break;

    default:
        _source = "UNKNOWN";
        break;
    }

    switch (type) {
    case GL_DEBUG_TYPE_ERROR:
        _type = "ERROR";
        break;

    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        _type = "DEPRECATED BEHAVIOR";
        break;

    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        _type = "UDEFINED BEHAVIOR";
        break;

    case GL_DEBUG_TYPE_PORTABILITY:
        _type = "PORTABILITY";
        break;

    case GL_DEBUG_TYPE_PERFORMANCE:
        _type = "PERFORMANCE";
        break;

    case GL_DEBUG_TYPE_OTHER:
        _type = "OTHER";
        break;

    case GL_DEBUG_TYPE_MARKER:
        _type = "MARKER";
        break;

    default:
        _type = "UNKNOWN";
        break;
    }

    switch (severity) {
    case GL_DEBUG_SEVERITY_HIGH:
        _severity = "HIGH";
        break;

    case GL_DEBUG_SEVERITY_MEDIUM:
        _severity = "MEDIUM";
        break;

    case GL_DEBUG_SEVERITY_LOW:
        _severity = "LOW";
        break;

    case GL_DEBUG_SEVERITY_NOTIFICATION:
        _severity = "NOTIFICATION";
        break;

    default:
        _severity = "UNKNOWN";
        break;
    }

    fprintf(stderr, "%d: %s of %s severity, raised from %s: %s\n", id, _type.c_str(), _severity.c_str(), _source.c_str(), msg);

    if (type == GL_DEBUG_TYPE_ERROR)
        exit(2);
}

EGLBoolean swap_buffers() {
    return EGL_CHECK(eglSwapBuffers(display, surface));
}

void init(int major = 4, int minor = 6, int samples = 4, bool debug = false) {
    EGL_CHECK(eglBindAPI(EGL_OPENGL_API));
    if (app::offscreen) {
        EGL_CHECK(display = eglGetDisplay(EGL_DEFAULT_DISPLAY));
    } else {
#ifdef _GCV_ONLY_X11
        EGL_CHECK(display = eglGetDisplay(x11::xdisplay));
#endif
    }
    EGL_CHECK(eglInitialize(display, nullptr, nullptr));

    const EGLint egl_config_constraints[] = {
    EGL_STENCIL_SIZE, static_cast<EGLint>(8),
    EGL_DEPTH_SIZE, static_cast<EGLint>(16),
    EGL_BUFFER_SIZE, static_cast<EGLint>(32),
    EGL_RED_SIZE, static_cast<EGLint>(8),
    EGL_GREEN_SIZE, static_cast<EGLint>(8),
    EGL_BLUE_SIZE, static_cast<EGLint>(8),
    EGL_ALPHA_SIZE, static_cast<EGLint>(8),
    EGL_SAMPLE_BUFFERS, EGL_TRUE,
    EGL_SAMPLES, samples,
    EGL_SURFACE_TYPE, EGL_WINDOW_BIT | EGL_PBUFFER_BIT,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
    EGL_CONFORMANT, EGL_OPENGL_BIT,
    EGL_CONFIG_CAVEAT, EGL_NONE,
    EGL_NONE };

    EGLint configCount;
    EGLConfig configs[1];
    EGL_CHECK(eglChooseConfig(display, egl_config_constraints, configs, 1, &configCount));

    EGLint stencilSize;
    eglGetConfigAttrib(display, configs[0],
    EGL_STENCIL_SIZE, &stencilSize);

    const EGLint contextVersion[] = {
    EGL_CONTEXT_MAJOR_VERSION, major,
    EGL_CONTEXT_MINOR_VERSION, minor,
    EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT,
    EGL_CONTEXT_OPENGL_DEBUG, debug ? EGL_TRUE : EGL_FALSE,
    EGL_NONE };
    EGL_CHECK(context = eglCreateContext(display, configs[0], EGL_NO_CONTEXT, contextVersion));

    if (!app::offscreen) {
#ifdef _GCV_ONLY_X11
        EGL_CHECK(surface = eglCreateWindowSurface(display, configs[0], x11::xwin, nullptr));
#endif
    } else {
        EGLint pbuffer_attrib_list[] = {
        EGL_WIDTH, int(app::window_width),
        EGL_HEIGHT, int(app::window_height),
        EGL_NONE };
        EGL_CHECK(surface = eglCreatePbufferSurface(display, configs[0], pbuffer_attrib_list));
    }


    EGL_CHECK(eglMakeCurrent(display, surface, surface, context));
    EGL_CHECK(eglSwapInterval(display, 1));

    if (debug) {
        GL_CHECK(glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS));
        auto glDebugMessageCallback = (void (*)(void*, void*)) eglGetProcAddress("glDebugMessageCallback");
        assert(glDebugMessageCallback);
        GL_CHECK(glDebugMessageCallback(reinterpret_cast<void*>(debugMessageCallback), nullptr));
    }
}

std::string get_info() {
    return eglQueryString(display, EGL_VERSION);
}

} //namespace egl
#endif
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

    GL_CHECK(glFlush());
    GL_CHECK(glFinish());
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

void blit_frame_buffer_to_screen() {
    GL_CHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, kb::gl::frame_buf));
    GL_CHECK(glReadBuffer(GL_COLOR_ATTACHMENT0));
    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
    GL_CHECK(glBlitFramebuffer(0, 0, app::window_width, app::window_height, 0, 0, app::window_width, app::window_height, GL_COLOR_BUFFER_BIT, GL_NEAREST));
}
} // namespace gl

namespace cl {
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
#ifdef _GCV_ONLY_X11
    auto ws = x11::get_window_size();
    w = ws.first;
    h = ws.second;
#else
    w = glfw::framebuffer_width;
    h = glfw::framebuffer_height;
#endif
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

void init(bool debug = false) {
    GL_CHECK(glViewport(0, 0, app::window_width, app::window_height));
    GL_CHECK(glEnable(GL_STENCIL_TEST));
    GL_CHECK(glStencilMask(~0));
    GL_CHECK(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

    vg = nvgCreateGL3(NVG_STENCIL_STROKES | debug ? NVG_DEBUG : 0);
    if (vg == NULL) {
        cerr << "Couldn't init nanovg." << endl;
        exit(24);
    }

    nvgCreateFont(vg, "serif", "assets/LinLibertine_RB.ttf");

    //FIXME workaround for color glitch in first frame. I don't know why yet but acquiring and releasing the framebuffer fixes it.
    cv::UMat fb;
    gl::acquire_from_gl(fb);
    gl::release_to_gl(fb);
}
} //namespace nvg

namespace app {
void init(const string &windowTitle, unsigned int width, unsigned int height, bool offscreen = false, int major = 4, int minor = 6, int samples = 4, bool debugContext = false) {
    app::window_width = width;
    app::window_height = height;
    app::offscreen = offscreen;

#ifdef _GCV_ONLY_X11
    if (!offscreen) {
        x11::init(windowTitle);
        //you can set OpenGL-version, multisample-buffer samples and enable debug context using egl::init()
        egl::init(major, minor, samples, debugContext);
    }
#else
    glfw::init(windowTitle, major, minor, samples, debugContext);
#endif

    //Initialize OpenCL Context for OpenGL
    gl::init();
    nvg::init();
}

bool display() {
#ifdef _GCV_ONLY_X11
        if(x11::initialized) {
            //Blit the framebuffer we have been working on to the screen
            gl::blit_frame_buffer_to_screen();

            //Check if the x11 wl_window was closed
            if(x11::initialized && x11::window_closed())
                return false;

            //Transfer the back buffer (which we have been using as frame buffer) to the native wl_window
            egl::swap_buffers();
        }
#else
    if(glfw::initialized) {
        if(!app::offscreen) {
            gl::blit_frame_buffer_to_screen();
            glfw::processInput(glfw::window);
            glfwSwapBuffers(glfw::window);
            glfwPollEvents();
            return !glfwWindowShouldClose(glfw::window);
        } else {
            glfwPollEvents();
        }
    }
#endif
    return true;
}

void print_system_info() {
#ifdef _GCV_ONLY_X11
        cerr << "EGL Version: " << egl::get_info() << endl;
#endif
    cerr << "OpenGL Version: " << gl::get_info() << endl;
    cerr << "OpenCL Platforms: " << cl::get_info() << endl;
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
#ifdef _GCV_ONLY_X11
    //TODO properly cleanup X11 and EGL
#else
    glfwTerminate();
#endif
}
} //namespace app
} //namespace kb

#endif /* SRC_SUBSYSTEMS_HPP_ */
