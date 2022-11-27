#ifndef SRC_SUBSYSTEMS_HPP_
#define SRC_SUBSYSTEMS_HPP_

#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <filesystem>
#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_backend.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core/va_intel.hpp"
#include <opencv2/videoio.hpp>
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <GL/glew.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
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

void egl_check_error(const std::filesystem::path &file, unsigned int line, const char *expression) {
    EGLint errorCode = eglGetError();

    if (errorCode != EGL_SUCCESS) {
        cerr << "EGL failed in " << file.filename() << " (" << line << ") : " << "\nExpression:\n   " << expression << "\nError code:\n   " << errorCode << "\n   " << endl;
        assert(false);
    }
}
#define EGL_CHECK(expr)                                 \
        expr;                                          \
        kb::egl_check_error(__FILE__, __LINE__, #expr);

namespace va {
cv::ocl::OpenCLExecutionContext context;

void init() {
    va::context = cv::ocl::OpenCLExecutionContext::getCurrent();
}

void bind() {
    va::context.bind();
}
} // namespace va

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

Display* get_x11_display() {
    return xdisplay;
}

Window get_x11_window() {
    return xwin;
}

bool is_initialized() {
    return initialized;
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

void init() {
    xdisplay = XOpenDisplay(nullptr);
    if (xdisplay == nullptr) {
        cerr << "Unable to open X11 display" << endl;
        exit(3);
    }

    xroot = DefaultRootWindow(xdisplay);
    XSetWindowAttributes swa;
    swa.event_mask = ClientMessage;
    xwin = XCreateWindow(xdisplay, xroot, 0, 0, WIDTH, HEIGHT, 0,
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
    XStoreName(xdisplay, xwin, "nanovg-demo");
    wmDeleteMessage = XInternAtom(xdisplay, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(xdisplay, xwin, &wmDeleteMessage, 1);

    initialized = true;
}
} // namespace x11

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

void init(bool debug = false) {
    bool offscreen = !x11::is_initialized();

    EGL_CHECK(eglBindAPI(EGL_OPENGL_API));
    if (offscreen) {
        EGL_CHECK(display = eglGetDisplay(EGL_DEFAULT_DISPLAY));
    } else {
        EGL_CHECK(display = eglGetDisplay(x11::get_x11_display()));
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
    EGL_SAMPLES, 16,
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

    if (!offscreen) {
        EGL_CHECK(surface = eglCreateWindowSurface(display, configs[0], x11::get_x11_window(), nullptr));
    } else {
        EGLint pbuffer_attrib_list[] = {
        EGL_WIDTH, WIDTH,
        EGL_HEIGHT, HEIGHT,
        EGL_NONE };
        EGL_CHECK(surface = eglCreatePbufferSurface(display, configs[0], pbuffer_attrib_list));
    }

    const EGLint contextVersion[] = {
    EGL_CONTEXT_MAJOR_VERSION, 4,
    EGL_CONTEXT_MINOR_VERSION, 6,
    EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT,
    EGL_CONTEXT_OPENGL_DEBUG, debug ? EGL_TRUE : EGL_FALSE,
    EGL_NONE };
    EGL_CHECK(context = eglCreateContext(display, configs[0], EGL_NO_CONTEXT, contextVersion));
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

void push() {
    GL_CHECK(glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS));
    GL_CHECK(glPushAttrib(GL_ALL_ATTRIB_BITS));
    GL_CHECK(glMatrixMode(GL_MODELVIEW));
    GL_CHECK(glPushMatrix());
    GL_CHECK(glMatrixMode(GL_PROJECTION));
    GL_CHECK(glPushMatrix());
    GL_CHECK(glMatrixMode(GL_TEXTURE));
    GL_CHECK(glPushMatrix());
}

void pop() {
    GL_CHECK(glMatrixMode(GL_TEXTURE));
    GL_CHECK(glPopMatrix());
    GL_CHECK(glMatrixMode(GL_PROJECTION));
    GL_CHECK(glPopMatrix());
    GL_CHECK(glMatrixMode(GL_MODELVIEW));
    GL_CHECK(glPopMatrix());
    GL_CHECK(glPopClientAttrib());
    GL_CHECK(glPopAttrib());
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
    GL_CHECK(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, WIDTH, HEIGHT));

    GL_CHECK(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, render_buf));

    frame_buf_tex = new cv::ogl::Texture2D(cv::Size(WIDTH, HEIGHT), cv::ogl::Texture2D::RGBA, false);
    frame_buf_tex->bind();
    GL_CHECK(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, frame_buf_tex->texId(), 0));

    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

    gl::context = cv::ocl::OpenCLExecutionContext::getCurrent();
}

std::string get_info() {
    return reinterpret_cast<const char*>(glGetString(GL_VERSION));
}

void acquire_from_gl(cv::UMat &m) {
    gl::begin();
    GL_CHECK(cv::ogl::convertFromGLTexture2D(*gl::frame_buf_tex, m));
    //The OpenGL frameBuffer is upside-down. Flip it. (OpenCL)
    cv::flip(m, m, 0);
}

void release_to_gl(cv::UMat &m) {
    //The OpenGL frameBuffer is upside-down. Flip it back. (OpenCL)
    cv::flip(m, m, 0);
    GL_CHECK(cv::ogl::convertToGLTexture2D(m, *gl::frame_buf_tex));
    gl::end();
}

void blit_frame_buffer_to_screen() {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, kb::gl::frame_buf);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, WIDTH, HEIGHT, 0, 0, WIDTH, HEIGHT, GL_COLOR_BUFFER_BIT, GL_NEAREST);

}

bool display() {
    if(x11::is_initialized()) {
        //Blit the framebuffer we have been working on to the screen
        gl::blit_frame_buffer_to_screen();

        //Check if the x11 window was closed
        if(x11::window_closed())
            return false;

        //Transfer the back buffer (which we have been using as frame buffer) to the native window
        egl::swap_buffers();
    }

    return true;
}
} // namespace gl

namespace cl {
std::string get_info() {
    std::stringstream ss;
    std::vector<cv::ocl::PlatformInfo> plt_info;
    cv::ocl::getPlatfomsInfo(plt_info);
    const cv::ocl::Device &device = cv::ocl::Device::getDefault();
    for (const auto &info : plt_info) {
        ss << "\t* " << info.version() << " = " << info.name() << endl;
    }

    ss << "\t  GL sharing: " << (device.isExtensionSupported("cl_khr_gl_sharing") ? "true" : "false") << endl;
    ss << "\t  VAAPI media sharing: " << (device.isExtensionSupported("cl_intel_va_api_media_sharing") ? "true" : "false") << endl;
    return ss.str();
}
} //namespace cl

namespace nvg {
NVGcontext *vg;

void clear(const float& r = 0.0f, const float& g = 0.0f, const float& b = 0.0f) {
    GL_CHECK(glClearColor(r, g, b, 1.0f));
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));
}

void begin() {
    gl::begin();
    gl::push();

    float w = WIDTH;
    float h = HEIGHT;
    if(x11::is_initialized()) {
        auto ws = x11::get_window_size();
        w = ws.first;
        h = ws.second;
    }

    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, kb::gl::frame_buf));
    nvgSave(vg);
    GL_CHECK(glViewport(0, HEIGHT - h, w, h));
    nvgBeginFrame(vg, w, h, std::fmax(WIDTH/w, HEIGHT/h));
}

void end() {
    nvgEndFrame(vg);
    nvgRestore(vg);
    gl::pop();
    gl::end();
}

void init(bool debug = false) {
    gl::push();

    GL_CHECK(glViewport(0, 0, WIDTH, HEIGHT));
    GL_CHECK(glEnable(GL_STENCIL_TEST));
    GL_CHECK(glStencilMask(~0));
    GL_CHECK(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

    vg = nvgCreateGL3(NVG_STENCIL_STROKES | debug ? NVG_DEBUG : 0);
    if (vg == NULL) {
        cerr << "Couldn't init nanovg." << endl;
        exit(24);
    }


    nvgCreateFont(vg, "icons", "assets/LinLibertine_R.ttf");
    /*nvgCreateFont(vg, "sans-bold", "fonts/DejaVuSans-Bold.ttf");
    nvgCreateFont(vg, "sans", "fonts/DejaVuSans.ttf");
    */

    gl::pop();
}
} //namespace nvg

void print_fps() {
    static uint64_t cnt = 0;
    static int64 start = cv::getTickCount();
    static double tickFreq = cv::getTickFrequency();
    static double fps = 1;

    if (cnt > 0 && cnt % uint64(ceil(fps)) == 0) {
        int64 tick = cv::getTickCount();
        fps = tickFreq / ((tick - start + 1) / cnt);
        cerr << "FPS : " << fps << '\r';
        start = tick;
        cnt = 1;
    }

    ++cnt;
}
}

#endif /* SRC_SUBSYSTEMS_HPP_ */
