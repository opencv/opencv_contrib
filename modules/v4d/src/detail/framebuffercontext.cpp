// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>
#define GL_GLEXT_PROTOTYPES 1
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <GL/glext.h>

#include "framebuffercontext.hpp"

#include "opencv2/v4d/util.hpp"
#include "opencv2/v4d/v4d.hpp"
#include "glcontext.hpp"
#include "nanovgcontext.hpp"
#include "nanoguicontext.hpp"
#include <opencv2/core/opengl.hpp>
namespace cv {
namespace v4d {
namespace detail {
long window_cnt = 0;

FrameBufferContext::FrameBufferContext(V4D& v4d, const string& title, const FrameBufferContext& other) : FrameBufferContext(v4d, other.frameBufferSize_, true, title, other.major_,  other.minor_, other.compat_, other.samples_, other.debug_, other.glfwWindow_, &other) {
}

FrameBufferContext::FrameBufferContext(V4D& v4d, const cv::Size& frameBufferSize, bool offscreen,
        const string& title, int major, int minor, bool compat, int samples, bool debug, GLFWwindow* sharedWindow, const FrameBufferContext* parent) :
        v4d_(&v4d), offscreen_(offscreen), title_(title), major_(major), minor_(
                minor), compat_(compat), samples_(samples), debug_(debug), viewport_(0, 0, frameBufferSize.width, frameBufferSize.height), windowSize_(frameBufferSize), frameBufferSize_(frameBufferSize), isShared_(false), sharedWindow_(sharedWindow), parent_(parent) {
        init();
}

FrameBufferContext::~FrameBufferContext() {
    teardown();
}

void FrameBufferContext::init() {
#ifndef OPENCV_V4D_USE_ES3
    if(parent_ != nullptr) {
        textureID_ = parent_->textureID_;
        isShared_ = true;
    }
#else
    isShared_ = false;
#endif
    if (glfwInit() != GLFW_TRUE)
        assert(false);

    glfwSetErrorCallback(cv::v4d::glfw_error_callback);

    if (debug_)
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);

    if (offscreen_)
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    else
        glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
    glfwSetTime(0);
#ifdef __APPLE__
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#elif defined(OPENCV_V4D_USE_ES3)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major_);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor_);
    glfwWindowHint(GLFW_OPENGL_PROFILE, compat_ ? GLFW_OPENGL_COMPAT_PROFILE : GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API) ;
#endif
    glfwWindowHint(GLFW_SAMPLES, samples_);
    glfwWindowHint(GLFW_RED_BITS, 8);
    glfwWindowHint(GLFW_GREEN_BITS, 8);
    glfwWindowHint(GLFW_BLUE_BITS, 8);
    glfwWindowHint(GLFW_ALPHA_BITS, 8);
    glfwWindowHint(GLFW_STENCIL_BITS, 8);
    glfwWindowHint(GLFW_DEPTH_BITS, 24);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

//    glfwWindowHint(GLFW_DOUBLEBUFFER, GL_FALSE);

    glfwWindow_ = glfwCreateWindow(frameBufferSize_.width, frameBufferSize_.height, std::to_string(++window_cnt).c_str(), nullptr,
            sharedWindow_);

    if (glfwWindow_ == NULL) {
        assert(false);
    }
    cerr << "WINDOW: " << glfwWindow_ << "/" << frameBufferSize_ << endl;
    this->resizeWindow(frameBufferSize_);
#ifndef OPENCV_V4D_USE_ES3
//    glewExperimental = true;
//    glewInit();

    try {
        this->makeCurrent();
        if (isClGlSharingSupported())
            cv::ogl::ocl::initializeContextFromGL();
        else
            clglSharing_ = false;
        this->makeNoneCurrent();

    } catch (std::exception& ex) {
        cerr << "CL-GL sharing failed: " << ex.what() << endl;
        clglSharing_ = false;
    } catch (...) {
        cerr << "CL-GL sharing failed with unknown error." << endl;
        clglSharing_ = false;
    }
    context_ = CLExecContext_t::getCurrent();
#else
    clglSharing_ = false;
#endif
    setup(frameBufferSize_);
    glfwSetWindowUserPointer(getGLFWWindow(), v4d_);

    glfwSetCursorPosCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, double x, double y) {
        V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
        v4d->nguiCtx().screen().cursor_pos_callback_event(x, y);
        auto cursor = v4d->getMousePosition();
        auto diff = cursor - cv::Vec2f(x, y);
        if (v4d->isMouseDrag()) {
            v4d->pan(diff[0], -diff[1]);
        }
        v4d->setMousePosition(x, y);
    }
    );
    glfwSetMouseButtonCallback(getGLFWWindow(),
            [](GLFWwindow* glfwWin, int button, int action, int modifiers) {
                V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
                v4d->nguiCtx().screen().mouse_button_callback_event(button, action, modifiers);
                if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                    v4d->setMouseDrag(action == GLFW_PRESS);
                }
            }
    );
    glfwSetKeyCallback(getGLFWWindow(),
            [](GLFWwindow* glfwWin, int key, int scancode, int action, int mods) {
                V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
                v4d->nguiCtx().screen().key_callback_event(key, scancode, action, mods);
            }
    );
    glfwSetCharCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, unsigned int codepoint) {
        V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
        v4d->nguiCtx().screen().char_callback_event(codepoint);
    }
    );
    glfwSetDropCallback(getGLFWWindow(),
            [](GLFWwindow* glfwWin, int count, const char** filenames) {
                V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
                v4d->nguiCtx().screen().drop_callback_event(count, filenames);
            }
    );
    glfwSetScrollCallback(getGLFWWindow(),
            [](GLFWwindow* glfwWin, double x, double y) {
                V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
                std::vector<nanogui::Widget*> widgets;
                find_widgets(&v4d->nguiCtx().screen(), widgets);
                for (auto* w : widgets) {
                    auto mousePos = nanogui::Vector2i(v4d->getMousePosition()[0] / v4d->fbCtx().getXPixelRatio(), v4d->getMousePosition()[1] / v4d->fbCtx().getYPixelRatio());
                    if(contains_absolute(w, mousePos)) {
                        v4d->nguiCtx().screen().scroll_callback_event(x, y);
                        return;
                    }
                }

//                v4d->zoom(y < 0 ? 1.1 : 0.9);
            }
    );

    glfwSetFramebufferSizeCallback(getGLFWWindow(),
            [](GLFWwindow* glfwWin, int width, int height) {
                V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
                v4d->setWindowSize(cv::Size(width, height));
                cv::Rect& vp = v4d->viewport();
                vp.x = 0;
                vp.y = 0;
                vp.width = width;
                vp.height = height;
#ifndef __EMSCRIPTEN__
                if(v4d->isResizable()) {
                    v4d->nvgCtx().fbCtx().teardown();
                    v4d->glCtx().fbCtx().teardown();
                    v4d->fbCtx().teardown();
                    v4d->fbCtx().setup(cv::Size(width, height));
                    v4d->glCtx().fbCtx().setup(cv::Size(width, height));
                    v4d->nvgCtx().fbCtx().setup(cv::Size(width, height));
                }
#endif
            });
}
void FrameBufferContext::setup(const cv::Size& sz) {
    frameBufferSize_ = sz;
    this->makeCurrent();
    if(!isShared_) {
        GL_CHECK(glGenFramebuffers(1, &frameBufferID_));
        cerr << "GENFB1: " << frameBufferID_ << "/" << getGLFWWindow() << endl;
        GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID_));
        GL_CHECK(glGenRenderbuffers(1, &renderBufferID_));

        GL_CHECK(glGenTextures(1, &textureID_));
        GL_CHECK(glBindTexture(GL_TEXTURE_2D, textureID_));
        texture_ = new cv::ogl::Texture2D(sz, cv::ogl::Texture2D::RGBA, textureID_);
        GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
        GL_CHECK(
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sz.width, sz.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0));

        GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, renderBufferID_));
        GL_CHECK(
                glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, sz.width, sz.height));
        GL_CHECK(
                glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderBufferID_));
        GL_CHECK(
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureID_, 0));
        assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    } else {
        assert(parent_ != nullptr);
        GL_CHECK(glGenFramebuffers(1, &frameBufferID_));
        cerr << "GENFB2: " << frameBufferID_ << "/" << getGLFWWindow() << endl;
        GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID_));

        GL_CHECK(glBindTexture(GL_TEXTURE_2D, textureID_));
        texture_ = new cv::ogl::Texture2D(sz, cv::ogl::Texture2D::RGBA, textureID_);
        GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
        GL_CHECK(
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sz.width, sz.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0));

        GL_CHECK(glGenRenderbuffers(1, &renderBufferID_));
        GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, renderBufferID_));
        GL_CHECK(
                glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, sz.width, sz.height));
        GL_CHECK(
                glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderBufferID_));
        GL_CHECK(
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureID_, 0));
        assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    }
    this->makeNoneCurrent();
}

void FrameBufferContext::teardown() {
    using namespace cv::ocl;
    this->makeCurrent();

#ifndef __EMSCRIPTEN__
    if(clImage_ != nullptr) {
        CLExecScope_t clExecScope(getCLExecContext());

        cl_int status = 0;
        cl_command_queue q = (cl_command_queue) Queue::getDefault().ptr();

        status = clEnqueueReleaseGLObjects(q, 1, &clImage_, 0, NULL, NULL);
        if (status != CL_SUCCESS)
            CV_Error_(cv::Error::OpenCLApiCallError, ("OpenCL: clEnqueueReleaseGLObjects failed: %d", status));

        status = clFinish(q); // TODO Use events
        if (status != CL_SUCCESS)
            CV_Error_(cv::Error::OpenCLApiCallError, ("OpenCL: clFinish failed: %d", status));

        status = clReleaseMemObject(clImage_); // TODO RAII
        if (status != CL_SUCCESS)
            CV_Error_(cv::Error::OpenCLApiCallError, ("OpenCL: clReleaseMemObject failed: %d", status));
        clImage_ = nullptr;
    }
#endif
    glBindTexture(GL_TEXTURE_2D, 0);
    glGetError();
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glGetError();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glGetError();
    assert(texture_ != nullptr);
    delete texture_;
    GL_CHECK(glDeleteTextures(1, &textureID_));
    GL_CHECK(glDeleteRenderbuffers(1, &renderBufferID_));
    GL_CHECK(glDeleteFramebuffers(1, &frameBufferID_));
    this->makeNoneCurrent();
}

void FrameBufferContext::toGLTexture2D(cv::UMat& u, cv::ogl::Texture2D& texture) {
#ifdef __EMSCRIPTEN__
    CV_UNUSED(u);
    CV_UNUSED(texture);
#else
    using namespace cv::ocl;

    cl_int status = 0;
    cl_command_queue q = (cl_command_queue) Queue::getDefault().ptr();

    if (clImage_ == nullptr) {
        Context& ctx = Context::getDefault();
        cl_context context = (cl_context) ctx.ptr();
        clImage_ = clCreateFromGLTexture(context, CL_MEM_WRITE_ONLY, 0x0DE1, 0, texture.texId(),
                &status);
        if (status != CL_SUCCESS)
            CV_Error_(cv::Error::OpenCLApiCallError,
                    ("OpenCL: clCreateFromGLTexture failed: %d", status));

        status = clEnqueueAcquireGLObjects(q, 1, &clImage_, 0, NULL, NULL);
        if (status != CL_SUCCESS)
            CV_Error_(cv::Error::OpenCLApiCallError,
                    ("OpenCL: clEnqueueAcquireGLObjects failed: %d", status));
    }

    cl_mem clBuffer = (cl_mem) u.handle(ACCESS_READ);

    size_t offset = 0;
    size_t dst_origin[3] = { 0, 0, 0 };
    size_t region[3] = { (size_t) u.cols, (size_t) u.rows, 1 };
    status = clEnqueueCopyBufferToImage(q, clBuffer, clImage_, offset, dst_origin, region, 0, NULL,
    NULL);
    if (status != CL_SUCCESS)
        CV_Error_(cv::Error::OpenCLApiCallError,
                ("OpenCL: clEnqueueCopyBufferToImage failed: %d", status));
#endif
    this->makeNoneCurrent();
}

void FrameBufferContext::fromGLTexture2D(const cv::ogl::Texture2D& texture, cv::UMat& u) {
#ifdef __EMSCRIPTEN__
    CV_UNUSED(u);
    CV_UNUSED(texture);
#else
    using namespace cv::ocl;

    const int dtype = CV_8UC4;
    int textureType = dtype;

    if (u.size() != texture.size() || u.type() != textureType) {
        u.create(texture.size(), textureType);
    }

    cl_command_queue q = (cl_command_queue) Queue::getDefault().ptr();

    cl_int status = 0;
    if (clImage_ == nullptr) {
        Context& ctx = Context::getDefault();
        cl_context context = (cl_context) ctx.ptr();
        clImage_ = clCreateFromGLTexture(context, CL_MEM_READ_ONLY, 0x0DE1, 0, texture.texId(),
                &status);
        if (status != CL_SUCCESS)
            CV_Error_(cv::Error::OpenCLApiCallError,
                    ("OpenCL: clCreateFromGLTexture failed: %d", status));

        status = clEnqueueAcquireGLObjects(q, 1, &clImage_, 0, NULL, NULL);
        if (status != CL_SUCCESS)
            CV_Error_(cv::Error::OpenCLApiCallError,
                    ("OpenCL: clEnqueueAcquireGLObjects failed: %d", status));
    }

    cl_mem clBuffer = (cl_mem) u.handle(ACCESS_WRITE);

    size_t offset = 0;
    size_t src_origin[3] = { 0, 0, 0 };
    size_t region[3] = { (size_t) u.cols, (size_t) u.rows, 1 };
    status = clEnqueueCopyImageToBuffer(q, clImage_, clBuffer, src_origin, region, offset, 0, NULL,
    NULL);
    if (status != CL_SUCCESS)
        CV_Error_(cv::Error::OpenCLApiCallError,
                ("OpenCL: clEnqueueCopyImageToBuffer failed: %d", status));
#endif
}

cv::Size FrameBufferContext::getSize() {
    return frameBufferSize_;
}

void FrameBufferContext::execute(std::function<void(cv::UMat&)> fn) {
    if(tmpBuffer_.empty())
        tmpBuffer_.create(getSize(), CV_8UC4);
    cv::resize(tmpBuffer_,frameBuffer_, getSize());
#ifndef __EMSCRIPTEN__
    CLExecScope_t clExecScope(getCLExecContext());
#endif
    FrameBufferContext::GLScope glScope(*this);
    FrameBufferContext::FrameBufferScope fbScope(*this, frameBuffer_);
    fn(frameBuffer_);
}

cv::ogl::Texture2D& FrameBufferContext::getTexture2D() {
    return *texture_;
}

GLFWwindow* FrameBufferContext::getGLFWWindow() {
    return glfwWindow_;
}

#ifndef __EMSCRIPTEN__
CLExecContext_t& FrameBufferContext::getCLExecContext() {
    return context_;
}
#endif

void FrameBufferContext::blitFrameBufferToScreen(const cv::Rect& viewport,
        const cv::Size& windowSize, bool stretch) {
    GL_CHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, frameBufferID_));
    if(!isShared_) {
        GL_CHECK(glReadBuffer(GL_COLOR_ATTACHMENT0));
    } else {
        GL_CHECK(glReadBuffer(GL_COLOR_ATTACHMENT0));
    }
    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
    GL_CHECK(
            glBlitFramebuffer( viewport.x, viewport.y, viewport.x + viewport.width, viewport.y + viewport.height, stretch ? 0 : windowSize.width - frameBufferSize_.width, stretch ? 0 : windowSize.height - frameBufferSize_.height, stretch ? windowSize.width : frameBufferSize_.width, stretch ? windowSize.height : frameBufferSize_.height, GL_COLOR_BUFFER_BIT, GL_NEAREST));
}

void FrameBufferContext::begin() {
    this->makeCurrent();
    glGetIntegerv( GL_VIEWPORT, viewport_ );
    glGetError();
    glBindTexture(GL_TEXTURE_2D, 0);
    glGetError();
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glGetError();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glGetError();

    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID_));
    GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, renderBufferID_));
    GL_CHECK(
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderBufferID_));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, textureID_));
        GL_CHECK(
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureID_, 0));
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, frameBufferSize_.width, frameBufferSize_.height);
    glGetError();
}

void FrameBufferContext::end() {
    glBindTexture(GL_TEXTURE_2D, 0);
    glGetError();
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glGetError();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glGetError();
    glViewport(viewport_[0], viewport_[1], viewport_[2], viewport_[3]);
    glGetError();
    GL_CHECK(glFlush());
    GL_CHECK(glFinish());
    this->makeNoneCurrent();
}

void FrameBufferContext::download(cv::UMat& m) {
    cv::Mat tmp = m.getMat(cv::ACCESS_WRITE);
    assert(tmp.data != nullptr);
    //this should use a PBO for the pixel transfer, but i couldn't get it to work for both opengl and webgl at the same time
    GL_CHECK(glReadPixels(0, 0, tmp.cols, tmp.rows, GL_RGBA, GL_UNSIGNED_BYTE, tmp.data));
    tmp.release();
}

void FrameBufferContext::upload(const cv::UMat& m) {
    cv::Mat tmp = m.getMat(cv::ACCESS_READ);
    assert(tmp.data != nullptr);
    GL_CHECK(
            glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, tmp.cols, tmp.rows, GL_RGBA, GL_UNSIGNED_BYTE, tmp.data));
    tmp.release();
}

void FrameBufferContext::acquireFromGL(cv::UMat& m) {
    if (clglSharing_) {
        GL_CHECK(fromGLTexture2D(getTexture2D(), m));
    } else {
        download(m);
        GL_CHECK(glFlush());
        GL_CHECK(glFinish());
    }
    //FIXME
    cv::flip(m, m, 0);
}

void FrameBufferContext::releaseToGL(cv::UMat& m) {
    //FIXME
    cv::flip(m, m, 0);
    if (clglSharing_) {
        GL_CHECK(toGLTexture2D(m, getTexture2D()));
    } else {
        if(m.empty())
            m.create(getSize(), CV_8UC4);
        upload(m);
        GL_CHECK(glFlush());
        GL_CHECK(glFinish());
    }
}

float FrameBufferContext::getXPixelRatio() {
    makeCurrent();
#ifdef __EMSCRIPTEN__
    float r = emscripten_get_device_pixel_ratio();
    makeNoneCurrent();
    return r;
#else
    float xscale, yscale;
    glfwGetWindowContentScale(getGLFWWindow(), &xscale, &yscale);
    makeNoneCurrent();
    return xscale;
#endif
}

float FrameBufferContext::getYPixelRatio() {
    makeCurrent();
#ifdef __EMSCRIPTEN__
    float r = emscripten_get_device_pixel_ratio();
    makeNoneCurrent();
    return r;
#else
    float xscale, yscale;
    glfwGetWindowContentScale(getGLFWWindow(), &xscale, &yscale);
    makeNoneCurrent();
    return yscale;
#endif
}

void FrameBufferContext::makeCurrent() {
    detail::proxy_to_mainv([this](){
        glfwMakeContextCurrent(getGLFWWindow());
    });
}

void FrameBufferContext::makeNoneCurrent() {
    detail::proxy_to_mainv([](){
        glfwMakeContextCurrent(nullptr);
    });
}

bool FrameBufferContext::isResizable() {
    makeCurrent();

    return detail::proxy_to_mainb([this](){
        return glfwGetWindowAttrib(getGLFWWindow(), GLFW_RESIZABLE) == GLFW_TRUE;
    });

    makeNoneCurrent();
}

void FrameBufferContext::setResizable(bool r) {
    makeCurrent();

    detail::proxy_to_mainv([r](){
        glfwWindowHint(GLFW_RESIZABLE, r ? GLFW_TRUE : GLFW_FALSE);

    });

    makeNoneCurrent();
}

cv::Size FrameBufferContext::getWindowSize() {
    return windowSize_;
}

void FrameBufferContext::setWindowSize(const cv::Size& sz) {
    windowSize_ = sz;
}

void FrameBufferContext::resizeWindow(const cv::Size& sz) {
    makeCurrent();
    detail::proxy_to_mainv([sz,this](){
        glfwSetWindowSize(getGLFWWindow(), sz.width, sz.height);
    });
    makeNoneCurrent();
}

bool FrameBufferContext::isFullscreen() {
    makeCurrent();
    return detail::proxy_to_mainb([this](){
        return glfwGetWindowMonitor(getGLFWWindow()) != nullptr;
    });
    makeNoneCurrent();
}

void FrameBufferContext::setFullscreen(bool f) {
    makeCurrent();

    detail::proxy_to_mainv([f,this](){
        auto monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        if (f) {
            glfwSetWindowMonitor(getGLFWWindow(), monitor, 0, 0, mode->width, mode->height,
                    mode->refreshRate);
            resizeWindow(getNativeFrameBufferSize());
        } else {
            glfwSetWindowMonitor(getGLFWWindow(), nullptr, 0, 0, getSize().width,
                    getSize().height, 0);
            resizeWindow(getSize());
        }

    });
    makeNoneCurrent();
}

cv::Size FrameBufferContext::getNativeFrameBufferSize() {
    makeCurrent();
    cv::Size* sz = reinterpret_cast<cv::Size*>(detail::proxy_to_mainl([this](){
        int w, h;
        glfwGetFramebufferSize(getGLFWWindow(), &w, &h);
        return reinterpret_cast<long>(new cv::Size{w, h});
    }));
    makeNoneCurrent();
    cv::Size copy = *sz;
    delete sz;
    return copy;
}

bool FrameBufferContext::isVisible() {
    makeCurrent();
    return detail::proxy_to_mainb([this]()-> bool {
        return glfwGetWindowAttrib(getGLFWWindow(), GLFW_VISIBLE) == GLFW_TRUE;
    });
    makeNoneCurrent();
}

void FrameBufferContext::setVisible(bool v) {
    makeCurrent();
    detail::proxy_to_mainv([v,this](){
        if (v)
            glfwShowWindow(getGLFWWindow());
        else
            glfwHideWindow(getGLFWWindow());

    });
    makeNoneCurrent();
}

}
}
}
