// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>
#include "opencv2/v4d/v4d.hpp"
#include "framebuffercontext.hpp"
#include "../util.hpp"
#include "glcontext.hpp"
#include "nanovgcontext.hpp"
#include "nanoguicontext.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/opengl.hpp"
#include <exception>
#include <iostream>

#ifdef __EMSCRIPTEN__
#  include <emscripten.h>
#  include <emscripten/bind.h>
#endif

namespace cv {
namespace v4d {
namespace detail {

void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error: (%d) %s\n", error, description);
}

static bool contains_absolute(nanogui::Widget* w, const nanogui::Vector2i& p) {
    nanogui::Vector2i d = p - w->absolute_position();
    return d.x() >= 0 && d.y() >= 0 && d.x() < w->size().x() && d.y() < w->size().y();
}

int frameBufferContextCnt = 0;

FrameBufferContext::FrameBufferContext(V4D& v4d, const string& title, const FrameBufferContext& other) : FrameBufferContext(v4d, other.framebufferSize_, true, title, other.major_,  other.minor_, other.compat_, other.samples_, other.debug_, other.glfwWindow_, &other) {
}

FrameBufferContext::FrameBufferContext(V4D& v4d, const cv::Size& framebufferSize, bool offscreen,
        const string& title, int major, int minor, bool compat, int samples, bool debug, GLFWwindow* sharedWindow, const FrameBufferContext* parent) :
        v4d_(&v4d), offscreen_(offscreen), title_(title), major_(major), minor_(
                minor), compat_(compat), samples_(samples), debug_(debug), isVisible_(offscreen), viewport_(0, 0, framebufferSize.width, framebufferSize.height), framebufferSize_(framebufferSize), isShared_(false), sharedWindow_(sharedWindow), parent_(parent), framebuffer_(framebufferSize, CV_8UC4) {
    run_sync_on_main<1>([this](){ init(); });
    index_ = ++frameBufferContextCnt;
}

FrameBufferContext::~FrameBufferContext() {
        teardown();
}

GLuint FrameBufferContext::getFramebufferID() {
    return frameBufferID_;
}

GLuint FrameBufferContext::getTextureID() {
    return textureID_;
}


void FrameBufferContext::loadShader() {
#if !defined(__EMSCRIPTEN__) && !defined(OPENCV_V4D_USE_ES3)
    const string shaderVersion = "330";
#else
    const string shaderVersion = "300 es";
#endif

    const string vert =
            "    #version " + shaderVersion
                    + R"(
    layout (location = 0) in vec3 aPos;
    
    void main()
    {
        gl_Position = vec4(aPos, 1.0);
    }
)";

    const string frag =
            "    #version " + shaderVersion
                    + R"(
    precision mediump float;
    out vec4 FragColor;
    
    uniform sampler2D texture0;
    uniform vec2 resolution;

    void main()
    {      
        //translate screen coordinates to texture coordinates and flip the y-axis.   
        vec4 texPos = gl_FragCoord / vec4(resolution.x, resolution.y * -1.0f, 1.0, 1.0);
        vec4 texColor0 = texture(texture0, texPos.xy);
        FragColor = texColor0;
    }
)";

    shader_program_hdl = cv::v4d::initShader(vert.c_str(), frag.c_str(), "FragColor");
}

void FrameBufferContext::loadBuffers() {
    glGenVertexArrays(1, &copyVao);
    glBindVertexArray(copyVao);

    glGenBuffers(1, &copyVbo);
    glGenBuffers(1, &copyEbo);

    glBindBuffer(GL_ARRAY_BUFFER, copyVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(copyVertices), copyVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, copyEbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(copyIndices), copyIndices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*) 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void FrameBufferContext::initWebGLCopy(FrameBufferContext& dst) {
#ifdef __EMSCRIPTEN__
    this->makeCurrent();
    GL_CHECK(glGenFramebuffers(1, &copyFramebuffer_));
    GL_CHECK(glGenTextures(1, &copyTexture_));

    loadShader();
    loadBuffers();

    // lookup the sampler locations.
    texture_hdl = glGetUniformLocation(shader_program_hdl, "texture0");
    resolution_hdl = glGetUniformLocation(shader_program_hdl, "resolution");
//    dst.makeCurrent();
#else
    throw std::runtime_error("WebGL not supported in none WASM builds");
#endif
}

void FrameBufferContext::doWebGLCopy(FrameBufferContext& dst) {
#ifdef __EMSCRIPTEN__
    int width = dst.getWindowSize().width;
    int height = dst.getWindowSize().height;
    {
        FrameBufferContext::GLScope glScope(dst, GL_READ_FRAMEBUFFER);
        dst.blitFrameBufferToScreen(
                cv::Rect(0,0, width, height),
                dst.getWindowSize(),
                false);
        emscripten_webgl_commit_frame();
    }
    this->makeCurrent();
    GL_CHECK(glEnable(GL_BLEND));
    GL_CHECK(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, copyFramebuffer_));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, copyTexture_));
    GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    GL_CHECK(
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, copyTexture_, 0));
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

    EM_ASM({
        var gl = Module.ctx;
        var canvas = document.getElementById('canvas' + $0);

        if(typeof Module.copyFramebuffer1 === 'undefined') {
            Module.copyFramebuffer1 = gl.getParameter(gl.FRAMEBUFFER_BINDING);
        }

        if(typeof Module.copyTexture1 === 'undefined') {
            Module.copyTexture1 = gl.getParameter(gl.TEXTURE_BINDING_2D);
        }
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.bindFramebuffer(gl.FRAMEBUFFER, Module.copyFramebuffer1);
        gl.bindTexture(gl.TEXTURE_2D, Module.copyTexture1);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, Module.copyTexture1, 0);
    }, dst.getIndex() - 1);
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    GL_CHECK(glReadBuffer(GL_COLOR_ATTACHMENT1));
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, this->getFramebufferID()));
    glViewport(0, 0, width, height);
    GL_CHECK(glUseProgram(shader_program_hdl));

    // set which texture units to render with.
    GL_CHECK(glUniform1i(texture_hdl, 0));  // texture unit 0
    float res[2] = {float(width), float(height)};
    GL_CHECK(glUniform2fv(resolution_hdl, 1, res));
    // Set each texture unit to use a particular texture.
    GL_CHECK(glActiveTexture(GL_TEXTURE0));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, copyTexture_));

    GL_CHECK(glBindVertexArray(copyVao));
    GL_CHECK(glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0));
    GL_CHECK(glDisable(GL_BLEND));
    GL_CHECK(glFlush());
#else
    throw std::runtime_error("WebGL not supported in none WASM builds");
#endif
}


void FrameBufferContext::init() {
#if !defined(__EMSCRIPTEN__)
    if(parent_ != nullptr) {
        textureID_ = parent_->textureID_;
        renderBufferID_ = parent_->renderBufferID_;
        isShared_ = true;
    }
#else
    isShared_ = false;

#endif
    if (glfwInit() != GLFW_TRUE) {
	cerr << "Can't init GLFW" << endl;
    	exit(1);
    }
    glfwSetErrorCallback(cv::v4d::detail::glfw_error_callback);

    if (debug_)
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);

    glfwSetTime(0);
#ifdef __APPLE__
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#elif defined(OPENCV_V4D_USE_ES3) || defined(__EMSCRIPTEN__)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major_);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor_);
    glfwWindowHint(GLFW_OPENGL_PROFILE, compat_ ? GLFW_OPENGL_COMPAT_PROFILE : GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_NATIVE_CONTEXT_API);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API) ;
#endif
    glfwWindowHint(GLFW_SAMPLES, samples_);
    glfwWindowHint(GLFW_RED_BITS, 8);
    glfwWindowHint(GLFW_GREEN_BITS, 8);
    glfwWindowHint(GLFW_BLUE_BITS, 8);
    glfwWindowHint(GLFW_ALPHA_BITS, 8);
    glfwWindowHint(GLFW_STENCIL_BITS, 8);
    glfwWindowHint(GLFW_DEPTH_BITS, 24);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
#ifndef __EMSCRIPTEN__
    glfwWindowHint(GLFW_VISIBLE, offscreen_ ? GLFW_FALSE : GLFW_TRUE );
#else
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
#endif
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);

    glfwWindow_ = glfwCreateWindow(framebufferSize_.width, framebufferSize_.height, title_.c_str(), nullptr,
            sharedWindow_);

    if (glfwWindow_ == NULL) {
        assert(false);
    }
    this->makeCurrent();
#ifndef __EMSCRIPTEN__
    glfwSwapInterval(0);
#endif
#if !defined(OPENCV_V4D_USE_ES3) && !defined(__EMSCRIPTEN__)
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress))
        throw std::runtime_error("Could not initialize GLAD!");
    glGetError(); // pull and ignore unhandled errors like GL_INVALID_ENUM
    try {
        if (isClGlSharingSupported())
            cv::ogl::ocl::initializeContextFromGL();
        else
            clglSharing_ = false;
    } catch (std::exception& ex) {
        cerr << "CL-GL sharing failed: " << ex.what() << endl;
        clglSharing_ = false;
    } catch (...) {
        cerr << "CL-GL sharing failed with unknown error." << endl;
        clglSharing_ = false;
    }
#else
    clglSharing_ = false;
#endif
#ifndef __EMSCRIPTEN__
    context_ = CLExecContext_t::getCurrent();
#endif

    setup(framebufferSize_);
    glfwSetWindowUserPointer(getGLFWWindow(), &getV4D());

    glfwSetCursorPosCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, double x, double y) {
        V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
#ifdef __EMSCRIPTEN__
        x *= v4d->pixelRatioX();
        y *= v4d->pixelRatioY();
#endif

        if(v4d->hasNguiCtx()) {
            v4d->nguiCtx().screen().cursor_pos_callback_event(x, y);
        }
#ifndef __EMSCRIPTEN__
        auto cursor = v4d->getMousePosition();
        auto diff = cursor - cv::Vec2f(x, y);
        if (v4d->isMouseDrag()) {
            v4d->pan(diff[0], -diff[1]);
        }
#endif
        v4d->setMousePosition(x, y);
    }
    );
    glfwSetMouseButtonCallback(getGLFWWindow(),
            [](GLFWwindow* glfwWin, int button, int action, int modifiers) {
                V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
                if(v4d->hasNguiCtx())
                    v4d->nguiCtx().screen().mouse_button_callback_event(button, action, modifiers);
                if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                    v4d->setMouseDrag(action == GLFW_PRESS);
                }
            }
    );
    glfwSetKeyCallback(getGLFWWindow(),
            [](GLFWwindow* glfwWin, int key, int scancode, int action, int mods) {
                V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
                if(v4d->hasNguiCtx())
                    v4d->nguiCtx().screen().key_callback_event(key, scancode, action, mods);
            }
    );
    glfwSetCharCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, unsigned int codepoint) {
        V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
        if(v4d->hasNguiCtx())
            v4d->nguiCtx().screen().char_callback_event(codepoint);
    }
    );
    glfwSetDropCallback(getGLFWWindow(),
            [](GLFWwindow* glfwWin, int count, const char** filenames) {
                V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
                if(v4d->hasNguiCtx())
                    v4d->nguiCtx().screen().drop_callback_event(count, filenames);
            }
    );
    glfwSetScrollCallback(getGLFWWindow(),
            [](GLFWwindow* glfwWin, double x, double y) {
                V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
                std::vector<nanogui::Widget*> widgets;
                if(v4d->hasNguiCtx()) {
                    for (auto* w : v4d->nguiCtx().screen().children()) {
                        auto pt = v4d->getMousePosition();
                        auto mousePos = nanogui::Vector2i(pt[0], pt[1]);
                        if(cv::v4d::detail::contains_absolute(w, mousePos)) {
                            v4d->nguiCtx().screen().scroll_callback_event(x, y);
                            return;
                        }
                    }
                }
#ifndef __EMSCRIPTEN__
                v4d->zoom(y < 0 ? 1.1 : 0.9);
#endif
            }
    );

    glfwSetWindowSizeCallback(getGLFWWindow(),
            [](GLFWwindow* glfwWin, int width, int height) {
                cerr << "glfwSetWindowSizeCallback: " << width << endl;
                run_sync_on_main<23>([glfwWin, width, height]() {
                    V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
                    cv::Rect& vp = v4d->viewport();
                    cv::Size fbsz = v4d->framebufferSize();
                    vp.x = 0;
                    vp.y = 0;
                    vp.width = fbsz.width;
                    vp.height = fbsz.height;
                    if(v4d->hasNguiCtx()) {
#ifdef __EMSCRIPTEN__
                        dynamic_cast<nanogui::Widget*>(&v4d->nguiCtx().screen())->set_size({int(v4d->getWindowSize().width), int(v4d->getWindowSize().height)});
#else
                        dynamic_cast<nanogui::Widget*>(&v4d->nguiCtx().screen())->set_size({int(v4d->getWindowSize().width / v4d->pixelRatioX()), int(v4d->getWindowSize().height / v4d->pixelRatioY())});
#endif
                    }
                });
            });

    glfwSetFramebufferSizeCallback(getGLFWWindow(),
            [](GLFWwindow* glfwWin, int width, int height) {
//                cerr << "glfwSetFramebufferSizeCallback: " << width << endl;
//                        run_sync_on_main<22>([glfwWin, width, height]() {
//                            V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
////                            v4d->makeCurrent();
//                            cv::Rect& vp = v4d->viewport();
//                            cv::Size fbsz = v4d->framebufferSize();
//                            vp.x = 0;
//                            vp.y = 0;
//                            vp.width = fbsz.width;
//                            vp.height = fbsz.height;
//
//                            if(v4d->hasNguiCtx())
//                                v4d->nguiCtx().screen().resize_callback_event(width, height);
//                        });
//        #ifndef __EMSCRIPTEN__
//                        if(v4d->isResizable()) {
//                            v4d->nvgCtx().fbCtx().teardown();
//                            v4d->glCtx().fbCtx().teardown();
//                            v4d->fbCtx().teardown();
//                            v4d->fbCtx().setup(cv::Size(width, height));
//                            v4d->glCtx().fbCtx().setup(cv::Size(width, height));
//                            v4d->nvgCtx().fbCtx().setup(cv::Size(width, height));
//                        }
//        #endif
            });
}

V4D& FrameBufferContext::getV4D() {
   return *v4d_;
}

int FrameBufferContext::getIndex() {
   return index_;
}

void FrameBufferContext::setup(const cv::Size& sz) {
    framebufferSize_ = sz;
    this->makeCurrent();

    if(!isShared_) {
        GL_CHECK(glGenFramebuffers(1, &frameBufferID_));
        GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID_));
        GL_CHECK(glGenRenderbuffers(1, &renderBufferID_));

        GL_CHECK(glGenTextures(1, &textureID_));
        GL_CHECK(glBindTexture(GL_TEXTURE_2D, textureID_));
        texture_ = new cv::ogl::Texture2D(sz, cv::ogl::Texture2D::RGBA, textureID_);
        GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
        GL_CHECK(
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sz.width, sz.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0));
        GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        GL_CHECK(
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureID_, 0));

        GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, renderBufferID_));
        GL_CHECK(
                glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, sz.width, sz.height));
        GL_CHECK(
                glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderBufferID_));

        assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    } else {
        assert(parent_ != nullptr);

        GL_CHECK(glGenFramebuffers(1, &frameBufferID_));
        GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID_));
        GL_CHECK(glBindTexture(GL_TEXTURE_2D, textureID_));
        texture_ = new cv::ogl::Texture2D(sz, cv::ogl::Texture2D::RGBA, textureID_);
        GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
        GL_CHECK(
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sz.width, sz.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0));
        GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        GL_CHECK(
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureID_, 0));
        GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, renderBufferID_));
        GL_CHECK(
                glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, sz.width, sz.height));
        GL_CHECK(
                glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderBufferID_));
        assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    }
}


void FrameBufferContext::teardown() {
    using namespace cv::ocl;
    this->makeCurrent();

#ifndef __EMSCRIPTEN__
    if(clImage_ != nullptr && !getCLExecContext().empty()) {
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

    status = clFinish(q);
    if (status != CL_SUCCESS)
        CV_Error_(cv::Error::OpenCLApiCallError, ("OpenCL: clFinish failed: %d", status));
#endif
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

    status = clFinish(q);
    if (status != CL_SUCCESS)
        CV_Error_(cv::Error::OpenCLApiCallError, ("OpenCL: clFinish failed: %d", status));
#endif
}

cv::Size FrameBufferContext::size() {
    return framebufferSize_;
}

void FrameBufferContext::copyTo(cv::UMat& dst) {
    run_sync_on_main<7>([&,this](){
#ifndef __EMSCRIPTEN__
        if(!getCLExecContext().empty()) {
            CLExecScope_t clExecScope(getCLExecContext());
#endif
            FrameBufferContext::GLScope glScope(*this, GL_FRAMEBUFFER);
            FrameBufferContext::FrameBufferScope fbScope(*this, framebuffer_);
            framebuffer_.copyTo(dst);
#ifndef __EMSCRIPTEN__
        } else {
            FrameBufferContext::GLScope glScope(*this, GL_FRAMEBUFFER);
            FrameBufferContext::FrameBufferScope fbScope(*this, framebuffer_);
            framebuffer_.copyTo(dst);
        }
#endif
    });
}

void FrameBufferContext::copyFrom(const cv::UMat& src) {
    run_sync_on_main<18>([&,this](){
#ifndef __EMSCRIPTEN__
        if(!getCLExecContext().empty()) {
            CLExecScope_t clExecScope(getCLExecContext());
#endif
            FrameBufferContext::GLScope glScope(*this, GL_FRAMEBUFFER);
            FrameBufferContext::FrameBufferScope fbScope(*this, framebuffer_);
            src.copyTo(framebuffer_);
#ifndef __EMSCRIPTEN__
        } else {
            FrameBufferContext::GLScope glScope(*this, GL_FRAMEBUFFER);
            FrameBufferContext::FrameBufferScope fbScope(*this, framebuffer_);
            src.copyTo(framebuffer_);
        }
#endif
    });
}

void FrameBufferContext::execute(std::function<void(cv::UMat&)> fn) {
    run_sync_on_main<2>([&,this](){
#ifndef __EMSCRIPTEN__
        if(!getCLExecContext().empty()) {
            CLExecScope_t clExecScope(getCLExecContext());
#endif
            FrameBufferContext::GLScope glScope(*this, GL_FRAMEBUFFER);
            FrameBufferContext::FrameBufferScope fbScope(*this, framebuffer_);
            fn(framebuffer_);
#ifndef __EMSCRIPTEN__
        } else {
            FrameBufferContext::GLScope glScope(*this, GL_FRAMEBUFFER);
            FrameBufferContext::FrameBufferScope fbScope(*this, framebuffer_);
            fn(framebuffer_);
        }
#endif
    });
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
        const cv::Size& windowSize, bool scale, GLuint drawFramebufferID) {
    this->makeCurrent();
    double hf = double(windowSize.height) / framebufferSize_.height;
    double wf = double(windowSize.width) / framebufferSize_.width;
    double f;
    if (framebufferSize_.width > framebufferSize_.height)
        f = wf;
    else
        f = hf;

    double fbws = framebufferSize_.width * f;
    double fbhs = framebufferSize_.height * f;

    double marginw = std::max((windowSize.width - framebufferSize_.width) / 2.0, 0.0);
    double marginh = std::max((windowSize.height - framebufferSize_.height) / 2.0, 0.0);
    double marginws = std::max((windowSize.width - fbws) / 2.0, 0.0);
    double marginhs = std::max((windowSize.height - fbhs) / 2.0, 0.0);

    GLint srcX0 = viewport.x;
    GLint srcY0 = viewport.y;
    GLint srcX1 = viewport.x + viewport.width;
    GLint srcY1 = viewport.y + viewport.height;
    GLint dstX0 = scale ? marginws : marginw;
    GLint dstY0 = scale ? marginhs : marginh;
    GLint dstX1 = scale ? marginws + fbws : marginw + framebufferSize_.width;
    GLint dstY1 = scale ? marginhs + fbhs : marginh + framebufferSize_.height;

    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, drawFramebufferID));
    GL_CHECK(glBlitFramebuffer( srcX0, srcY0, srcX1, srcY1,
            dstX0, dstY0, dstX1, dstY1,
            GL_COLOR_BUFFER_BIT, GL_NEAREST));
}

void FrameBufferContext::begin(GLenum framebufferTarget) {
    this->makeCurrent();
    GL_CHECK(glFinish());
    GL_CHECK(glBindFramebuffer(framebufferTarget, frameBufferID_));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, textureID_));
    GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, renderBufferID_));
    GL_CHECK(
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, size().width, size().height));
    GL_CHECK(
            glFramebufferRenderbuffer(framebufferTarget, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderBufferID_));
    GL_CHECK(
            glFramebufferTexture2D(framebufferTarget, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureID_, 0));
    assert(glCheckFramebufferStatus(framebufferTarget) == GL_FRAMEBUFFER_COMPLETE);
}

void FrameBufferContext::end() {
    this->makeCurrent();
    GL_CHECK(glFlush());
}

void FrameBufferContext::download(cv::UMat& m) {
    cv::Mat tmp = m.getMat(cv::ACCESS_WRITE);
    assert(tmp.data != nullptr);
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
        upload(m);
    }
}

cv::Vec2f FrameBufferContext::position() {
    makeCurrent();
    int x, y;
    glfwGetWindowPos(getGLFWWindow(), &x, &y);
    return cv::Vec2f(x, y);
}

float FrameBufferContext::pixelRatioX() {
    makeCurrent();
#ifdef __EMSCRIPTEN__
    float r = emscripten_get_device_pixel_ratio();

    return r;
#else
    float xscale, yscale;
    glfwGetWindowContentScale(getGLFWWindow(), &xscale, &yscale);

    return xscale;
#endif
}

float FrameBufferContext::pixelRatioY() {
    makeCurrent();
#ifdef __EMSCRIPTEN__
    float r = emscripten_get_device_pixel_ratio();

    return r;
#else
    float xscale, yscale;
    glfwGetWindowContentScale(getGLFWWindow(), &xscale, &yscale);

    return yscale;
#endif
}

void FrameBufferContext::makeCurrent() {
    assert(getGLFWWindow() != nullptr);
    glfwMakeContextCurrent(getGLFWWindow());
}

bool FrameBufferContext::isResizable() {
    makeCurrent();
    return glfwGetWindowAttrib(getGLFWWindow(), GLFW_RESIZABLE) == GLFW_TRUE;
}

void FrameBufferContext::setResizable(bool r) {
    makeCurrent();
    glfwSetWindowAttrib(getGLFWWindow(), GLFW_RESIZABLE, r ? GLFW_TRUE : GLFW_FALSE);
}

void FrameBufferContext::setWindowSize(const cv::Size& sz) {
    makeCurrent();
    glfwSetWindowSize(getGLFWWindow(), sz.width, sz.height);
}

//FIXME cache window size
cv::Size FrameBufferContext::getWindowSize() {
    makeCurrent();
    cv::Size sz;
    glfwGetWindowSize(getGLFWWindow(), &sz.width, &sz.height);
    return sz;
}

bool FrameBufferContext::isFullscreen() {
    makeCurrent();
    return glfwGetWindowMonitor(getGLFWWindow()) != nullptr;
}

void FrameBufferContext::setFullscreen(bool f) {
    makeCurrent();
    auto monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    if (f) {
        glfwSetWindowMonitor(getGLFWWindow(), monitor, 0, 0, mode->width, mode->height,
                mode->refreshRate);
        setWindowSize(getNativeFrameBufferSize());
    } else {
        glfwSetWindowMonitor(getGLFWWindow(), nullptr, 0, 0, size().width,
                size().height, 0);
        setWindowSize(size());
    }
}

cv::Size FrameBufferContext::getNativeFrameBufferSize() {
    makeCurrent();
    int w, h;
    glfwGetFramebufferSize(getGLFWWindow(), &w, &h);
    return cv::Size{w, h};
}

//cache window visibility instead of performing a heavy window attrib query.
bool FrameBufferContext::isVisible() {
    return isVisible_;
}

void FrameBufferContext::setVisible(bool v) {
    isVisible_ = v;
    makeCurrent();
    if (isVisible_)
        glfwShowWindow(getGLFWWindow());
    else
        glfwHideWindow(getGLFWWindow());
}

bool FrameBufferContext::isClosed() {
    return glfwWindow_ == nullptr;
}

void FrameBufferContext::close() {
    makeCurrent();
    teardown();
    glfwDestroyWindow(getGLFWWindow());
    glfwWindow_ = nullptr;
}

bool FrameBufferContext::isShared() {
    return isShared_;
}


}
}
}
