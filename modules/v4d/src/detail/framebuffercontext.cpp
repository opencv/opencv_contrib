// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>
#include "opencv2/v4d/v4d.hpp"
#include "opencv2/v4d/detail/framebuffercontext.hpp"
#include "opencv2/v4d/util.hpp"
#include "glcontext.hpp"
#include "nanovgcontext.hpp"
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

static void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error: (%d) %s\n", error, description);
}


int frameBufferContextCnt = 0;

FrameBufferContext::FrameBufferContext(V4D& v4d, const string& title, const FrameBufferContext& other) : FrameBufferContext(v4d, other.framebufferSize_, !other.debug_, title, other.major_,  other.minor_, other.samples_, other.debug_, other.glfwWindow_, &other) {
}

FrameBufferContext::FrameBufferContext(V4D& v4d, const cv::Size& framebufferSize, bool offscreen,
        const string& title, int major, int minor, int samples, bool debug, GLFWwindow* sharedWindow, const FrameBufferContext* parent) :
        v4d_(&v4d), offscreen_(offscreen), title_(title), major_(major), minor_(
                minor), samples_(samples), debug_(debug), isVisible_(offscreen), viewport_(0, 0, framebufferSize.width, framebufferSize.height), framebufferSize_(framebufferSize), isShared_(false), sharedWindow_(sharedWindow), parent_(parent), framebuffer_() {
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


void FrameBufferContext::loadShader(const size_t& index) {
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
        if(texColor0.a == 0.0)
            discard;
        else
            FragColor = texColor0;
    }
)";

    shader_program_hdls_[index] = cv::v4d::initShader(vert.c_str(), frag.c_str(), "FragColor");
}

void FrameBufferContext::loadBuffers(const size_t& index) {
    glGenVertexArrays(1, &copyVaos[index]);
    glBindVertexArray(copyVaos[index]);

    glGenBuffers(1, &copyVbos[index]);
    glGenBuffers(1, &copyEbos[index]);

    glBindBuffer(GL_ARRAY_BUFFER, copyVbos[index]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(copyVertices), copyVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, copyEbos[index]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(copyIndices), copyIndices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*) 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void FrameBufferContext::initWebGLCopy(const size_t& index) {
#ifdef __EMSCRIPTEN__
    this->makeCurrent();
    GL_CHECK(glGenFramebuffers(1, &copyFramebuffers_[index]));
    GL_CHECK(glGenTextures(1, &copyTextures_[index]));
    loadShader(index);
    loadBuffers(index);

    // lookup the sampler locations.
    texture_hdls_[index] = glGetUniformLocation(shader_program_hdls_[index], "texture0");
    resolution_hdls_[index] = glGetUniformLocation(shader_program_hdls_[index], "resolution");
#else
    CV_UNUSED(index);
    throw std::runtime_error("WebGL not supported in none WASM builds");
#endif
}

void FrameBufferContext::doWebGLCopy(FrameBufferContext& other) {
#ifdef __EMSCRIPTEN__
    size_t index = other.getIndex();
    this->makeCurrent();
    int width = getWindowSize().width;
    int height = getWindowSize().height;
    {
        FrameBufferContext::GLScope glScope(*this, GL_READ_FRAMEBUFFER);
        other.blitFrameBufferToScreen(
                cv::Rect(0,0, other.size().width, other.size().height),
                this->getWindowSize(),
                false);
        GL_CHECK(glFinish());
        emscripten_webgl_commit_frame();
    }
    this->makeCurrent();
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, copyFramebuffers_[index]));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, copyTextures_[index]));

    EM_ASM({
        var gl = Module.ctx;

        var canvas;
        if($0 > 0)
            canvas = document.getElementById('canvas' + $0);
        else
            canvas = document.getElementById('canvas');

        if(typeof Module["copyFramebuffers" + $0] === 'undefined') {
            Module["copyFramebuffers" + $0] = gl.getParameter(gl.FRAMEBUFFER_BINDING);
        }

        if(typeof Module["copyTextures" + $0] === 'undefined') {
            Module["copyTextures" + $0]= gl.getParameter(gl.TEXTURE_BINDING_2D);
        }
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.bindFramebuffer(gl.FRAMEBUFFER, Module["copyFramebuffers" + $0]);
        gl.bindTexture(gl.TEXTURE_2D, Module["copyTextures" + $0] );
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, Module["copyTextures" + $0] , 0);
    }, index - 1);
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    GL_CHECK(glReadBuffer(GL_COLOR_ATTACHMENT1));
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, this->getFramebufferID()));
    glViewport(0, 0, width, height);
    GL_CHECK(glUseProgram(shader_program_hdls_[index]));

    // set which texture units to render with.
    GL_CHECK(glUniform1i(texture_hdls_[index], 0));  // texture unit 0
    float res[2] = {float(width), float(height)};
    GL_CHECK(glUniform2fv(resolution_hdls_[index], 1, res));
    // Set each texture unit to use a particular texture.
    GL_CHECK(glActiveTexture(GL_TEXTURE0));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, copyTextures_[index]));

    GL_CHECK(glBindVertexArray(copyVaos[index]));
    GL_CHECK(glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0));
    GL_CHECK(glDisable(GL_BLEND));
    GL_CHECK(glFinish());
#else
    CV_UNUSED(other);
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
    if (parent_ == nullptr && glfwInit() != GLFW_TRUE) {
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
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
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
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
#ifndef __EMSCRIPTEN__
    glfwWindowHint(GLFW_VISIBLE, offscreen_ ? GLFW_FALSE : GLFW_TRUE );
#else
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
#endif
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
//    glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);

    glfwWindow_ = glfwCreateWindow(framebufferSize_.width, framebufferSize_.height, title_.c_str(), nullptr,
            sharedWindow_);

    if (glfwWindow_ == nullptr) {
        //retry with native api
        glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_NATIVE_CONTEXT_API);
        glfwWindow_ = glfwCreateWindow(framebufferSize_.width, framebufferSize_.height, title_.c_str(), nullptr,
                sharedWindow_);

        if (glfwWindow_ == nullptr) {
            throw std::runtime_error("Unable to initialize window.");
        }
    }
    this->makeCurrent();
#ifndef __EMSCRIPTEN__
    glfwSwapInterval(0);
#endif
#if !defined(OPENCV_V4D_USE_ES3) && !defined(__EMSCRIPTEN__)
    if (parent_ == nullptr) {
        GLenum err = glewInit();
        if (GLEW_OK != err) {
            throw std::runtime_error("Could not initialize GLEW!");
        }
    }
    try {
        if (parent_ == nullptr && isClGlSharingSupported())
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
    glfwSetWindowUserPointer(getGLFWWindow(), getV4D().get());
//
//    glfwSetCursorPosCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, double x, double y) {
//        V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
//        v4d->setMousePosition(cv::Point2f(float(x), float(y)));
//    }
//    );
//    glfwSetMouseButtonCallback(getGLFWWindow(),
//            [](GLFWwindow* glfwWin, int button, int action, int modifiers) {
//            }
//    );
//    glfwSetKeyCallback(getGLFWWindow(),
//            [](GLFWwindow* glfwWin, int key, int scancode, int action, int mods) {
//            }
//    );
//    glfwSetCharCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, unsigned int codepoint) {
//    }
//    );
//    glfwSetDropCallback(getGLFWWindow(),
//            [](GLFWwindow* glfwWin, int count, const char** filenames) {
//            }
//    );
//    glfwSetScrollCallback(getGLFWWindow(),
//            [](GLFWwindow* glfwWin, double x, double y) {
//            }
//    );
//
//    glfwSetWindowSizeCallback(getGLFWWindow(),
//            [](GLFWwindow* glfwWin, int width, int height) {
//                cerr << "glfwSetWindowSizeCallback: " << width << endl;
//                run_sync_on_main<23>([glfwWin, width, height]() {
//                    V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
//                    cv::Rect& vp = v4d->viewport();
//                    cv::Size fbsz = v4d->framebufferSize();
//                    vp.x = 0;
//                    vp.y = 0;
//                    vp.width = fbsz.width;
//                    vp.height = fbsz.height;
//                });
//            });
//
//    glfwSetFramebufferSizeCallback(getGLFWWindow(),
//            [](GLFWwindow* glfwWin, int width, int height) {
////                cerr << "glfwSetFramebufferSizeCallback: " << width << endl;
////                        run_sync_on_main<22>([glfwWin, width, height]() {
////                            V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
//////                            v4d->makeCurrent();
////                            cv::Rect& vp = v4d->viewport();
////                            cv::Size fbsz = v4d->framebufferSize();
////                            vp.x = 0;
////                            vp.y = 0;
////                            vp.width = fbsz.width;
////                            vp.height = fbsz.height;
////
////                            if(v4d->hasNguiCtx())
////                                v4d->nguiCtx().screen().resize_callback_event(width, height);
////                        });
////        #ifndef __EMSCRIPTEN__
////                        if(v4d->isResizable()) {
////                            v4d->nvgCtx().fbCtx().teardown();
////                            v4d->glCtx().fbCtx().teardown();
////                            v4d->fbCtx().teardown();
////                            v4d->fbCtx().setup(cv::Size(width, height));
////                            v4d->glCtx().fbCtx().setup(cv::Size(width, height));
////                            v4d->nvgCtx().fbCtx().setup(cv::Size(width, height));
////                        }
////        #endif
//            });
    glfwSetWindowFocusCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, int i) {
                V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
                v4d->makeCurrent();
    });
}

cv::Ptr<V4D> FrameBufferContext::getV4D() {
   return v4d_->self();
}

int FrameBufferContext::getIndex() {
   return index_;
}

void FrameBufferContext::setup(const cv::Size& sz) {
    framebufferSize_ = sz;
    this->makeCurrent();
#ifndef __EMSCRIPTEN__
    CLExecScope_t clExecScope(getCLExecContext());
#endif
    framebuffer_.create(sz, CV_8UC4);

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
//    glBindTexture(GL_TEXTURE_2D, 0);
//    glGetError();
//    glBindRenderbuffer(GL_RENDERBUFFER, 0);
//    glGetError();
//    glBindFramebuffer(GL_FRAMEBUFFER, 0);
//    glGetError();
//    GL_CHECK(glFinish());
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
    cl_command_queue q = (cl_command_queue) context_.getQueue().ptr();

    if (clImage_ == nullptr) {
        Context& ctx = context_.getContext();
        cl_context context = (cl_context) ctx.ptr();
        clImage_ = clCreateFromGLTexture(context, CL_MEM_WRITE_ONLY, 0x0DE1, 0, texture.texId(),
                &status);
        if (status != CL_SUCCESS)
            throw std::runtime_error("OpenCL: clCreateFromGLTexture failed: " + std::to_string(status));

        status = clEnqueueAcquireGLObjects(q, 1, &clImage_, 0, NULL, NULL);
        if (status != CL_SUCCESS)
            throw std::runtime_error("OpenCL: clEnqueueAcquireGLObjects failed: " + std::to_string(status));
    }

    cl_mem clBuffer = (cl_mem) u.handle(ACCESS_READ);

    size_t offset = 0;
    size_t dst_origin[3] = { 0, 0, 0 };
    size_t region[3] = { (size_t) u.cols, (size_t) u.rows, 1 };
    status = clEnqueueCopyBufferToImage(q, clBuffer, clImage_, offset, dst_origin, region, 0, NULL,
    NULL);
    if (status != CL_SUCCESS)
        throw std::runtime_error("OpenCL: clEnqueueCopyBufferToImage failed: " + std::to_string(status));

    status = clFinish(q);
    if (status != CL_SUCCESS)
        throw std::runtime_error("OpenCL: clFinish failed: " + std::to_string(status));
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

    cl_command_queue q = (cl_command_queue) context_.getQueue().ptr();

    cl_int status = 0;
    if (clImage_ == nullptr) {
        Context& ctx = context_.getContext();
        cl_context context = (cl_context) ctx.ptr();
        clImage_ = clCreateFromGLTexture(context, CL_MEM_READ_ONLY, 0x0DE1, 0, texture.texId(),
                &status);
        if (status != CL_SUCCESS)
            throw std::runtime_error("OpenCL: clCreateFromGLTexture failed: " + std::to_string(status));

        status = clEnqueueAcquireGLObjects(q, 1, &clImage_, 0, NULL, NULL);
        if (status != CL_SUCCESS)
            throw std::runtime_error("OpenCL: clEnqueueAcquireGLObjects failed: " + std::to_string(status));
    }

    cl_mem clBuffer = (cl_mem) u.handle(ACCESS_WRITE);

    size_t offset = 0;
    size_t src_origin[3] = { 0, 0, 0 };
    size_t region[3] = { (size_t) u.cols, (size_t) u.rows, 1 };
    status = clEnqueueCopyImageToBuffer(q, clImage_, clBuffer, src_origin, region, offset, 0, NULL,
    NULL);
    if (status != CL_SUCCESS)
        throw std::runtime_error("OpenCL: clEnqueueCopyImageToBuffer failed: " + std::to_string(status));

    status = clFinish(q);
    if (status != CL_SUCCESS)
        throw std::runtime_error("OpenCL: clFinish failed: " + std::to_string(status));
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
//#ifdef __EMSCRIPTEN__
//    {
//        //FIXME WebGL2 workaround for webkit. without we have flickering
//        cv::UMat tmp(size(), CV_8UC4);
//        FrameBufferContext::FrameBufferScope fbScope(*this, tmp);
//    }
//#endif
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
        try {
            GL_CHECK(fromGLTexture2D(getTexture2D(), m));
        } catch(...) {
            clglSharing_ = false;
            download(m);
        }
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
        try {
            GL_CHECK(toGLTexture2D(m, getTexture2D()));
        } catch(...) {
            clglSharing_ = false;
            upload(m);
        }
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
//    makeCurrent();
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
