// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/detail/framebuffercontext.hpp"
#include "opencv2/v4d/v4d.hpp"
#include "opencv2/v4d/util.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/v4d/detail/gl.hpp"

#include "opencv2/core/opengl.hpp"
#include <opencv2/core/utils/logger.hpp>
#include <exception>
#include <iostream>
#include "imgui_impl_glfw.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
using std::cerr;
using std::cout;
using std::endl;

namespace cv {
namespace v4d {

namespace detail {

static void glfw_error_callback(int error, const char* description) {
#ifndef NDEBUG
    fprintf(stderr, "GLFW Error: (%d) %s\n", error, description);
#endif
}

bool FrameBufferContext::firstSync_ = true;

int frameBufferContextCnt = 0;

FrameBufferContext::FrameBufferContext(V4D& v4d, const string& title, cv::Ptr<FrameBufferContext> other) :
		FrameBufferContext(v4d, other->framebufferSize_, !other->debug_, title, other->major_,  other->minor_, other->samples_, other->debug_, other->rootWindow_, other, false) {
}

FrameBufferContext::FrameBufferContext(V4D& v4d, const cv::Size& framebufferSize, bool offscreen,
        const string& title, int major, int minor, int samples, bool debug, GLFWwindow* rootWindow, cv::Ptr<FrameBufferContext> parent, bool root) :
        v4d_(&v4d), offscreen_(offscreen), title_(title), major_(major), minor_(
                minor), samples_(samples), debug_(debug), isVisible_(offscreen), viewport_(0, 0, framebufferSize.width, framebufferSize.height), framebufferSize_(framebufferSize), hasParent_(false), rootWindow_(rootWindow), parent_(parent), framebuffer_(), isRoot_(root) {
    init();
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
#if !defined(OPENCV_V4D_USE_ES3)
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

    unsigned int handles[3];
    cv::v4d::initShader(handles, vert.c_str(), frag.c_str(), "fragColor");
    shader_program_hdls_[index] = handles[0];
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

void FrameBufferContext::init() {
	static std::mutex initMtx;
	std::unique_lock<std::mutex> lock(initMtx);

    if(parent_) {
    	hasParent_ = true;

        if(isRoot()) {
            textureID_ = 0;
            renderBufferID_ = 0;
    		onscreenTextureID_ = parent_->textureID_;
    		onscreenRenderBufferID_ = parent_->renderBufferID_;
        } else {
            textureID_ = parent_->textureID_;
            renderBufferID_ = parent_->renderBufferID_;
            onscreenTextureID_ = parent_->onscreenTextureID_;
            onscreenRenderBufferID_ = parent_->onscreenRenderBufferID_;
        }
    } else if (glfwInit() != GLFW_TRUE) {
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
#elif defined(OPENCV_V4D_USE_ES3)
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
    glfwWindowHint(GLFW_VISIBLE, offscreen_ ? GLFW_FALSE : GLFW_TRUE );
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);

    glfwWindow_ = glfwCreateWindow(framebufferSize_.width, framebufferSize_.height, title_.c_str(), nullptr, rootWindow_);


    if (glfwWindow_ == nullptr) {
        //retry with native api
        glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_NATIVE_CONTEXT_API);
        glfwWindow_ = glfwCreateWindow(framebufferSize_.width, framebufferSize_.height, title_.c_str(), nullptr,
        		rootWindow_);

        if (glfwWindow_ == nullptr) {
        	CV_Error(Error::StsError, "Unable to initialize window.");
        }
    }

    this->makeCurrent();

    if(isRoot()) {
    	rootWindow_ = glfwWindow_;
        glfwSwapInterval(1);
    } else {
        glfwSwapInterval(0);
    }

#if !defined(OPENCV_V4D_USE_ES3)
    if (!parent_) {
        GLenum err = glewInit();
        if (err != GLEW_OK && err != GLEW_ERROR_NO_GLX_DISPLAY) {
        	CV_Error(Error::StsError, "Could not initialize GLEW!");
        }
    }
#endif
    try {
        if (isRoot() && isClGlSharingSupported())
            cv::ogl::ocl::initializeContextFromGL();
        else
            clglSharing_ = false;
    } catch (std::exception& ex) {
        CV_LOG_WARNING(nullptr, "CL-GL sharing failed: %s" << ex.what());
        clglSharing_ = false;
    } catch (...) {
    	CV_LOG_WARNING(nullptr, "CL-GL sharing failed with unknown error");
        clglSharing_ = false;
    }
//#else
//    clglSharing_ = false;
//#endif

    context_ = CLExecContext_t::getCurrent();

    setup();
    if(isRoot()) {
    glfwSetWindowUserPointer(getGLFWWindow(), getV4D().get());
    glfwSetCursorPosCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, double x, double y) {
        V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
        if(v4d->hasImguiCtx()) {
            ImGui_ImplGlfw_CursorPosCallback(glfwWin, x, y);
            if (!ImGui::GetIO().WantCaptureMouse) {
                v4d->setMousePosition(cv::Point2f(float(x), float(y)));
            }
        }
    });

    glfwSetMouseButtonCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, int button, int action, int modifiers) {
        V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));

        if(v4d->hasImguiCtx()) {
            ImGui_ImplGlfw_MouseButtonCallback(glfwWin, button, action, modifiers);

            if (!ImGui::GetIO().WantCaptureMouse) {
                // Pass event further
            } else {
                // Do nothing, since imgui already reacted to mouse click. It would be weird if unrelated things started happening when you click something on UI.
            }
        }
    });

    glfwSetKeyCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, int key, int scancode, int action, int mods) {
        V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));

        if(v4d->hasImguiCtx()) {
            ImGui_ImplGlfw_KeyCallback(glfwWin, key, scancode, action, mods);
            if (!ImGui::GetIO().WantCaptureKeyboard) {
                // Pass event further
            } else {
                // Do nothing, since imgui already reacted to mouse click. It would be weird if unrelated things started happening when you click something on UI.
            }
        }
    });
    glfwSetCharCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, unsigned int codepoint) {
        V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));

        if(v4d->hasImguiCtx()) {
            ImGui_ImplGlfw_CharCallback(glfwWin, codepoint);
        }
    });
////    glfwSetDropCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, int count, const char** filenames) {
////        V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
////    });
//
//    glfwSetScrollCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, double x, double y) {
//        V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
//        if (v4d->hasImguiCtx()) {
//            ImGui_ImplGlfw_ScrollCallback(glfwWin, x, y);
//        }
//    });
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
////                        if(v4d->isResizable()) {
////                            v4d->nvgCtx().fbCtx()->teardown();
////                            v4d->glCtx().fbCtx()->teardown();
////                            v4d->fbCtx()->teardown();
////                            v4d->fbCtx()->setup(cv::Size(width, height));
////                            v4d->glCtx().fbCtx()->setup(cv::Size(width, height));
////                            v4d->nvgCtx().fbCtx()->setup(cv::Size(width, height));
////                        }
//            });
    glfwSetWindowFocusCallback(getGLFWWindow(), [](GLFWwindow* glfwWin, int i) {
            V4D* v4d = reinterpret_cast<V4D*>(glfwGetWindowUserPointer(glfwWin));
            if(v4d->getGLFWWindow() == glfwWin) {
                v4d->setFocused(i == 1);
            }
    });
    }
}

cv::Ptr<V4D> FrameBufferContext::getV4D() {
   return v4d_->self();
}

int FrameBufferContext::getIndex() {
   return index_;
}

void FrameBufferContext::setup() {
	cv::Size sz = framebufferSize_;
    CLExecScope_t clExecScope(getCLExecContext());
    framebuffer_.create(sz, CV_8UC4);
    if(isRoot()) {
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
    } else if(hasParent()) {
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
    } else
    	CV_Assert(false);
}

void FrameBufferContext::teardown() {
    using namespace cv::ocl;
    this->makeCurrent();
#ifdef HAVE_OPENCL
    if(cv::ocl::useOpenCL() && clImage_ != nullptr && !getCLExecContext().empty()) {
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
    CV_Assert(texture_ != nullptr);
    delete texture_;
    GL_CHECK(glDeleteTextures(1, &textureID_));
    GL_CHECK(glDeleteRenderbuffers(1, &renderBufferID_));
    GL_CHECK(glDeleteFramebuffers(1, &frameBufferID_));
    this->makeNoneCurrent();
}

#ifdef HAVE_OPENCL
void FrameBufferContext::toGLTexture2D(cv::UMat& u, cv::ogl::Texture2D& texture) {
    CV_Assert(clImage_ != nullptr);

	using namespace cv::ocl;

    cl_int status = 0;
    cl_command_queue q = (cl_command_queue) context_.getQueue().ptr();
    cl_mem clBuffer = (cl_mem) u.handle(ACCESS_READ);

    size_t offset = 0;
    size_t dst_origin[3] = { 0, 0, 0 };
    size_t region[3] = { (size_t) u.cols, (size_t) u.rows, 1 };
    status = clEnqueueCopyBufferToImage(q, clBuffer, clImage_, offset, dst_origin, region, 0, NULL,
    NULL);
    if (status != CL_SUCCESS)
        throw std::runtime_error("OpenCL: clEnqueueCopyBufferToImage failed: " + std::to_string(status));

    status = clEnqueueReleaseGLObjects(q, 1, &clImage_, 0, NULL, NULL);
    if (status != CL_SUCCESS)
         throw std::runtime_error("OpenCL: clEnqueueReleaseGLObjects failed: " + std::to_string(status));
}

void FrameBufferContext::fromGLTexture2D(const cv::ogl::Texture2D& texture, cv::UMat& u) {
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
    }

    status = clEnqueueAcquireGLObjects(q, 1, &clImage_, 0, NULL, NULL);
    if (status != CL_SUCCESS)
        throw std::runtime_error("OpenCL: clEnqueueAcquireGLObjects failed: " + std::to_string(status));

    cl_mem clBuffer = (cl_mem) u.handle(ACCESS_READ);

    size_t offset = 0;
    size_t src_origin[3] = { 0, 0, 0 };
    size_t region[3] = { (size_t) u.cols, (size_t) u.rows, 1 };
    status = clEnqueueCopyImageToBuffer(q, clImage_, clBuffer, src_origin, region, offset, 0, NULL,
    NULL);
    if (status != CL_SUCCESS)
        throw std::runtime_error("OpenCL: clEnqueueCopyImageToBuffer failed: " + std::to_string(status));
}
#endif
const cv::Size& FrameBufferContext::size() const {
    return framebufferSize_;
}

void FrameBufferContext::copyTo(cv::UMat& dst) {
	if(!getCLExecContext().empty()) {
		CLExecScope_t clExecScope(getCLExecContext());
		FrameBufferContext::GLScope glScope(this, GL_FRAMEBUFFER);
		FrameBufferContext::FrameBufferScope fbScope(this, framebuffer_);
		framebuffer_.copyTo(dst);
	} else {
		FrameBufferContext::GLScope glScope(this, GL_FRAMEBUFFER);
		FrameBufferContext::FrameBufferScope fbScope(this, framebuffer_);
		framebuffer_.copyTo(dst);
	}
}

void FrameBufferContext::copyFrom(const cv::UMat& src) {
	if(!getCLExecContext().empty()) {
		CLExecScope_t clExecScope(getCLExecContext());
		FrameBufferContext::GLScope glScope(this, GL_FRAMEBUFFER);
		FrameBufferContext::FrameBufferScope fbScope(this, framebuffer_);
		src.copyTo(framebuffer_);
	} else {
		FrameBufferContext::GLScope glScope(this, GL_FRAMEBUFFER);
		FrameBufferContext::FrameBufferScope fbScope(this, framebuffer_);
		src.copyTo(framebuffer_);
	}
}

void FrameBufferContext::copyToRootWindow() {
	GLScope scope(self_, GL_READ_FRAMEBUFFER);
	GL_CHECK(glReadBuffer(GL_COLOR_ATTACHMENT0));

	GL_CHECK(glActiveTexture(GL_TEXTURE0));
	GL_CHECK(glBindTexture(GL_TEXTURE_2D, onscreenTextureID_));
	GL_CHECK(glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, size().width, size().height));
}

cv::ogl::Texture2D& FrameBufferContext::getTexture2D() {
    return *texture_;
}

GLFWwindow* FrameBufferContext::getGLFWWindow() const {
    return glfwWindow_;
}

CLExecContext_t& FrameBufferContext::getCLExecContext() {
    return context_;
}

void FrameBufferContext::blitFrameBufferToFrameBuffer(const cv::Rect& srcViewport,
        const cv::Size& targetFbSize, GLuint targetFramebufferID, bool stretch, bool flipY) {
    double hf = double(targetFbSize.height) / framebufferSize_.height;
    double wf = double(targetFbSize.width) / framebufferSize_.width;
    double f;
    if (hf > wf)
        f = wf;
    else
        f = hf;

    double fbws = framebufferSize_.width * f;
    double fbhs = framebufferSize_.height * f;

    double marginw = (targetFbSize.width - framebufferSize_.width) / 2.0;
    double marginh = (targetFbSize.height - framebufferSize_.height) / 2.0;
    double marginws = (targetFbSize.width - fbws) / 2.0;
    double marginhs = (targetFbSize.height - fbhs) / 2.0;

    GLint srcX0 = srcViewport.x;
    GLint srcY0 = srcViewport.y;
    GLint srcX1 = srcViewport.x + srcViewport.width;
    GLint srcY1 = srcViewport.y + srcViewport.height;
    GLint dstX0 = stretch ? marginws : marginw;
    GLint dstY0 = stretch ? marginhs : marginh;
    GLint dstX1 = stretch ? marginws + fbws : marginw + framebufferSize_.width;
    GLint dstY1 = stretch ? marginhs + fbhs : marginh + framebufferSize_.height;
    if(flipY) {
        GLint tmp = dstY0;
        dstY0 = dstY1;
        dstY1 = tmp;
    }
    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, targetFramebufferID));
    GL_CHECK(glBlitFramebuffer( srcX0, srcY0, srcX1, srcY1,
            dstX0, dstY0, dstX1, dstY1,
            GL_COLOR_BUFFER_BIT, GL_NEAREST));
}

cv::UMat& FrameBufferContext::fb() {
	return framebuffer_;
}

void FrameBufferContext::begin(GLenum framebufferTarget) {
    this->makeCurrent();
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
    this->makeNoneCurrent();
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
#ifdef HAVE_OPENCL
	if (cv::ocl::useOpenCL() && clglSharing_) {
        try {
            GL_CHECK(fromGLTexture2D(getTexture2D(), m));
        } catch(...) {
            clglSharing_ = false;
            download(m);
        }
        return;
	}
#endif
    {
        download(m);
    }
    //FIXME
    cv::flip(m, m, 0);
}

void FrameBufferContext::releaseToGL(cv::UMat& m) {
    //FIXME
    cv::flip(m, m, 0);
#ifdef HAVE_OPENCL
    if (cv::ocl::useOpenCL() && clglSharing_) {
        try {
            GL_CHECK(toGLTexture2D(m, getTexture2D()));
        } catch(...) {
            clglSharing_ = false;
            upload(m);
        }
        return;
    }
#endif
    {
        upload(m);
    }
}

cv::Vec2f FrameBufferContext::position() {
    int x, y;
    glfwGetWindowPos(getGLFWWindow(), &x, &y);
    return cv::Vec2f(x, y);
}

float FrameBufferContext::pixelRatioX() {
    float xscale, yscale;
    glfwGetWindowContentScale(getGLFWWindow(), &xscale, &yscale);

    return xscale;
}

float FrameBufferContext::pixelRatioY() {
    float xscale, yscale;
    glfwGetWindowContentScale(getGLFWWindow(), &xscale, &yscale);

    return yscale;
}

void FrameBufferContext::makeCurrent() {
    assert(getGLFWWindow() != nullptr);
    glfwMakeContextCurrent(getGLFWWindow());
}

void FrameBufferContext::makeNoneCurrent() {
	glfwMakeContextCurrent(nullptr);
}


bool FrameBufferContext::isResizable() {
    return glfwGetWindowAttrib(getGLFWWindow(), GLFW_RESIZABLE) == GLFW_TRUE;
}

void FrameBufferContext::setResizable(bool r) {
    glfwSetWindowAttrib(getGLFWWindow(), GLFW_RESIZABLE, r ? GLFW_TRUE : GLFW_FALSE);
}

void FrameBufferContext::setWindowSize(const cv::Size& sz) {
    glfwSetWindowSize(getGLFWWindow(), sz.width, sz.height);
}

//FIXME cache window size
cv::Size FrameBufferContext::getWindowSize() {
    cv::Size sz;
    glfwGetWindowSize(getGLFWWindow(), &sz.width, &sz.height);
    return sz;
}

bool FrameBufferContext::isFullscreen() {
    return glfwGetWindowMonitor(getGLFWWindow()) != nullptr;
}

void FrameBufferContext::setFullscreen(bool f) {
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
    if (isVisible_)
        glfwShowWindow(getGLFWWindow());
    else
        glfwHideWindow(getGLFWWindow());
}

bool FrameBufferContext::isClosed() {
    return glfwWindow_ == nullptr;
}

void FrameBufferContext::close() {
    teardown();
    glfwDestroyWindow(getGLFWWindow());
    glfwWindow_ = nullptr;
}

bool FrameBufferContext::isRoot() {
    return isRoot_;
}


bool FrameBufferContext::hasParent() {
    return hasParent_;
}

bool FrameBufferContext::hasRootWindow() {
    return rootWindow_ != nullptr;
}

void FrameBufferContext::fence() {
    CV_Assert(currentSyncObject_ == 0);
    currentSyncObject_ = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    CV_Assert(currentSyncObject_ != 0);
}

bool FrameBufferContext::wait(const uint64_t& timeout) {
    if(firstSync_) {
        currentSyncObject_ = 0;
        firstSync_ = false;
        return true;
    }
    CV_Assert(currentSyncObject_ != 0);
    GLuint ret = glClientWaitSync(static_cast<GLsync>(currentSyncObject_),
    GL_SYNC_FLUSH_COMMANDS_BIT, timeout);
    GL_CHECK();
    CV_Assert(GL_WAIT_FAILED != ret);
    if(GL_CONDITION_SATISFIED == ret || GL_ALREADY_SIGNALED == ret) {
        currentSyncObject_ = 0;
        return true;
    } else {
        currentSyncObject_ = 0;
        return false;
    }
}
}
}
}
