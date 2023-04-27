// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_FRAMEBUFFERCONTEXT_HPP_
#define SRC_OPENCV_FRAMEBUFFERCONTEXT_HPP_

#ifndef __EMSCRIPTEN__
#  ifndef CL_TARGET_OPENCL_VERSION
#    define CL_TARGET_OPENCL_VERSION 120
#  endif
#  ifdef __APPLE__
#    include <OpenCL/cl_gl_ext.h>
#  else
#    include <CL/cl_gl.h>
#  endif
#else
#  include <emscripten/threading.h>
#endif

#include <opencv2/core/ocl.hpp>
#include <iostream>

#include "opencv2/v4d/util.hpp"

struct GLFWwindow;
namespace cv {
namespace v4d {
class V4D;
namespace detail {
typedef cv::ocl::OpenCLExecutionContext CLExecContext_t;
typedef cv::ocl::OpenCLExecutionContextScope CLExecScope_t;

/*!
 * The FrameBufferContext acquires the framebuffer from OpenGL (either by up-/download or by cl-gl sharing)
 */
class FrameBufferContext {
    typedef unsigned int GLuint;
    typedef signed int GLint;

    friend class CLVAContext;
    friend class GLContext;
    friend class NanoVGContext;
    friend class NanoguiContext;
    friend class cv::v4d::V4D;
    V4D* v4d_ = nullptr;
    bool offscreen_;
    string title_;
    int major_;
    int minor_;
    bool compat_;
    int samples_;
    bool debug_;
    GLFWwindow* glfwWindow_ = nullptr;
    bool clglSharing_ = true;
    GLuint frameBufferID_ = 0;
    GLuint textureID_ = 0;
    GLuint renderBufferID_ = 0;
    GLint viewport_[4];
#ifndef __EMSCRIPTEN__
    cl_mem clImage_ = nullptr;
    CLExecContext_t context_;
#endif
    cv::Size windowSize_;
    cv::Size frameBufferSize_;
    bool isShared_ = false;
    GLFWwindow* sharedWindow_;
    const FrameBufferContext* parent_;
    /*!
     * The internal framebuffer exposed as OpenGL Texture2D.
     * @return The texture object.
     */
    cv::ogl::Texture2D& getTexture2D();
    GLFWwindow* getGLFWWindow();

#ifndef __EMSCRIPTEN__
    /*!
     * Get the current OpenCLExecutionContext
     * @return The current OpenCLExecutionContext
     */
    CLExecContext_t& getCLExecContext();
#endif
    /*!
     * Blit the framebuffer to the screen
     * @param viewport ROI to blit
     * @param windowSize The size of the window to blit to
     * @param stretch if true stretch the framebuffer to window size
     */
    void blitFrameBufferToScreen(const cv::Rect& viewport, const cv::Size& windowSize,
            bool stretch = false);

    void toGLTexture2D(cv::UMat& u, cv::ogl::Texture2D& texture);
    void fromGLTexture2D(const cv::ogl::Texture2D& texture, cv::UMat& u);
public:
    /*!
     * Acquires and releases the framebuffer from and to OpenGL.
     */
    class FrameBufferScope {
        FrameBufferContext& ctx_;
        cv::UMat& m_;
    public:
#ifdef __EMSCRIPTEN__
    static void glacquire(FrameBufferContext* ctx, cv::UMat* m) {
        ctx->acquireFromGL(*m);
    }

    static void glrelease(FrameBufferContext* ctx, cv::UMat* m) {
        ctx->releaseToGL(*m);
    }
#endif
        /*!
         * Aquires the framebuffer via cl-gl sharing.
         * @param ctx The corresponding #FrameBufferContext.
         * @param m The UMat to bind the OpenGL framebuffer to.
         */
        FrameBufferScope(FrameBufferContext& ctx, cv::UMat& m) :
                ctx_(ctx), m_(m) {
#ifdef __EMSCRIPTEN__
            emscripten_sync_run_in_main_runtime_thread(EM_FUNC_SIG_VII, glacquire, &ctx_, &m_);
#else
            ctx_.acquireFromGL(m_);
#endif
        }
        /*!
         * Releases the framebuffer via cl-gl sharing.
         */
        ~FrameBufferScope() {
#ifdef __EMSCRIPTEN__
            emscripten_sync_run_in_main_runtime_thread(EM_FUNC_SIG_VII, glrelease, &ctx_, &m_);
#else
            ctx_.releaseToGL(m_);
#endif
        }
    };

#ifdef __EMSCRIPTEN__
    static void glbegin(FrameBufferContext* ctx) {
        ctx->begin();
    }

    static void glend(FrameBufferContext* ctx) {
        ctx->end();
    }
#endif

    /*!
     * Setups and tears-down OpenGL states.
     */
    class GLScope {
        FrameBufferContext& ctx_;
    public:
        /*!
         * Setup OpenGL states.
         * @param ctx The corresponding #FrameBufferContext.
         */
        GLScope(FrameBufferContext& ctx) :
                ctx_(ctx) {
#ifdef __EMSCRIPTEN__
            emscripten_sync_run_in_main_runtime_thread(EM_FUNC_SIG_VI, glbegin, &ctx_);
#else
            ctx_.begin();
#endif
        }
        /*!
         * Tear-down OpenGL states.
         */
        ~GLScope() {
#ifdef __EMSCRIPTEN__
            emscripten_sync_run_in_main_runtime_thread(EM_FUNC_SIG_VI, glend, &ctx_);
#else
            ctx_.end();
#endif
        }
    };

    /*!
     * Create a FrameBufferContext with given size.
     * @param frameBufferSize The frame buffer size.
     */
    FrameBufferContext(V4D& v4d, const cv::Size& frameBufferSize, bool offscreen,
            const string& title, int major, int minor, bool compat, int samples, bool debug, GLFWwindow* sharedWindow, const FrameBufferContext* parent);

    FrameBufferContext(V4D& v4d, const string& title, const FrameBufferContext& other);

    /*!
     * Default destructor.
     */
    virtual ~FrameBufferContext();

    void init();
    void setup(const cv::Size& sz);
    void teardown();
    /*!
     * Get the framebuffer size.
     * @return The framebuffer size.
     */
    cv::Size getSize();
    /*!
      * Execute function object fn inside a framebuffer context.
      * The context acquires the framebuffer from OpenGL (either by up-/download or by cl-gl sharing)
      * and provides it to the functon object. This is a good place to use OpenCL
      * directly on the framebuffer.
      * @param fn A function object that is passed the framebuffer to be read/manipulated.
      */
    void execute(std::function<void(cv::UMat&)> fn);

    /*!
     * Get the pixel ratio of the display x-axis.
     * @return The pixel ratio of the display x-axis.
     */
    CV_EXPORTS float getXPixelRatio();
    /*!
     * Get the pixel ratio of the display y-axis.
     * @return The pixel ratio of the display y-axis.
     */
    CV_EXPORTS float getYPixelRatio();
    CV_EXPORTS void makeCurrent();
    CV_EXPORTS void makeNoneCurrent();
    CV_EXPORTS bool isResizable();
    CV_EXPORTS void setResizable(bool r);
    /*!
     * To make it possible for other V4D objects to become current all other
     * V4D instances have to become non-current.
     */
    CV_EXPORTS void setWindowSize(const cv::Size& sz);
    CV_EXPORTS cv::Size getWindowSize();
    CV_EXPORTS void resizeWindow(const cv::Size& sz);
    CV_EXPORTS bool isFullscreen();
    CV_EXPORTS void setFullscreen(bool f);
    CV_EXPORTS cv::Size getNativeFrameBufferSize();
    CV_EXPORTS void setVisible(bool v);
    CV_EXPORTS bool isVisible();

protected:
    /*!
     * Setup OpenGL states.
     */
    void begin();
    /*!
     * Tear-down OpenGL states.
     */
    void end();
    /*!
     * Download the framebuffer to UMat m.
     * @param m The target UMat.
     */
    void download(cv::UMat& m);
    /*!
     * Uploat UMat m to the framebuffer.
     * @param m The UMat to upload.
     */
    void upload(const cv::UMat& m);
    /*!
     * Acquire the framebuffer using cl-gl sharing.
     * @param m The UMat the framebuffer will be bound to.
     */
    void acquireFromGL(cv::UMat& m);
    /*!
     * Release the framebuffer using cl-gl sharing.
     * @param m The UMat the framebuffer is bound to.
     */
    void releaseToGL(cv::UMat& m);
    /*!
     * The UMat used to copy or bind (depending on cl-gl sharing capability) the OpenGL framebuffer.
     */
    cv::UMat frameBuffer_;
    cv::UMat tmpBuffer_;
    /*!
     * The texture bound to the OpenGL framebuffer.
     */
    cv::ogl::Texture2D* texture_ = nullptr;
};
}
}
}

#endif /* SRC_OPENCV_FRAMEBUFFERCONTEXT_HPP_ */
