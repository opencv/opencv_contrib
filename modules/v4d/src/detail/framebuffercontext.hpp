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
#  include <CL/cl.h>
#  include <CL/cl_gl.h>
#else
#  define OPENCV_V4D_USE_ES2 1
#endif

#ifndef OPENCV_V4D_USE_ES2
#  include <GL/glew.h>
#  define GLFW_INCLUDE_GLCOREARB
#else
#  define GLFW_INCLUDE_ES3
#  define GLFW_INCLUDE_GLEXT
#endif

#include <GLFW/glfw3.h>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/opengl.hpp>
#include <iostream>

#include "opencv2/v4d/util.hpp"

namespace cv {
namespace viz {
class V4D;
namespace detail {
typedef cv::ocl::OpenCLExecutionContext CLExecContext_t;
typedef cv::ocl::OpenCLExecutionContextScope CLExecScope_t;

/*!
 * The FrameBufferContext acquires the framebuffer from OpenGL (either by up-/download or by cl-gl sharing)
 */
class FrameBufferContext {
    friend class CLVAContext;
    friend class NanoVGContext;
    friend class cv::viz::V4D;
    bool clglSharing_ = true;
    GLuint frameBufferID_ = 0;
    GLuint textureID_ = 0;
    GLuint renderBufferID_ = 0;
    GLint viewport_[4];
    cl_mem clImage_ = nullptr;
#ifndef __EMSCRIPTEN__
    CLExecContext_t context_;
#endif
    cv::Size frameBufferSize_;
    /*!
     * The internal framebuffer exposed as OpenGL Texture2D.
     * @return The texture object.
     */
    cv::ogl::Texture2D& getTexture2D();
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
        /*!
         * Aquires the framebuffer via cl-gl sharing.
         * @param ctx The corresponding #FrameBufferContext.
         * @param m The UMat to bind the OpenGL framebuffer to.
         */
        FrameBufferScope(FrameBufferContext& ctx, cv::UMat& m) :
                ctx_(ctx), m_(m) {
            ctx_.acquireFromGL(m_);
        }
        /*!
         * Releases the framebuffer via cl-gl sharing.
         */
        ~FrameBufferScope() {
            ctx_.releaseToGL(m_);
        }
    };

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
            ctx_.begin();
        }
        /*!
         * Tear-down OpenGL states.
         */
        ~GLScope() {
            ctx_.end();
        }
    };

    /*!
     * Create a FrameBufferContext with given size.
     * @param frameBufferSize The frame buffer size.
     */
    FrameBufferContext(const cv::Size& frameBufferSize);
    /*!
     * Default destructor.
     */
    virtual ~FrameBufferContext();
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
    /*!
     * The texture bound to the OpenGL framebuffer.
     */
    cv::ogl::Texture2D* texture_ = nullptr;
};
}
}
}

#endif /* SRC_OPENCV_FRAMEBUFFERCONTEXT_HPP_ */
