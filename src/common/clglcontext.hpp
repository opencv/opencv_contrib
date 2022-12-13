#ifndef SRC_COMMON_CLGLCONTEXT_HPP_
#define SRC_COMMON_CLGLCONTEXT_HPP_

#define CL_TARGET_OPENCL_VERSION 120

#include <GL/glew.h>
#include <GL/gl.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/opengl.hpp>
#define GLFW_INCLUDE_GLCOREARB
#include <GLFW/glfw3.h>

#include "util.hpp"

namespace kb {

class Viz2D;

class CLGLContext {
    friend class CLVAContext;
    friend class NanoVGContext;
    friend class Viz2D;
    cv::ogl::Texture2D *frameBufferTex_;
    GLuint frameBufferID;
    GLuint renderBufferID;
    GLint viewport_[4];
    CLExecContext_t context_;
    cv::Size frameBufferSize_;
    cv::ogl::Texture2D& getTexture2D();
    CLExecContext_t& getCLExecContext();
    void blitFrameBufferToScreen(const cv::Size& size);
public:
    class FrameBufferScope {
        CLGLContext& ctx_;
        cv::UMat& m_;
    public:
        FrameBufferScope(CLGLContext& ctx, cv::UMat& m) : ctx_(ctx), m_(m) {
            ctx_.acquireFromGL(m_);
        }

        ~FrameBufferScope() {
            ctx_.releaseToGL(m_);
        }
    };

    class GLScope {
        CLGLContext& ctx_;
    public:
        GLScope(CLGLContext& ctx) : ctx_(ctx) {
            ctx_.begin();
        }

        ~GLScope() {
            ctx_.end();
        }
    };

    CLGLContext(const cv::Size& frameBufferSize);
    virtual ~CLGLContext();
    cv::ogl::Texture2D& getFrameBufferTexture();
    cv::Size getSize();
    void opencl(std::function<void(cv::UMat&)> fn);
protected:
    void begin();
    void end();
    void acquireFromGL(cv::UMat &m);
    void releaseToGL(cv::UMat &m);
    cv::UMat frameBuffer_;
};
} /* namespace kb */

#endif /* SRC_COMMON_CLGLCONTEXT_HPP_ */
