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
public:
    class Scope {
        CLGLContext& ctx_;
        cv::UMat& m_;
    public:
        Scope(CLGLContext& ctx, cv::UMat& m) : ctx_(ctx), m_(m) {
            ctx_.acquireFromGL(m_);
        }

        ~Scope() {
            ctx_.releaseToGL(m_);
        }
    };

    CLGLContext(const cv::Size& frameBufferSize);
    virtual ~CLGLContext();
    cv::ogl::Texture2D& getFrameBufferTexture();
    cv::Size getSize();
    void opencl(std::function<void(cv::UMat&)> fn);
private:
    cv::ogl::Texture2D& getTexture2D();
    CLExecContext_t& getCLExecContext();
    void blitFrameBufferToScreen(const cv::Size& size);
    void begin();
    void end();
protected:
    cv::UMat frameBuffer_;
    void acquireFromGL(cv::UMat &m);
    void releaseToGL(cv::UMat &m);
};
} /* namespace kb */

#endif /* SRC_COMMON_CLGLCONTEXT_HPP_ */
