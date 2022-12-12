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

    cv::UMat frameBuffer_;
    cv::ogl::Texture2D *frameBufferTex_;
    GLuint frameBufferID;
    GLuint renderBufferID;
    CLExecContext_t context_;
    cv::Size frameBufferSize_;
public:
    CLGLContext(const cv::Size& frameBufferSize);
    cv::ogl::Texture2D& getFrameBufferTexture();
    cv::Size getSize();
    void render(std::function<void(cv::Size&)> fn);
    void compute(std::function<void(cv::UMat&)> fn);
private:
    cv::ogl::Texture2D& getTexture2D();
    CLExecContext_t& getCLExecContext();
    void blitFrameBufferToScreen(const cv::Size& size);
    void begin();
    void end();
    void acquireFromGL(cv::UMat &m);
    void releaseToGL(cv::UMat &m);
};
} /* namespace kb */

#endif /* SRC_COMMON_CLGLCONTEXT_HPP_ */
