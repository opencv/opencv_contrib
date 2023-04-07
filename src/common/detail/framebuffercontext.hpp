// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_COMMON_CLGLCONTEXT_HPP_
#define SRC_COMMON_CLGLCONTEXT_HPP_

#define CL_TARGET_OPENCL_VERSION 120

#ifndef __EMSCRIPTEN__
#include <GL/glew.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>
#endif
#ifndef __EMSCRIPTEN__
#  define GLFW_INCLUDE_GLCOREARB
#else
#  define GLFW_INCLUDE_ES3
#  define GLFW_INCLUDE_GLEXT
#endif
#include <GLFW/glfw3.h>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/opengl.hpp>
#include <iostream>

#include "../util.hpp"

namespace cv {
namespace viz {
class Viz2D;
namespace detail {
typedef cv::ocl::OpenCLExecutionContext CLExecContext_t;
typedef cv::ocl::OpenCLExecutionContextScope CLExecScope_t;

class FrameBufferContext {
    friend class CLVAContext;
    friend class NanoVGContext;
    friend class cv::viz::Viz2D;
    bool clglSharing_ = true;
    GLuint frameBufferID_ = 0;
    GLuint textureID_ = 0;
    GLuint renderBufferID_ = 0;
    GLint viewport_[4];
#ifndef __EMSCRIPTEN__
    CLExecContext_t context_;
#endif
    cv::Size frameBufferSize_;
    cv::ogl::Texture2D& getTexture2D();
#ifndef __EMSCRIPTEN__
    CLExecContext_t& getCLExecContext();
#endif
    void blitFrameBufferToScreen(const cv::Rect& viewport, const cv::Size& windowSize, bool stretch = false);
public:
    class FrameBufferScope {
        FrameBufferContext& ctx_;
        cv::UMat& m_;
    public:
        FrameBufferScope(FrameBufferContext& ctx, cv::UMat& m) : ctx_(ctx), m_(m) {
            ctx_.acquireFromGL(m_);
        }

        ~FrameBufferScope() {
            ctx_.releaseToGL(m_);
        }
    };

    class GLScope {
        FrameBufferContext& ctx_;
    public:
        GLScope(FrameBufferContext& ctx) : ctx_(ctx) {
            ctx_.begin();
        }

        ~GLScope() {
            ctx_.end();
        }
    };

    FrameBufferContext(const cv::Size& frameBufferSize);
    virtual ~FrameBufferContext();
    cv::Size getSize();
    void execute(std::function<void(cv::UMat&)> fn);
protected:
    void begin();
    void end();
    void download(cv::UMat& m);
    void upload(const cv::UMat& m);
    void acquireFromGL(cv::UMat &m);
    void releaseToGL(cv::UMat &m);
    cv::UMat frameBuffer_;
    cv::ogl::Texture2D* texture_ = nullptr;
};
}
}
}

#endif /* SRC_COMMON_CLGLCONTEXT_HPP_ */
