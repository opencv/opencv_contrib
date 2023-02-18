#include "clglcontext.hpp"
#include "../util.hpp"
#include "../viz2d.hpp"

namespace kb {
namespace viz2d {
namespace detail {

//FIXME use cv::ogl
CLGLContext::CLGLContext(const cv::Size& frameBufferSize) :
        frameBufferSize_(frameBufferSize) {
#ifndef __EMSCRIPTEN__
    glewExperimental = true;
    glewInit();
    cv::ogl::ocl::initializeContextFromGL();
#endif
    frameBufferID_ = 0;
    GL_CHECK(glGenFramebuffers(1, &frameBufferID_));
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID_));
    GL_CHECK(glGenRenderbuffers(1, &renderBufferID_));
    textureID_ = 0;
    GL_CHECK(glGenTextures(1, &textureID_));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, textureID_));
    texture_ = new cv::ogl::Texture2D(frameBufferSize_, cv::ogl::Texture2D::RGBA, textureID_);
    GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, frameBufferSize_.width, frameBufferSize_.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0));

    GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, renderBufferID_));
    GL_CHECK(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, frameBufferSize_.width, frameBufferSize_.height));
    GL_CHECK(glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderBufferID_));

    GL_CHECK(glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureID_, 0));
    assert(glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
#ifndef __EMSCRIPTEN__
    context_ = CLExecContext_t::getCurrent();
#endif
}

CLGLContext::~CLGLContext() {
    end();
    glDeleteTextures(1, &textureID_);
    glDeleteRenderbuffers( 1, &renderBufferID_);
    glDeleteFramebuffers( 1, &frameBufferID_);
}

cv::Size CLGLContext::getSize() {
    return frameBufferSize_;
}

void CLGLContext::execute(std::function<void(cv::UMat&)> fn) {
#ifndef __EMSCRIPTEN__
    CLExecScope_t clExecScope(getCLExecContext());
#endif
    CLGLContext::GLScope glScope(*this);
    CLGLContext::FrameBufferScope fbScope(*this, frameBuffer_);
    fn(frameBuffer_);
}

cv::ogl::Texture2D& CLGLContext::getTexture2D() {
    return *texture_;
}

#ifndef __EMSCRIPTEN__
CLExecContext_t& CLGLContext::getCLExecContext() {
    return context_;
}
#endif

void CLGLContext::blitFrameBufferToScreen(const cv::Rect& viewport, const cv::Size& windowSize, bool stretch) {
    cerr << viewport << " " << windowSize << endl;
    GL_CHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, frameBufferID_));
    GL_CHECK(glReadBuffer(GL_COLOR_ATTACHMENT0));
    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
    GL_CHECK(glBlitFramebuffer(viewport.x, viewport.y, viewport.x + viewport.width, viewport.y + viewport.height,
            0, stretch ? 0 : windowSize.height - frameBufferSize_.height, stretch ? windowSize.width : frameBufferSize_.width, windowSize.height, GL_COLOR_BUFFER_BIT, GL_NEAREST));
}

void CLGLContext::begin() {
    GL_CHECK(glGetIntegerv( GL_VIEWPORT, viewport_ ));
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID_));
    GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, renderBufferID_));
    GL_CHECK(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderBufferID_));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, textureID_));
    GL_CHECK(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureID_, 0));
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
}

void CLGLContext::end() {
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));
    GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, 0));
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    //glFlush seems enough but i wanna make sure that there won't be race conditions.
    //At least on TigerLake/Iris it doesn't make a difference in performance.
//    GL_CHECK(glViewport(viewport_[0], viewport_[1], viewport_[2], viewport_[3]));
    GL_CHECK(glFlush());
    GL_CHECK(glFinish());
}

void CLGLContext::download(cv::UMat& m) {
    cv::Mat tmp = m.getMat(cv::ACCESS_RW);
    assert(tmp.data != nullptr);
    GL_CHECK(glReadPixels(0, 0, tmp.cols, tmp.rows, GL_RGBA, GL_UNSIGNED_BYTE, tmp.data));
    tmp.release();
}

void CLGLContext::upload(const cv::UMat& m) {
    cv::Mat tmp = m.getMat(cv::ACCESS_RW);
    assert(tmp.data != nullptr);
    GL_CHECK(glTexSubImage2D(
        GL_TEXTURE_2D,
        0,
        0,
        0,
        tmp.cols,
        tmp.rows,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        tmp.data)
    );
    tmp.release();
}

void CLGLContext::acquireFromGL(cv::UMat &m) {
#ifndef __EMSCRIPTEN__
    GL_CHECK(cv::ogl::convertFromGLTexture2D(getTexture2D(), m));
#else
    if(m.empty())
        m.create(frameBufferSize_, CV_8UC4);
    download(m);
    GL_CHECK(glFlush());
    GL_CHECK(glFinish());
#endif
    //FIXME
    cv::flip(m, m, 0);
}

void CLGLContext::releaseToGL(cv::UMat &m) {
    //FIXME
    cv::flip(m, m, 0);
#ifndef __EMSCRIPTEN__
    GL_CHECK(cv::ogl::convertToGLTexture2D(m, getTexture2D()));
#else
    if(m.empty())
        m.create(frameBufferSize_, CV_8UC4);
    upload(m);
    GL_CHECK(glFlush());
    GL_CHECK(glFinish());
#endif
}

}
}
}
