#include "clglcontext.hpp"
#include "glwindow.hpp"
#include "util.hpp"

namespace kb {
CLGLContext::CLGLContext(cv::Size frameBufferSize) :
    frameBufferSize_(frameBufferSize) {
    glewExperimental = true;
    glewInit();
    cv::ogl::ocl::initializeContextFromGL();
    frameBufferID = 0;
    GL_CHECK(glGenFramebuffers(1, &frameBufferID));
    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frameBufferID));
    GL_CHECK(glGenRenderbuffers(1, &renderBufferID));

    frameBufferTex_ = new cv::ogl::Texture2D(frameBufferSize_, cv::ogl::Texture2D::RGBA, false);
    frameBufferTex_->bind();

    GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, renderBufferID));
    GL_CHECK(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, frameBufferSize_.width, frameBufferSize_.height));
    GL_CHECK(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderBufferID));

    GL_CHECK(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, frameBufferTex_->texId(), 0));
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

    context_ = CLExecContext_t::getCurrentRef();
}

cv::ogl::Texture2D& CLGLContext::getFrameBufferTexture() {
    return *frameBufferTex_;
}

cv::Size CLGLContext::getSize() {
    return frameBufferSize_;
}

void CLGLContext::render(std::function<void(cv::Size&)> fn) {
    CLExecScope_t scope(context_);
    begin();
    fn(frameBufferSize_);
    end();
}


void CLGLContext::compute(std::function<void(cv::UMat&)> fn) {
    CLExecScope_t scope(getCLExecContext());
    acquireFromGL(frameBuffer_);
    fn(frameBuffer_);
    releaseToGL(frameBuffer_);
}

cv::ogl::Texture2D& CLGLContext::getTexture2D() {
    return *frameBufferTex_;
}

CLExecContext_t& CLGLContext::getCLExecContext() {
    return context_;
}

void CLGLContext::blitFrameBufferToScreen(int x, int y) {
    GL_CHECK(glBindFramebuffer(GL_READ_FRAMEBUFFER, frameBufferID));
    GL_CHECK(glReadBuffer(GL_COLOR_ATTACHMENT0));
    GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
    GL_CHECK(glBlitFramebuffer(0, 0, frameBufferSize_.width, frameBufferSize_.height, x, y, x + frameBufferSize_.width, y + frameBufferSize_.height, GL_COLOR_BUFFER_BIT, GL_NEAREST));
}

void CLGLContext::begin() {
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID));
    GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, renderBufferID));
    GL_CHECK(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderBufferID));
    frameBufferTex_->bind();
}

void CLGLContext::end() {
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));
    GL_CHECK(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, 0));
    GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, 0));
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));

    //glFlush seems enough but i wanna make sure that there won't be race conditions.
    //At least on TigerLake/Iris it doesn't make a difference in performance.
    GL_CHECK(glFlush());
    GL_CHECK(glFinish());
}

void CLGLContext::acquireFromGL(cv::UMat &m) {
    begin();
    GL_CHECK(cv::ogl::convertFromGLTexture2D(getTexture2D(), m));
    //FIXME
    cv::flip(m, m, 0);
}

void CLGLContext::releaseToGL(cv::UMat &m) {
    //FIXME
    cv::flip(m, m, 0);
    GL_CHECK(cv::ogl::convertToGLTexture2D(m, getTexture2D()));
    end();
}

} /* namespace kb */
