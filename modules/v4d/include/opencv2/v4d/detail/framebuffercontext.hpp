// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_FRAMEBUFFERCONTEXT_HPP_
#define SRC_OPENCV_FRAMEBUFFERCONTEXT_HPP_

#include "cl.hpp"
#include "context.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include "opencv2/v4d/util.hpp"
#include <iostream>
#include <map>
#include <vector>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
typedef unsigned int GLenum;
#define GL_FRAMEBUFFER 0x8D40

namespace cv {
namespace v4d {
class V4D;

namespace detail {
#ifdef HAVE_OPENCL
typedef cv::ocl::OpenCLExecutionContext CLExecContext_t;
class CLExecScope_t
{
    CLExecContext_t ctx_;
public:
    inline CLExecScope_t(const CLExecContext_t& ctx)
    {
        if(ctx.empty())
            return;
        ctx_ = CLExecContext_t::getCurrentRef();
        ctx.bind();
    }

    inline ~CLExecScope_t()
    {
        if (!ctx_.empty())
        {
            ctx_.bind();
        }
    }
};
#else
struct CLExecContext_t {
	bool empty() {
		return true;
	}
	static CLExecContext_t getCurrent() {
		return CLExecContext_t();
	}
};
class CLExecScope_t
{
    CLExecContext_t ctx_;
public:
    inline CLExecScope_t(const CLExecContext_t& ctx)
    {
    }

    inline ~CLExecScope_t()
    {
    }
};
#endif
/*!
 * The FrameBufferContext acquires the framebuffer from OpenGL (either by up-/download or by cl-gl sharing)
 */
class CV_EXPORTS FrameBufferContext : public V4DContext {
    typedef unsigned int GLuint;
    typedef signed int GLint;

    friend class SourceContext;
    friend class SinkContext;
    friend class GLContext;
    friend class NanoVGContext;
    friend class ImGuiContextImpl;
    friend class cv::v4d::V4D;
    cv::Ptr<FrameBufferContext> self_ = this;
    V4D* v4d_ = nullptr;
    bool offscreen_;
    string title_;
    int major_;
    int minor_;
    int samples_;
    bool debug_;
    GLFWwindow* glfwWindow_ = nullptr;
    bool clglSharing_ = true;
    bool isVisible_;
    GLuint onscreenTextureID_ = 0;
    GLuint onscreenRenderBufferID_ = 0;
    GLuint frameBufferID_ = 0;
    GLuint textureID_ = 0;
    GLuint renderBufferID_ = 0;
    GLint viewport_[4];
    cl_mem clImage_ = nullptr;
    CLExecContext_t context_;
    const cv::Size framebufferSize_;
    bool hasParent_ = false;
    GLFWwindow* rootWindow_;
    cv::Ptr<FrameBufferContext> parent_;
    bool isRoot_ = true;

    //data and handles for webgl copying
    std::map<size_t, GLint> texture_hdls_;
    std::map<size_t, GLint> resolution_hdls_;

    std::map<size_t, GLuint> shader_program_hdls_;

    //gl object maps
    std::map<size_t, GLuint> copyVaos, copyVbos, copyEbos;

    // vertex position, color
    const float copyVertices[12] = {
    //    x      y      z
    -1.0f, -1.0f, -0.0f,
    1.0f, 1.0f, -0.0f,
    -1.0f, 1.0f, -0.0f,
    1.0f, -1.0f, -0.0f };

    const unsigned int copyIndices[6] = {
    //  2---,1
    //  | .' |
    //  0'---3
            0, 1, 2, 0, 3, 1 };

    std::map<size_t, GLuint> copyFramebuffers_;
    std::map<size_t, GLuint> copyTextures_;
    int index_;

    void* currentSyncObject_ = 0;
    static bool firstSync_;
public:
    /*!
     * Acquires and releases the framebuffer from and to OpenGL.
     */
    class CV_EXPORTS FrameBufferScope {
    	cv::Ptr<FrameBufferContext> ctx_;
        cv::UMat& m_;
#ifdef HAVE_OPENCL
        std::shared_ptr<CLExecContext_t> pExecCtx;
#endif
    public:
        /*!
         * Aquires the framebuffer via cl-gl sharing.
         * @param ctx The corresponding #FrameBufferContext.
         * @param m The UMat to bind the OpenGL framebuffer to.
         */
        CV_EXPORTS FrameBufferScope(cv::Ptr<FrameBufferContext> ctx, cv::UMat& m) :
                ctx_(ctx), m_(m)
#ifdef HAVE_OPENCL
        , pExecCtx(std::static_pointer_cast<CLExecContext_t>(m.u->allocatorContext))
#endif
        {
            CV_Assert(!m.empty());
#ifdef HAVE_OPENCL
            if(pExecCtx) {
                CLExecScope_t execScope(*pExecCtx.get());
                ctx_->acquireFromGL(m_);
            } else {
#endif
                ctx_->acquireFromGL(m_);
#ifdef HAVE_OPENCL
            }
#endif
        }
        /*!
         * Releases the framebuffer via cl-gl sharing.
         */
        CV_EXPORTS virtual ~FrameBufferScope() {
#ifdef HAVE_OPENCL
            if (pExecCtx) {
                CLExecScope_t execScope(*pExecCtx.get());
                ctx_->releaseToGL(m_);
            }
            else {
#endif
                ctx_->releaseToGL(m_);
#ifdef HAVE_OPENCL
            }
#endif
        }
    };

    /*!
     * Setups and tears-down OpenGL states.
     */
    class CV_EXPORTS GLScope {
    	cv::Ptr<FrameBufferContext> ctx_;
    public:
        /*!
         * Setup OpenGL states.
         * @param ctx The corresponding #FrameBufferContext.
         */
        CV_EXPORTS GLScope(cv::Ptr<FrameBufferContext> ctx, GLenum framebufferTarget = GL_FRAMEBUFFER) :
                ctx_(ctx) {
            ctx_->begin(framebufferTarget);
        }
        /*!
         * Tear-down OpenGL states.
         */
        CV_EXPORTS ~GLScope() {
            ctx_->end();
        }
    };

    /*!
     * Create a FrameBufferContext with given size.
     * @param frameBufferSize The frame buffer size.
     */
    FrameBufferContext(V4D& v4d, const cv::Size& frameBufferSize, bool offscreen,
            const string& title, int major, int minor, int samples, bool debug, GLFWwindow* rootWindow, cv::Ptr<FrameBufferContext> parent, bool root);

    FrameBufferContext(V4D& v4d, const string& title, cv::Ptr<FrameBufferContext> other);

    /*!
     * Default destructor.
     */
    virtual ~FrameBufferContext();

    cv::Ptr<FrameBufferContext> self() {
    	return self_;
    }

    GLuint getFramebufferID();
    GLuint getTextureID();
    /*!
     * Get the framebuffer size.
     * @return The framebuffer size.
     */
    const cv::Size& size() const;
    void copyTo(cv::UMat& dst);
    void copyFrom(const cv::UMat& src);
    void copyToRootWindow();

    /*!
      * Execute function object fn inside a framebuffer context.
      * The context acquires the framebuffer from OpenGL (either by up-/download or by cl-gl sharing)
      * and provides it to the functon object. This is a good place to use OpenCL
      * directly on the framebuffer.
      * @param fn A function object that is passed the framebuffer to be read/manipulated.
      */
    virtual void execute(std::function<void()> fn) override {
		if(!getCLExecContext().empty()) {
			CLExecScope_t clExecScope(getCLExecContext());
			FrameBufferContext::GLScope glScope(self(), GL_FRAMEBUFFER);
			FrameBufferContext::FrameBufferScope fbScope(self(), framebuffer_);
			fn();
		} else {
			FrameBufferContext::GLScope glScope(self(), GL_FRAMEBUFFER);
			FrameBufferContext::FrameBufferScope fbScope(self(), framebuffer_);
			fn();
		}
    }
    cv::Vec2f position();
    float pixelRatioX();
    float pixelRatioY();
    void makeCurrent();
    void makeNoneCurrent();
    bool isResizable();
    void setResizable(bool r);
    void setWindowSize(const cv::Size& sz);
    cv::Size getWindowSize();
    bool isFullscreen();
    void setFullscreen(bool f);
    cv::Size getNativeFrameBufferSize();
    void setVisible(bool v);
    bool isVisible();
    void close();
    bool isClosed();
    bool isRoot();
    bool hasParent();
    bool hasRootWindow();

    /*!
     * Blit the framebuffer to the screen
     * @param viewport ROI to blit
     * @param windowSize The size of the window to blit to
     * @param stretch if true stretch the framebuffer to window size
     */
    void blitFrameBufferToFrameBuffer(const cv::Rect& srcViewport, const cv::Size& targetFbSize,
            GLuint targetFramebufferID = 0, bool stretch = true, bool flipY = false);
protected:
    void fence();
    bool wait(const uint64_t& timeout = 0);
    CLExecContext_t& getCLExecContext();
    cv::Ptr<V4D> getV4D();
    int getIndex();
    void setup();
    void teardown();
    /*!
     * The UMat used to copy or bind (depending on cl-gl interop capability) the OpenGL framebuffer.
     */
    /*!
     * The internal framebuffer exposed as OpenGL Texture2D.
     * @return The texture object.
     */
    cv::ogl::Texture2D& getTexture2D();

    GLFWwindow* getGLFWWindow() const;
private:
    void loadBuffers(const size_t& index);
    void loadShader(const size_t& index);
    void init();
    CV_EXPORTS cv::UMat& fb();
    /*!
     * Setup OpenGL states.
     */
    CV_EXPORTS void begin(GLenum framebufferTarget);
    /*!
     * Tear-down OpenGL states.
     */
    CV_EXPORTS void end();
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
    void toGLTexture2D(cv::UMat& u, cv::ogl::Texture2D& texture);
    void fromGLTexture2D(const cv::ogl::Texture2D& texture, cv::UMat& u);

    cv::UMat framebuffer_;
    /*!
     * The texture bound to the OpenGL framebuffer.
     */
    cv::ogl::Texture2D* texture_ = nullptr;
};
}
}
}

#endif /* SRC_OPENCV_FRAMEBUFFERCONTEXT_HPP_ */
