// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/v4d.hpp"
#include "opencv2/v4d/detail/framebuffercontext.hpp"
#include <sstream>
#include <algorithm>
#include <opencv2/core.hpp>
#include <vector>

namespace cv {
namespace v4d {

cv::Ptr<V4D> V4D::make(const cv::Size& size, const string& title, AllocateFlags flags, bool offscreen, bool debug, int samples) {
    V4D* v4d = new V4D(size, cv::Size(), title, flags, offscreen, debug, samples);
    v4d->setVisible(!offscreen);
    v4d->fbCtx()->makeCurrent();
    return v4d->self();
}

cv::Ptr<V4D> V4D::make(const cv::Size& size, const cv::Size& fbsize, const string& title, AllocateFlags flags, bool offscreen, bool debug, int samples) {
    V4D* v4d = new V4D(size, fbsize, title, flags, offscreen, debug, samples);
    v4d->setVisible(!offscreen);
    v4d->fbCtx()->makeCurrent();
    return v4d->self();
}

cv::Ptr<V4D> V4D::make(const V4D& other, const string& title) {
    V4D* v4d = new V4D(other, title);
    v4d->setVisible(other.debug_);
    v4d->fbCtx()->makeCurrent();
    return v4d->self();
}

V4D::V4D(const cv::Size& size, const cv::Size& fbsize, const string& title, AllocateFlags aflags, bool offscreen, bool debug, int samples) :
        initialSize_(size), flags_(aflags), debug_(debug), viewport_(0, 0, size.width, size.height), stretching_(true), samples_(samples) {
    self_ = cv::Ptr<V4D>(this);
    mainFbContext_ = new detail::FrameBufferContext(*this, fbsize.empty() ? size : fbsize, offscreen, title, 3,
                2, samples, debug, nullptr, nullptr, true);
    CLExecScope_t scope(mainFbContext_->getCLExecContext());
    if(flags() & NANOVG)
        nvgContext_ = new detail::NanoVGContext(mainFbContext_);
    sourceContext_ = new detail::SourceContext(mainFbContext_);
    sinkContext_ = new detail::SinkContext(mainFbContext_);
    onceContext_ = new detail::OnceContext();
    plainContext_ = new detail::PlainContext();
    if(flags() & IMGUI)
        imguiContext_ = new detail::ImGuiContextImpl(mainFbContext_);

    //preallocate the primary gl context
    glCtx(-1);
}

V4D::V4D(const V4D& other, const string& title) :
        initialSize_(other.initialSize_), flags_(other.flags_), debug_(other.debug_), viewport_(0, 0, other.fbSize().width, other.fbSize().height), stretching_(other.stretching_), samples_(other.samples_) {
	workerIdx_ = Global::next_worker_idx();
    self_ = cv::Ptr<V4D>(this);
    mainFbContext_ = new detail::FrameBufferContext(*this, other.fbSize(), !other.debug_, title, 3,
                2, other.samples_, other.debug_, other.fbCtx()->rootWindow_, other.fbCtx(), true);

    CLExecScope_t scope(mainFbContext_->getCLExecContext());
    if(flags() & NANOVG)
    	nvgContext_ = new detail::NanoVGContext(mainFbContext_);
    sourceContext_ = new detail::SourceContext(mainFbContext_);
    sinkContext_ = new detail::SinkContext(mainFbContext_);
    onceContext_ = new detail::OnceContext();
    plainContext_ = new detail::PlainContext();

    //preallocate the primary gl context
    glCtx(-1);
}

V4D::~V4D() {

}

const int32_t& V4D::workerIndex() const {
	return workerIdx_;
}

size_t V4D::workers_running() {
	return Global::workers_started();
}

cv::ogl::Texture2D& V4D::texture() {
    return mainFbContext_->getTexture2D();
}

std::string V4D::title() const {
    return fbCtx()->title_;
}

cv::Point2f V4D::getMousePosition() {
    return mousePos_;
}

void V4D::setMousePosition(const cv::Point2f& pt) {
    mousePos_ = pt;
}

cv::Ptr<FrameBufferContext> V4D::fbCtx() const {
    assert(mainFbContext_ != nullptr);
    return mainFbContext_;
}

cv::Ptr<SourceContext> V4D::sourceCtx() {
    assert(sourceContext_ != nullptr);
    return sourceContext_;
}

cv::Ptr<SinkContext> V4D::sinkCtx() {
    assert(sinkContext_ != nullptr);
    return sinkContext_;
}

cv::Ptr<NanoVGContext> V4D::nvgCtx() {
    assert(nvgContext_ != nullptr);
    return nvgContext_;
}

cv::Ptr<OnceContext> V4D::onceCtx() {
    assert(onceContext_ != nullptr);
    return onceContext_;
}

cv::Ptr<PlainContext> V4D::plainCtx() {
    assert(plainContext_ != nullptr);
    return plainContext_;
}

cv::Ptr<ImGuiContextImpl> V4D::imguiCtx() {
    assert(imguiContext_ != nullptr);
    return imguiContext_;
}

cv::Ptr<GLContext> V4D::glCtx(int32_t idx) {
    auto it = glContexts_.find(idx);
    if(it != glContexts_.end())
        return (*it).second;
    else {
        cv::Ptr<GLContext> ctx = new GLContext(idx, mainFbContext_);
        glContexts_.insert({idx, ctx});
        return ctx;
    }
}

bool V4D::hasFbCtx() {
    return mainFbContext_ != nullptr;
}

bool V4D::hasSourceCtx() {
    return sourceContext_ != nullptr;
}

bool V4D::hasSinkCtx() {
    return sinkContext_ != nullptr;
}

bool V4D::hasNvgCtx() {
    return nvgContext_ != nullptr;
}

bool V4D::hasOnceCtx() {
    return onceContext_ != nullptr;
}

bool V4D::hasParallelCtx() {
    return plainContext_ != nullptr;
}

bool V4D::hasImguiCtx() {
    return imguiContext_ != nullptr;
}

bool V4D::hasGlCtx(uint32_t idx) {
    return glContexts_.find(idx) != glContexts_.end();
}

size_t V4D::numGlCtx() {
    return std::max(off_t(0), off_t(glContexts_.size()) - 1);
}

void V4D::copyTo(cv::UMat& m) {
	fbCtx()->copyTo(m);
}

void V4D::copyFrom(const cv::UMat& m) {
	fbCtx()->copyFrom(m);
}

void V4D::setSource(cv::Ptr<Source> src) {
    source_ = src;
}
cv::Ptr<Source> V4D::getSource() {
    return source_;
}

bool V4D::hasSource() {
    return source_ != nullptr;
}

void V4D::feed(cv::UMat& in) {
	static thread_local cv::UMat frame;

	plain([](cv::UMat& src, cv::UMat& f, const cv::Size sz) {
		cv::UMat rgb;

		resizePreserveAspectRatio(src, rgb, sz);
		cv::cvtColor(rgb, f, cv::COLOR_RGB2BGRA);
	}, in, frame, mainFbContext_->size());

    fb([](cv::UMat& frameBuffer, const cv::UMat& f) {
        f.copyTo(frameBuffer);
    }, frame);
}

cv::UMat V4D::fetch() {
	static thread_local cv::UMat frame;
	fb([](const cv::UMat& fb, cv::UMat& f) {
		fb.copyTo(f);
	}, frame);
    return frame;
}


void V4D::setSink(cv::Ptr<Sink> sink) {
    sink_ = sink;
}

cv::Ptr<Sink> V4D::getSink() {
    return sink_;
}

bool V4D::hasSink() {
    return sink_ != nullptr;
}

cv::Vec2f V4D::position() {
    return fbCtx()->position();
}

cv::Rect& V4D::viewport() {
    return viewport_;
}

float V4D::pixelRatioX() {
    return fbCtx()->pixelRatioX();
}

float V4D::pixelRatioY() {
    return fbCtx()->pixelRatioY();
}

const cv::Size& V4D::fbSize() const {
    return fbCtx()->size();
}

const cv::Size& V4D::initialSize() const {
    return initialSize_;
}

cv::Size V4D::size() {
    return fbCtx()->getWindowSize();
}

void V4D::setSize(const cv::Size& sz) {
    fbCtx()->setWindowSize(sz);
}

void V4D::setShowFPS(bool s) {
    showFPS_ = s;
}

bool V4D::getShowFPS() {
    return showFPS_;
}

void V4D::setPrintFPS(bool p) {
    printFPS_ = p;
}

bool V4D::getPrintFPS() {
    return printFPS_;
}

void V4D::setShowTracking(bool st) {
    showTracking_ = st;
}

void V4D::setDisableIO(bool d) {
	disableIO_ = d;
}

bool V4D::getShowTracking() {
    return showTracking_;
}

bool V4D::isFullscreen() {
    return fbCtx()->isFullscreen();
}

void V4D::setFullscreen(bool f) {
    fbCtx()->setFullscreen(f);
}

bool V4D::isResizable() {
    return fbCtx()->isResizable();
}

void V4D::setResizable(bool r) {
    fbCtx()->setResizable(r);
}

bool V4D::isVisible() {
    return fbCtx()->isVisible();
}

void V4D::setVisible(bool v) {
    fbCtx()->setVisible(v);
}

void V4D::setStretching(bool s) {
    stretching_ = s;
}

bool V4D::isStretching() {
    return stretching_;
}

void V4D::setFocused(bool f) {
    focused_ = f;
}

bool V4D::isFocused() {
    return focused_;
}

void V4D::swapContextBuffers() {
	{
        FrameBufferContext::GLScope glScope(glCtx(-1)->fbCtx(), GL_READ_FRAMEBUFFER);
        glCtx(-1)->fbCtx()->blitFrameBufferToFrameBuffer(viewport(), glCtx(-1)->fbCtx()->getWindowSize(), 0, isStretching());
//        GL_CHECK(glFinish());
        glfwSwapBuffers(glCtx(-1)->fbCtx()->getGLFWWindow());
	}

    for(size_t i = 0; i < numGlCtx(); ++i) {
        FrameBufferContext::GLScope glScope(glCtx(i)->fbCtx(), GL_READ_FRAMEBUFFER);
        glCtx(i)->fbCtx()->blitFrameBufferToFrameBuffer(viewport(), glCtx(i)->fbCtx()->getWindowSize(), 0, isStretching());
//        GL_CHECK(glFinish());
        glfwSwapBuffers(glCtx(i)->fbCtx()->getGLFWWindow());
    }

    if(hasNvgCtx()) {
		FrameBufferContext::GLScope glScope(nvgCtx()->fbCtx(), GL_READ_FRAMEBUFFER);
		nvgCtx()->fbCtx()->blitFrameBufferToFrameBuffer(viewport(), nvgCtx()->fbCtx()->getWindowSize(), 0, isStretching());
//        GL_CHECK(glFinish());
		glfwSwapBuffers(nvgCtx()->fbCtx()->getGLFWWindow());
    }
}

bool V4D::display() {
    bool result = true;
    if(!Global::is_main())
    	Global::next_frame_cnt();

	if(debug_) {
		swapContextBuffers();
	}
	if (Global::is_main()) {
		auto start = Global::start_time();
		auto now = get_epoch_nanos();
		auto diff = now - start;
		double diffSeconds = diff / 1000000000.0;

		if(Global::fps() > 0 && diffSeconds > 1.0) {
			Global::add_to_start_time(diff / 2.0);
			Global::mul_frame_cnt(0.5);
		} else {
			Global::set_fps((Global::fps() * 3.0 + (Global::frame_cnt() / diffSeconds)) / 4.0);
		}

		if(getPrintFPS())
			cerr << "\rFPS:" << Global::fps() << endl;
		{
			FrameBufferContext::GLScope glScope(fbCtx(), GL_READ_FRAMEBUFFER);
			fbCtx()->blitFrameBufferToFrameBuffer(viewport(), fbCtx()->getWindowSize(), 0, isStretching());
		}

		if(hasImguiCtx())
			imguiCtx()->render(getShowFPS());

		glfwSwapBuffers(fbCtx()->getGLFWWindow());
		glfwPollEvents();
	} else {
		fbCtx()->copyToRootWindow();
	}

	result = !glfwWindowShouldClose(getGLFWWindow());

//FIXME doesn't have any effect
//	GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
//#if !defined(OPENCV_V4D_USE_ES3)
//	GL_CHECK(glDrawBuffer(GL_BACK));
//#endif
//	GL_CHECK(glViewport(0, 0, size().width, size().height));
//	GL_CHECK(glClearColor(1,0,0,255));
//	GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));

    if (frameCnt_ == (std::numeric_limits<uint64_t>().max() - 1))
        frameCnt_ = 0;
    else
        ++frameCnt_;

    return result;
}

const uint64_t& V4D::frameCount() const {
    return frameCnt_;
}

bool V4D::isClosed() {
    return fbCtx()->isClosed();
}

void V4D::close() {
    fbCtx()->close();
}

GLFWwindow* V4D::getGLFWWindow() const {
    return fbCtx()->getGLFWWindow();
}

void V4D::printSystemInfo() {
	cerr << "OpenGL: " << getGlInfo() << endl;
#ifdef HAVE_OPENCL
	if(cv::ocl::useOpenCL())
		cerr << "OpenCL Platforms: " << getClInfo() << endl;
#endif
}

AllocateFlags V4D::flags() {
	return flags_;
}

cv::Ptr<V4D> V4D::self() {
       return self_;
}


}
}
