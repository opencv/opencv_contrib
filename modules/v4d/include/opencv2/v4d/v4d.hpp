// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_V4D_V4D_HPP_
#define SRC_OPENCV_V4D_V4D_HPP_

#ifndef OPENCV_V4D_USE_ES3
#include <glad/glad.h>
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/threading.h>
#endif

#ifdef OPENCV_V4D_USE_ES3
#define GLFW_INCLUDE_ES3
#define GLFW_INCLUDE_GLEXT
#endif

#include <GLFW/glfw3.h>
#include "source.hpp"
#include "sink.hpp"
#include "util.hpp"
#include <filesystem>
#include <iostream>
#include <future>
#include <set>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/v4d/formhelper.hpp>
#include "detail/threadpool.hpp"

using std::cout;
using std::cerr;
using std::endl;
using std::string;

struct GLFWwindow;

namespace nanogui {
    class Widget;
}
/*!
 * OpenCV namespace
 */
namespace cv {
/*!
 * Visualization namespace
 */
namespace v4d {
class FormHelper;
/*!
 * Private namespace
 */
namespace detail {
class FrameBufferContext;
class CLVAContext;
class GLContext;
class NanoVGContext;
class NanoguiContext;

/*!
 * The GFLW error callback.
 * @param error Error number
 * @param description Error description
 */
void glfw_error_callback(int error, const char* description);
/*!
 * Find widgets that are of type T.
 * @tparam T The type of widget to find
 * @param parent The parent widget
 * @param widgets A vector of widgets of type T to append newly found widgets to.
 */
template<typename T> void find_widgets(const nanogui::Widget* parent, std::vector<T>& widgets) {
    T w;
    for (auto* child : parent->children()) {
        find_widgets(child, widgets);
        if ((w = dynamic_cast<T>(child)) != nullptr) {
            widgets.push_back(w);
        }
    }
}
}

/*!
 * Convenience function to check for OpenGL errors. Should only be used via the macro #GL_CHECK.
 * @param file The file path of the error.
 * @param line The file line of the error.
 * @param expression The expression that failed.
 */
void gl_check_error(const std::filesystem::path& file, unsigned int line, const char* expression);
/*!
 * Convenience macro to check for OpenGL errors.
 */
#define GL_CHECK(expr)                            \
    expr;                                        \
    cv::v4d::gl_check_error(__FILE__, __LINE__, #expr);

/*!
 * Convenience function to color convert from Scalar to Scalar
 * @param src The scalar to color convert
 * @param code The color converions code
 * @return The color converted scalar
 */
CV_EXPORTS cv::Scalar colorConvert(const cv::Scalar& src, cv::ColorConversionCodes code);

using namespace cv::v4d::detail;

class NVG;

class CV_EXPORTS V4D {
    friend class detail::NanoVGContext;
    friend class detail::FrameBufferContext;
    cv::Size initialSize_;
    bool offscreen_;
    const string& title_;
    int major_;
    int minor_;
    bool compat_;
    int samples_;
    bool debug_;
    cv::Rect viewport_;
    float scale_;
    cv::Vec2f mousePos_;
    bool stretch_;
    FrameBufferContext* mainFbContext_ = nullptr;
    CLVAContext* clvaContext_ = nullptr;
    GLContext* glContext_ = nullptr;
    NanoVGContext* nvgContext_ = nullptr;
    NanoguiContext* nguiContext_ = nullptr;
    bool closed_ = false;
    bool mouseDrag_ = false;
    Source source_;
    Sink sink_;
    concurrent::threadpool pool_;
    cv::UMat currentReaderFrame_;
    cv::UMat nextReaderFrame_;
    cv::UMat currentWriterFrame_;
    cv::UMat readerFrameBuffer_;
    cv::UMat writerFrameBuffer_;
    std::future<bool> futureReader_;
    std::future<void> futureWriter_;
    std::function<bool(int key, int scancode, int action, int modifiers)> keyEventCb_;
    uint64_t frameCnt_ = 0;
    cv::TickMeter tick_;
    float fps_ = 0;
public:
    /*!
     * Creates a V4D object which is the central object to perform visualizations with.
     * @param initialSize The initial size of the heavy-weight window.
     * @param frameBufferSize The initial size of the framebuffer backing the window (needs to be equal or greate then initial size).
     * @param offscreen Don't create a window and rather render offscreen.
     * @param title The window title.
     * @param major The OpenGL major version to request.
     * @param minor The OpenGL minor version to request.
     * @param compat Request a compatibility context.
     * @param samples MSAA samples.
     * @param debug Create a debug OpenGL context.
     */
    CV_EXPORTS static cv::Ptr<V4D> make(const cv::Size& size, const cv::Size& fbsize, const string& title, bool offscreen = false, bool debug = false, int major = 3,
            int minor = 2, bool compat = false, int samples = 0);
    /*!
     * Default destructor
     */
    CV_EXPORTS virtual ~V4D();
    CV_EXPORTS void init();
    /*!
     * The internal framebuffer exposed as OpenGL Texture2D.
     * @return The texture object.
     */
    CV_EXPORTS cv::ogl::Texture2D& texture();

    /*!
     * Execute function object fn inside an opengl context.
     * This is how all OpenGL operations should be executed.
     * @param fn A function object that is passed the size of the framebuffer
     */
    CV_EXPORTS void gl(std::function<void(const cv::Size&)> fn);
    CV_EXPORTS void gl(std::function<void()> fn);

    /*!
     * Execute function object fn inside a framebuffer context.
     * The context acquires the framebuffer from OpenGL (either by up-/download or by cl-gl sharing)
     * and provides it to the functon object. This is a good place to use OpenCL
     * directly on the framebuffer.
     * @param fn A function object that is passed the framebuffer to be read/manipulated.
     */
    CV_EXPORTS void fb(std::function<void(cv::UMat&)> fn);
    /*!
     * Execute function object fn inside a nanovg context.
     * The context takes care of setting up opengl and nanovg states.
     * A function object passed like that can use the functions in cv::viz::nvg.
     * @param fn A function that is passed the size of the framebuffer
     * and performs drawing using cv::viz::nvg
     */
    CV_EXPORTS void nvg(std::function<void(const cv::Size&)> fn);
    CV_EXPORTS void nvg(std::function<void()> fn);

    /*!
     * Execute function object fn inside a nanogui context.
     * The context provides a #cv::viz::FormHelper instance to the function object
     * which can be used to build a gui.
     * @param fn A function that is passed the size of the framebuffer
     * and performs drawing using cv::viz::nvg.
     */
    CV_EXPORTS void nanogui(std::function<void(FormHelper& form)> fn);
    /*!
     * Execute function object fn in a loop.
     * This function main purpose is to abstract the run loop for portability reasons.
     * @param fn A functor that will be called repeatetly until the application terminates or the functor returns false
     */
    CV_EXPORTS void run(std::function<bool()> fn);

    /*!
     * Clear the framebuffer.
     * @param bgra The color to use for clearing.
     */
    CV_EXPORTS void clear(const cv::Scalar& bgra = cv::Scalar(0, 0, 0, 255));
    /*!
     * Called to feed an image directly to the framebuffer
     */
    CV_EXPORTS void feed(cv::InputArray& in);
    /*!
     * Called to capture to the framebuffer from a #cv::viz::Source object provided via #V4D::setSource().
     * @return true if successful.
     */
    CV_EXPORTS bool capture();

    /*!
     * Called to capture from a function object.
     * The functor fn is passed a UMat which it writes to which in turn is captured to the framebuffer.
     * @param fn The functor that provides the data.
     * @return true if successful-
     */
    CV_EXPORTS bool capture(std::function<void(cv::UMat&)> fn);
    /*!
     * Called to write the framebuffer to a #cv::viz::Sink object provided via #V4D::setSink()
     */
    CV_EXPORTS void write();
    /*!
     * Called to pass the frambuffer to a functor which consumes it (e.g. writes to a video file).
     * @param fn The functor that consumes the data,
     */
    CV_EXPORTS void write(std::function<void(const cv::UMat&)> fn);

    /*!
     * Set the current #cv::viz::Source object. Usually created using #makeCaptureSource().
     * @param src A #cv::viz::Source object.
     */
    CV_EXPORTS void setSource(const Source& src);
    /*!
     * Checks if the current #cv::viz::Source is ready.
     * @return true if it is ready.
     */
    CV_EXPORTS bool isSourceReady();
    /*!
     * Set the current #cv::viz::Sink object. Usually created using #makeWriterSink().
     * @param sink A #cv::viz::Sink object.
     */
    CV_EXPORTS void setSink(const Sink& sink);
    /*!
     * Checks if the current #cv::viz::Sink is ready.
     * @return true if it is ready.
     */
    CV_EXPORTS bool isSinkReady();
    /*!
     * Shows or hides the GUI.
     * @param s if true show the GUI.
     */
    CV_EXPORTS void showGui(bool s);
    /*!
     * if zoomed in, move the content by x and y
     * @param x The amount on the x-axis to move
     * @param y The amount on the y-axis to move
     */
    CV_EXPORTS void pan(int x, int y);
    /*!
     * Zoom by factor.
     * @param factor The zoom factor.
     */
    CV_EXPORTS void zoom(float factor);
    /*!
     * Get the window position.
     * @return The window position.
     */
    CV_EXPORTS cv::Vec2f getPosition();
    /*!
     * Get current zoom scale.
     * @return The zoom scale.
     */
    CV_EXPORTS float getScale();
    /*!
     * Get the current viewport.
     * @return The current viewport.
     */
    CV_EXPORTS cv::Rect& viewport();
    CV_EXPORTS float getXPixelRatio();
    CV_EXPORTS float getYPixelRatio();
    /*!
     * Set the window size.
     * @param sz The new window size.
     */
    CV_EXPORTS void resizeWindow(const cv::Size& sz);
    /*!
     * Get the initial size.
     * @return The initial size.
     */
    CV_EXPORTS cv::Size getInitialSize();
    /*!
     * Get the video frame size
     * @return The current video frame size.
     */
    CV_EXPORTS cv::Size getVideoFrameSize();
    /*!
     * Get the frambuffer size.
     * @return The framebuffer size.
     */
    CV_EXPORTS cv::Size getFrameBufferSize();
    /*!
     * Get the frambuffer size of the native window.
     * @return The framebuffer size of the native window.
     */
    CV_EXPORTS cv::Size getNativeFrameBufferSize();
    /*!
     * Determine if the window is in fullscreen mode.
     * @return true if in fullscreen mode.
     */
    CV_EXPORTS bool isFullscreen();
    /*!
     * Enable or disable fullscreen mode.
     * @param f if true enable fullscreen mode else disable.
     */
    CV_EXPORTS void setFullscreen(bool f);
    /*!
     * Determines if the window is resizeable.
     * @return true if the window is resizeable.
     */
    CV_EXPORTS bool isResizable();
    /*!
     * Set the window resizable.
     * @param r if r is true set the window resizable.
     */
    CV_EXPORTS void setResizable(bool r);
    /*!
     * Determine if the window is visible.
     * @return true if the window is visible.
     */
    CV_EXPORTS bool isVisible();
    /*!
     * Set the window visible or invisible.
     * @param v if v is true set the window visible.
     */
    CV_EXPORTS void setVisible(bool v);
    /*!
     * Determine if offscreen rendering is enabled.
     * @return true if offscreen rendering is enabled.
     */
    CV_EXPORTS bool isOffscreen();
    /*!
     * Enable or disable offscreen rendering.
     * @param o if o is true enable offscreen rendering.
     */
    CV_EXPORTS void setOffscreen(bool o);
    /*!
     * Enable or disable stretching of the framebuffer to window size during blitting.
     * @param s if s is true enable stretching.
     */
    CV_EXPORTS void setStretching(bool s);
    /*!
     * Determine if framebuffer stretching during blitting is enabled.
     * @return true if framebuffer stretching during blitting is enabled.
     */
    CV_EXPORTS bool isStretching();
    CV_EXPORTS uint64_t frameCount();
    /*!
     * Determine if the window is closed.
     * @return true if the window is closed.
     */
    CV_EXPORTS bool isClosed();
    /*!
     * Close the window.
     */
    CV_EXPORTS void close();
    /*!
     * Display the framebuffer in the native window by blitting.
     * @return false if the window is closed.
     */
    CV_EXPORTS bool display();
    CV_EXPORTS void printSystemInfo();
    CV_EXPORTS void updateFps(bool graphical = true);
    FrameBufferContext& fbCtx();
private:
    V4D(const cv::Size& size, const cv::Size& fbsize,
            const string& title, bool offscreen, bool debug, int major, int minor, bool compat, int samples);
    void setDefaultKeyboardEventCallback();
    void setKeyboardEventCallback(
            std::function<bool(int key, int scancode, int action, int modifiers)> fn);
    void setMouseDrag(bool d);
    bool isMouseDrag();
    cv::Vec2f getMousePosition();
    bool keyboard_event(int key, int scancode, int action, int modifiers);
    void setMousePosition(int x, int y);
    CLVAContext& clvaCtx();
    NanoVGContext& nvgCtx();
    NanoguiContext& nguiCtx();
    GLContext& glCtx();

    bool hasFbCtx();
    bool hasClvaCtx();
    bool hasNvgCtx();
    bool hasNguiCtx();
    bool hasGlCtx();

    GLFWwindow* getGLFWWindow();
};
}
} /* namespace kb */

#include <opencv2/v4d/nvg.hpp>

#endif /* SRC_OPENCV_V4D_V4D_HPP_ */
