// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_V4D_V4D_HPP_
#define SRC_OPENCV_V4D_V4D_HPP_

#if !defined(__EMSCRIPTEN__) && !defined(OPENCV_V4D_USE_ES3)
#include <glad/glad.h>
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/threading.h>
#endif

#if defined(__EMSCRIPTEN__) || defined(OPENCV_V4D_USE_ES3)
#define GLFW_INCLUDE_ES3
#define GLFW_INCLUDE_GLEXT
#endif

#include <GLFW/glfw3.h>

#include "source.hpp"
#include "sink.hpp"
#include "util.hpp"
#include "formhelper.hpp"
#include "nvg.hpp"
#include "detail/threadpool.hpp"

#include <iostream>
#include <future>
#include <set>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>


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
 * V4D namespace
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
 * Convenience function to color convert from Scalar to Scalar
 * @param src The scalar to color convert
 * @param code The color converions code
 * @return The color converted scalar
 */
CV_EXPORTS cv::Scalar colorConvert(const cv::Scalar& src, cv::ColorConversionCodes code);

using namespace cv::v4d::detail;

class CV_EXPORTS V4D {
    friend class detail::FrameBufferContext;
    friend class HTML5Capture;
    cv::Size initialSize_;
    const string& title_;
    bool compat_;
    int samples_;
    bool debug_;
    cv::Rect viewport_;
    float zoomScale_;
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
    std::future<bool> futureReader_;
    std::future<void> futureWriter_;
    std::function<bool(int key, int scancode, int action, int modifiers)> keyEventCb_;
    uint64_t frameCnt_ = 0;
    bool showFPS_ = true;
    bool printFPS_ = true;
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
    CV_EXPORTS static cv::Ptr<V4D> make(const cv::Size& size, const cv::Size& fbsize, const string& title, bool offscreen = false, bool debug = false, bool compat = false, int samples = 0);
    /*!
     * Default destructor
     */
    CV_EXPORTS virtual ~V4D();
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
     * Copy the framebuffer contents to an OutputArray.
     * @param arr The array to copy to.
     */
    CV_EXPORTS void copyTo(cv::OutputArray arr);
    /*!
     * Copy the InputArray contents to the framebuffer.
     * @param arr The array to copy.
     */
    CV_EXPORTS void copyFrom(cv::InputArray arr);
    /*!
     * Execute function object fn in a loop.
     * This function main purpose is to abstract the run loop for portability reasons.
     * @param fn A functor that will be called repeatetly until the application terminates or the functor returns false
     */
    CV_EXPORTS void run(std::function<bool()> fn);
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
    CV_EXPORTS cv::Vec2f position();
    /*!
     * Get current zoom scale.
     * @return The zoom scale.
     */
    CV_EXPORTS float zoomScale();
    /*!
     * Get the current viewport.
     * @return The current viewport.
     */
    CV_EXPORTS cv::Rect& viewport();
    /*!
     * Get the pixel ratio of the display x-axis.
     * @return The pixel ratio of the display x-axis.
     */
    CV_EXPORTS float pixelRatioX();
    /*!
     * Get the pixel ratio of the display y-axis.
     * @return The pixel ratio of the display y-axis.
     */
    CV_EXPORTS float pixelRatioY();
    /*!
     * Set the window size.
     * @param sz The new window size.
     */
    CV_EXPORTS cv::Size initialSize();
    /*!
     * Get the current size of the window
     * @return The window size
     */
    CV_EXPORTS cv::Size framebufferSize();
    /*!
     * Determine if the window is in fullscreen mode.
     * @return true if in fullscreen mode.
     */
    CV_EXPORTS void setWindowSize(const cv::Size& sz);
    /*!
     * Get the initial size.
     * @return The initial size.
     */
    CV_EXPORTS cv::Size getWindowSize();
    /*!
     * Get the frambuffer size.
     * @return The framebuffer size.
     */

    CV_EXPORTS bool getShowFPS();
    CV_EXPORTS void setShowFPS(bool s);
    CV_EXPORTS bool getPrintFPS();
    CV_EXPORTS void setPrintFPS(bool p);

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
     * Enable/Disable scaling the framebuffer during blitting.
     * @param s if true enable scaling
     */
    CV_EXPORTS void setScaling(bool s);
    /*!
     * Determine if framebuffer is scaled during blitting.
     * @return true if framebuffer is scaled during blitting.
     */
    CV_EXPORTS bool isScaling();
    /*!
     * Everytime a frame is displayed this count is incremented
     * @return the current frame count
     */
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
    /*!
     * Print basic system information to stderr
     */
    CV_EXPORTS void printSystemInfo();

    CV_EXPORTS void makeCurrent();

    void setDefaultKeyboardEventCallback();
    void setKeyboardEventCallback(
            std::function<bool(int key, int scancode, int action, int modifiers)> fn);
private:
    V4D(const cv::Size& size, const cv::Size& fbsize,
            const string& title, bool offscreen, bool debug, bool compat, int samples);

    void init();

    void setMouseDrag(bool d);
    bool isMouseDrag();
    cv::Vec2f getMousePosition();
    bool keyboard_event(int key, int scancode, int action, int modifiers);
    void setMousePosition(int x, int y);

    FrameBufferContext& fbCtx();
    CLVAContext& clvaCtx();
    NanoVGContext& nvgCtx();
    GLContext& glCtx();
    NanoguiContext& nguiCtx();

    bool hasFbCtx();
    bool hasClvaCtx();
    bool hasNvgCtx();
    bool hasNguiCtx();
    bool hasGlCtx();

    GLFWwindow* getGLFWWindow();
    void swapContextBuffers();
};
}
} /* namespace kb */


#endif /* SRC_OPENCV_V4D_V4D_HPP_ */
