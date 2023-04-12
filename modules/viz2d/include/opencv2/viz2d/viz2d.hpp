// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_COMMON_VIZ2D_HPP_
#define SRC_COMMON_VIZ2D_HPP_

#include "source.hpp"
#include "sink.hpp"
#include "dialog.hpp"
#include "formhelper.hpp"
#include "util.hpp"
#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#ifdef __EMSCRIPTEN__
#define VIZ2D_USE_ES3 1
#include <emscripten.h>
#endif

#ifndef VIZ2D_USE_ES3
#define NANOGUI_USE_OPENGL
#else
#define NANOGUI_USE_GLES
#define NANOGUI_GLES_VERSION 3
#endif
#include <nanogui/nanogui.h>
#ifndef VIZ2D_USE_ES3
#include <GL/glew.h>
#else
#include <GLES3/gl3.h>
#endif

using std::cout;
using std::cerr;
using std::endl;
using std::string;
/*!
 * OpenCV namespace
 */
namespace cv {
/*!
 * Visualization namespace
 */
namespace viz {
/*!
 * Private namespace
 */
namespace detail {
class FrameBufferContext;
class CLVAContext;
class NanoVGContext;

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
    cv::viz::gl_check_error(__FILE__, __LINE__, #expr);

/*!
 * The GFLW error callback.
 * @param error Error number
 * @param description Error description
 */
void glfw_error_callback(int error, const char* description);
/*!
 * Checks if a widget contains an absolute point.
 * @param w The widget.
 * @param p The point.
 * @return true if the points is inside the widget
 */
bool contains_absolute(nanogui::Widget* w, const nanogui::Vector2i& p);
/*!
 * Find widgets that are of type T.
 * @tparam T The type of widget to find
 * @param parent The parent widget
 * @param widgets A vector of widgets of type T to append newly found widgets to.
 */
template<typename T> void find_widgets(nanogui::Widget* parent, std::vector<T>& widgets) {
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

CV_EXPORTS void resizeKeepAspectRatio(const cv::UMat& src, cv::UMat& output, const cv::Size& dstSize,
        const cv::Scalar& bgcolor = {0,0,0,255});

using namespace cv::viz::detail;

class NVG;

CV_EXPORTS class Viz2D {
    friend class NanoVGContext;
    const cv::Size initialSize_;
    cv::Size frameBufferSize_;
    cv::Rect viewport_;
    float scale_;
    cv::Vec2f mousePos_;
    bool offscreen_;
    bool stretch_;
    string title_;
    int major_;
    int minor_;
    int samples_;
    bool debug_;
    GLFWwindow* glfwWindow_ = nullptr;
    FrameBufferContext* clglContext_ = nullptr;
    CLVAContext* clvaContext_ = nullptr;
    NanoVGContext* nvgContext_ = nullptr;
    cv::VideoCapture* capture_ = nullptr;
    cv::VideoWriter* writer_ = nullptr;
    FormHelper* form_ = nullptr;
    bool closed_ = false;
    cv::Size videoFrameSize_ = cv::Size(0, 0);
    int vaCaptureDeviceIndex_ = 0;
    int vaWriterDeviceIndex_ = 0;
    bool mouseDrag_ = false;
    nanogui::Screen* screen_ = nullptr;
    Source source_;
    Sink sink_;
    std::function<bool(int key, int scancode, int action, int modifiers)> keyEventCb_;
public:
    /*!
     * Creates a Viz2D object which is the central object to perform visualizations with.
     * @param size The window and framebuffer size
     * @param title The window title.
     * @param debug Create a debug OpenGL context.
     */
    CV_EXPORTS static cv::Ptr<Viz2D> make(const cv::Size& size, const string& title, bool debug =
            false);

    /*!
     * Creates a Viz2D object which is the central object to perform visualizations with.
     * @param initialSize The initial size of the heavy-weight window.
     * @param frameBufferSize The initial size of the framebuffer backing the window (needs to be equal or greate then initial size).
     * @param offscreen Don't create a window and rather render offscreen.
     * @param title The window title.
     * @param major The OpenGL major version to request.
     * @param minor The OpenGL minor version to request.
     * @param samples MSAA samples.
     * @param debug Create a debug OpenGL context.
     */
    CV_EXPORTS static cv::Ptr<Viz2D> make(const cv::Size& initialSize,
            const cv::Size& frameBufferSize, bool offscreen, const string& title, int major = 3,
            int minor = 2, int samples = 0, bool debug = false);
    /*!
     * Default destructor
     */
    CV_EXPORTS virtual ~Viz2D();
    /*!
     * In case several Viz2D objects are in use all objects not in use have to
     * call #makeNoneCurrent() and only the one to be active call #makeCurrent().
     */
    CV_EXPORTS void makeCurrent();
    /*!
     * To make it possible for other Viz2D objects to become current all other
     * Viz2d instances have to become non-current.
     */
    CV_EXPORTS void makeNoneCurrent();

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
     * Called to capture to the framebuffer from a #cv::viz::Source object provided via #Viz2D::setSource().
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
     * Called to write the framebuffer to a #cv::viz::Sink object provided via #Viz2D::setSink()
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
    CV_EXPORTS cv::Rect getViewport();
    /*!
     * Set the window size.
     * @param sz The new window size.
     */
    CV_EXPORTS void setWindowSize(const cv::Size& sz);
    /*!
     * Get the window size
     * @return The current window size.
     */
    CV_EXPORTS cv::Size getWindowSize();
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
     * Get the pixel ratio of the display x-axis.
     * @return The pixel ratio of the display x-axis.
     */
    CV_EXPORTS float getXPixelRatio();
    /*!
     * Get the pixel ratio of the display y-axis.
     * @return The pixel ratio of the display y-axis.
     */
    CV_EXPORTS float getYPixelRatio();
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
private:
    /*!
     * Creates a Viz2D object which is the central object to perform visualizations with.
     * @param initialSize The initial size of the heavy-weight window.
     * @param frameBufferSize The initial size of the framebuffer backing the window (needs to be equal or greate then initial size).
     * @param offscreen Don't create a window and rather render offscreen.
     * @param title The window title.
     * @param major The OpenGL major version to request.
     * @param minor The OpenGL minor version to request.
     * @param samples MSAA samples.
     * @param debug Create a debug OpenGL context.
     */
    CV_EXPORTS Viz2D(const cv::Size& initialSize, const cv::Size& frameBufferSize, bool offscreen,
            const string& title, int major = 3, int minor = 2, int samples = 0, bool debug = false);
    void setDefaultKeyboardEventCallback();
    void setKeyboardEventCallback(
            std::function<bool(int key, int scancode, int action, int modifiers)> fn);
    bool initializeWindowing();
    void setMouseDrag(bool d);
    bool isMouseDrag();
    cv::Vec2f getMousePosition();
    bool keyboard_event(int key, int scancode, int action, int modifiers);
    void setMousePosition(int x, int y);
    nanogui::Screen& screen();
    FormHelper& form();
    FrameBufferContext& fb();
    CLVAContext& clva();
    NanoVGContext& nvg();
    GLFWwindow* getGLFWWindow();
    NVGcontext* getNVGcontext();
};
}
} /* namespace kb */

#endif /* SRC_COMMON_VIZ2D_HPP_ */
