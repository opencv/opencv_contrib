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

#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <nanogui/nanogui.h>
#include <GL/glew.h>
#ifdef __EMSCRIPTEN__
#include <GLES3/gl3.h>
#include <emscripten.h>
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
namespace detail {
class FrameBufferContext;
class CLVAContext;
class NanoVGContext;

void gl_check_error(const std::filesystem::path& file, unsigned int line, const char* expression);

#define GL_CHECK(expr)                            \
    expr;                                        \
    cv::viz::gl_check_error(__FILE__, __LINE__, #expr);

void error_callback(int error, const char* description);
}

cv::Scalar color_convert(const cv::Scalar& src, cv::ColorConversionCodes code);

std::function<bool(int, int, int, int)> make_default_keyboard_event_callback();

using namespace cv::viz::detail;

class NVG;

class Viz2D {
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
    std::filesystem::path capturePath_;
    std::filesystem::path writerPath_;
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
    Viz2D(const cv::Size& initialSize, const cv::Size& frameBufferSize, bool offscreen,
            const string& title, int major = 4, int minor = 6, int samples = 0, bool debug = false);
    virtual ~Viz2D();
    bool initializeWindowing();
    void makeCurrent();
    void makeNoneCurrent();

    cv::ogl::Texture2D& texture();

    void gl(std::function<void(const cv::Size&)> fn);
    void fb(std::function<void(cv::UMat&)> fn);
    void nvg(std::function<void(const cv::Size&)> fn);
    void nanogui(std::function<void(FormHelper& form)>);

    void clear(const cv::Scalar& rgba = cv::Scalar(0, 0, 0, 255));

    bool capture();
    bool capture(std::function<void(cv::UMat&)> fn);
    void write();
    void write(std::function<void(const cv::UMat&)> fn);

    void setSource(const Source& src);
    bool isSourceReady();
    void setSink(const Sink& sink);
    bool isSinkReady();

    void showGui(bool s);

    void setMouseDrag(bool d);
    bool isMouseDrag();
    void pan(int x, int y);
    void zoom(float factor);
    cv::Vec2f getPosition();
    cv::Vec2f getMousePosition();
    float getScale();
    cv::Rect getViewport();
    void setWindowSize(const cv::Size& sz);
    cv::Size getWindowSize();
    cv::Size getInitialSize();
    void setVideoFrameSize(const cv::Size& sz);
    cv::Size getVideoFrameSize();
    cv::Size getFrameBufferSize();
    cv::Size getNativeFrameBufferSize();
    float getXPixelRatio();
    float getYPixelRatio();
    bool isFullscreen();
    void setFullscreen(bool f);
    bool isResizable();
    void setResizable(bool r);
    bool isVisible();
    void setVisible(bool v);
    bool isOffscreen();
    void setOffscreen(bool o);
    void setStretching(bool s);
    bool isStretching();
    bool isClosed();
    void close();
    bool display();

    void setDefaultKeyboardEventCallback();
    void setKeyboardEventCallback(
            std::function<bool(int key, int scancode, int action, int modifiers)> fn);
private:
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
