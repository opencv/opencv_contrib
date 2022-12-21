#ifndef SRC_COMMON_VIZ2D_HPP_
#define SRC_COMMON_VIZ2D_HPP_

#include <filesystem>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <nanogui/nanogui.h>
#include <GL/glew.h>

using std::cout;
using std::cerr;
using std::endl;
using std::string;

namespace kb {
namespace viz2d {
namespace detail {
class CLGLContext;
class CLVAContext;
class NanoVGContext;

void gl_check_error(const std::filesystem::path &file, unsigned int line, const char *expression);

#define GL_CHECK(expr)                            \
    expr;                                        \
    kb::viz2d::gl_check_error(__FILE__, __LINE__, #expr);

void error_callback(int error, const char *description);
}

cv::Scalar color_convert(const cv::Scalar& src, cv::ColorConversionCodes code);

using namespace kb::viz2d::detail;

class Viz2DWindow : public nanogui::Window {
private:
    static std::function<bool(Viz2DWindow*, Viz2DWindow*)> viz2DWin_Xcomparator;
    static std::set<Viz2DWindow*, decltype(viz2DWin_Xcomparator)> all_windows_xsorted_;
    nanogui::Screen* screen_;
    nanogui::Vector2i lastDragPos_;
    nanogui::Vector2i maximizedPos_;
    nanogui::Button* minBtn_;
    nanogui::Button* maxBtn_;
    nanogui::ref<nanogui::AdvancedGridLayout> oldLayout_;
    nanogui::ref<nanogui::AdvancedGridLayout> newLayout_;
    bool minimized_ = false;
public:
    Viz2DWindow(nanogui::Screen* screen, int x, int y, const string& title);
    virtual ~Viz2DWindow();
    bool isMinimized();
    bool mouse_drag_event(const nanogui::Vector2i &p, const nanogui::Vector2i &rel, int button, int mods) override;
};

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
    CLGLContext* clglContext_ = nullptr;
    CLVAContext* clvaContext_ = nullptr;
    NanoVGContext* nvgContext_ = nullptr;
    cv::VideoCapture* capture_ = nullptr;
    cv::VideoWriter* writer_ = nullptr;
    nanogui::FormHelper* form_ = nullptr;
    bool closed_ = false;
    cv::Size videoFrameSize_ = cv::Size(0,0);
    int vaCaptureDeviceIndex_ = 0;
    int vaWriterDeviceIndex_ = 0;
    bool mouseDrag_ = false;
    nanogui::Screen* screen_ = nullptr;
public:
    Viz2D(const cv::Size &initialSize, const cv::Size& frameBufferSize, bool offscreen, const string &title, int major = 4, int minor = 6, int samples = 0, bool debug = false);
    virtual ~Viz2D();
    bool initializeWindowing();
    void makeCurrent();

    cv::ogl::Texture2D& texture();
    void opengl(std::function<void(const cv::Size&)> fn);
    void opencl(std::function<void(cv::UMat&)> fn);
    void nanovg(std::function<void(const cv::Size&)> fn);
    void clear(const cv::Scalar& rgba = cv::Scalar(0,0,0,255));

    bool capture();
    void write();
    cv::VideoWriter& makeVAWriter(const string& outputFilename, const int fourcc, const float fps, const cv::Size& frameSize, const int vaDeviceIndex);
    cv::VideoCapture& makeVACapture(const string& intputFilename, const int vaDeviceIndex);
    cv::VideoWriter& makeWriter(const string& outputFilename, const int fourcc, const float fps, const cv::Size& frameSize);
    cv::VideoCapture& makeCapture(const string& intputFilename);
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
    bool isAccelerated();
    void setAccelerated(bool u);
    void close();
    bool display();

    Viz2DWindow* makeWindow(int x, int y, const string& title);
    nanogui::Label* makeGroup(const string& label);
    nanogui::detail::FormWidget<bool>* makeFormVariable(const string &name, bool &v, const string &tooltip = "", bool visible = true, bool enabled = true);
    template<typename T> nanogui::detail::FormWidget<T>* makeFormVariable(const string &name, T &v, const T &min, const T &max, bool spinnable, const string &unit, const string tooltip, bool visible = true, bool enabled = true) {
        auto var = form()->add_variable(name, v);
        var->set_enabled(enabled);
        var->set_visible(visible);
        var->set_spinnable(spinnable);
        var->set_min_value(min);
        var->set_max_value(max);
        if (!unit.empty())
            var->set_units(unit);
        if (!tooltip.empty())
            var->set_tooltip(tooltip);
        return var;
    }

    nanogui::ColorPicker* makeColorPicker(const string& label, nanogui::Color& color, const string& tooltip = "", std::function<void(const nanogui::Color)> fn = nullptr, bool visible = true, bool enabled = true);
    template<typename T> nanogui::ComboBox* makeComboBox(const string &label, T& e, const std::vector<string>& items) {
        auto* var = form()->add_variable("Mode", e, true);
        var->set_items(items);
        return var;
    }

    nanogui::Button* makeButton(const string& caption, std::function<void()> fn);
private:
    virtual bool keyboard_event(int key, int scancode, int action, int modifiers);
    void setMousePosition(int x, int y);
    nanogui::FormHelper* form();
    CLGLContext& clgl();
    CLVAContext& clva();
    NanoVGContext& nvg();
    nanogui::Screen& screen();
    GLFWwindow* getGLFWWindow();
    NVGcontext* getNVGcontext();
};
}
} /* namespace kb */

#endif /* SRC_COMMON_VIZ2D_HPP_ */
