#ifndef SRC_COMMON_GLWINDOW_HPP_
#define SRC_COMMON_GLWINDOW_HPP_

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include "clglcontext.hpp"
#include "clvacontext.hpp"
#include "nanovgcontext.hpp"

using std::cout;
using std::cerr;
using std::endl;
using std::string;

namespace kb {

void gl_check_error(const std::filesystem::path &file, unsigned int line, const char *expression);
#define GL_CHECK(expr)                            \
    expr;                                        \
    kb::gl_check_error(__FILE__, __LINE__, #expr);

class GLWindow {
    cv::Size size_;
    bool offscreen_;
    string title_;
    int major_;
    int minor_;
    int samples_;
    bool debug_;
    GLFWwindow* glfwWindow_ = nullptr;
    CLGLContext* clglContext_ = nullptr;
    CLVAContext* clvaContext_ = nullptr;
    NanoVGContext* nvgContext_ = nullptr;
    cv::VideoCapture* capture_ = nullptr;
    cv::VideoWriter* writer_ = nullptr;
    nanogui::Screen* screen_ = nullptr;
    nanogui::FormHelper* form_ = nullptr;
    bool closed_ = false;
public:

    GLWindow(const cv::Size &size, bool offscreen, const string &title, int major = 4, int minor = 6, int samples = 0, bool debug = false);
    ~GLWindow();
    void initialize();

    nanogui::FormHelper* form();
    CLGLContext& clgl();
    CLVAContext& clva();
    NanoVGContext& nvg();
    cv::TickMeter& getTickMeter();
    void render(std::function<void(const cv::Size&)> fn);
    void compute(std::function<void(cv::UMat&)> fn);
    void renderNVG(std::function<void(NVGcontext*, const cv::Size&)> fn);
    bool captureVA();
    void writeVA();
    cv::VideoWriter& makeVAWriter(const string& outputFilename, const int fourcc, const float fps, const cv::Size& frameSize, const int vaDeviceIndex);
    cv::VideoCapture& makeVACapture(const string& intputFilename, const int vaDeviceIndex);
    void clear(const cv::Scalar& rgba = cv::Scalar(0,0,0,255));
    cv::Size getFrameBufferSize();
    cv::Size getSize();
    float getPixelRatio();
    void setSize(const cv::Size& sz);
    bool isFullscreen();
    void setFullscreen(bool f);
    bool isResizable();
    void setResizable(bool r);
    bool isVisible();
    void setVisible(bool v);
    bool isOffscreen();
    nanogui::Window* makeWindow(int x, int y, const string& title);
    nanogui::Label* makeGroup(const string& label);
    nanogui::detail::FormWidget<bool>* makeFormVariable(const string &name, bool &v, const string &tooltip = "");
    template<typename T> nanogui::detail::FormWidget<T>* makeFormVariable(const string &name, T &v, const T &min, const T &max, bool spinnable, const string &unit, const string tooltip) {
        auto var = form()->add_variable(name, v);
        var->set_spinnable(spinnable);
        var->set_min_value(min);
        var->set_max_value(max);
        if (!unit.empty())
            var->set_units(unit);
        if (!tooltip.empty())
            var->set_tooltip(tooltip);
        return var;
    }
    void setUseOpenCL(bool u);
    bool display();
    bool isClosed();
    void close();
private:
    void makeGLFWContextCurrent();
    GLFWwindow* getGLFWWindow();
    NVGcontext* getNVGcontext();
};

} /* namespace kb */

#endif /* SRC_COMMON_GLWINDOW_HPP_ */
