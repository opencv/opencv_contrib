#include "util.hpp"

#include "viz2d.hpp"

namespace kb {

void error_callback(int error, const char *description) {
    fprintf(stderr, "GLFW Error: %s\n", description);
}

std::string get_gl_info() {
    return reinterpret_cast<const char*>(glGetString(GL_VERSION));
}

std::string get_cl_info() {
    std::stringstream ss;
    std::vector<cv::ocl::PlatformInfo> plt_info;
    cv::ocl::getPlatfomsInfo(plt_info);
    const cv::ocl::Device &defaultDevice = cv::ocl::Device::getDefault();
    cv::ocl::Device current;
    ss << endl;
    for (const auto &info : plt_info) {
        for (int i = 0; i < info.deviceNumber(); ++i) {
            ss << "\t";
            info.getDevice(current, i);
            if (defaultDevice.name() == current.name())
                ss << "* ";
            else
                ss << "  ";
            ss << info.version() << " = " << info.name() << endl;
            ss << "\t\t  GL sharing: " << (current.isExtensionSupported("cl_khr_gl_sharing") ? "true" : "false") << endl;
            ss << "\t\t  VAAPI media sharing: " << (current.isExtensionSupported("cl_intel_va_api_media_sharing") ? "true" : "false") << endl;
        }
    }

    return ss.str();
}

void print_system_info() {
    cerr << "OpenGL Version: " << get_gl_info() << endl;
    cerr << "OpenCL Platforms: " << get_cl_info() << endl;
}

void update_fps(cv::Ptr<kb::Viz2D> window, bool graphical) {
    static uint64_t cnt = 0;
    static cv::TickMeter tick;
    float fps;

    if (cnt > 0) {
        tick.stop();

        if (tick.getTimeMilli() > 1000) {
            cerr << "FPS : " << (fps = tick.getFPS()) << '\r';
            if (graphical) {
                window->nanovg([&](NVGcontext *vg, const cv::Size &size) {
                    string text = "FPS: " + std::to_string(fps);
                    nvgBeginPath(vg);
                    nvgRoundedRect(vg, 10, 10, 30 * text.size() + 10, 60, 10);
                    nvgFillColor(vg, nvgRGBA(255, 255, 255, 180));
                    nvgFill(vg);

                    nvgBeginPath(vg);
                    nvgFontSize(vg, 60.0f);
                    nvgFontFace(vg, "mono");
                    nvgFillColor(vg, nvgRGBA(90, 90, 90, 255));
                    nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
                    nvgText(vg, 22, 37, text.c_str(), nullptr);
                });
            }
            cnt = 0;
        }
    }

    tick.start();
    ++cnt;
}

} //namespace kb
