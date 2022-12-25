#include "util.hpp"

#include "viz2d.hpp"
#include "nvg.hpp"

namespace kb {
namespace viz2d {

std::string get_gl_info() {
    return reinterpret_cast<const char*>(glGetString(GL_VERSION));
}

std::string get_cl_info() {
    std::stringstream ss;
#ifndef __EMSCRIPTEN__
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
#endif
    return ss.str();
}

void print_system_info() {
    cerr << "OpenGL Version: " << get_gl_info() << endl;
    cerr << "OpenCL Platforms: " << get_cl_info() << endl;
}

void update_fps(cv::Ptr<kb::viz2d::Viz2D> window, bool graphically) {
    static uint64_t cnt = 0;
    static cv::TickMeter tick;
    static float fps;

    if (cnt > 0) {
        tick.stop();

        if (tick.getTimeMilli() > 50) {
            cerr << "FPS : " << (fps = tick.getFPS());
#ifndef __EMSCRIPTEN__
            cerr << '\r';
#else
            cerr << endl;
#endif
            cnt = 0;
            tick.reset();
        }

        if (graphically) {
            window->nanovg([&](const cv::Size &size) {
                using namespace kb;
                string text = "FPS: " + std::to_string(fps);
                nvg::beginPath();
                nvg::roundedRect(5, 5, 15 * text.size() + 5, 30, 5);
                nvg::fillColor(cv::Scalar(255, 255, 255, 180));
                nvg::fill();

                nvg::fontSize(30.0f);
                nvg::fontFace("mono");
                nvg::fillColor(cv::Scalar(90, 90, 90, 255));
                nvg::textAlign(NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
                nvg::text(10, 20, text.c_str(), nullptr);
            });
        }
    }

    tick.start();
    ++cnt;
}

}
}
