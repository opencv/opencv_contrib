#include "util.hpp"

#include "viz2d.hpp"
#include "nvg.hpp"

namespace kb {
namespace viz2d {

void gl_check_error(const std::filesystem::path &file, unsigned int line, const char *expression) {
    int errorCode = glGetError();

    if (errorCode != 0) {
        std::cerr << "GL failed in " << file.filename() << " (" << line << ") : " << "\nExpression:\n   " << expression << "\nError code:\n   " << errorCode << "\n   " << std::endl;
        assert(false);
    }
}

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

void update_fps(cv::Ptr<kb::viz2d::Viz2D> window, bool graphical) {
    static uint64_t cnt = 0;
    static cv::TickMeter tick;
    float fps;

    if (cnt > 0) {
        tick.stop();

        if (tick.getTimeMilli() > 1000) {
            cerr << "FPS : " << (fps = tick.getFPS()) << '\r';
            if (graphical) {
                window->nanovg([&](const cv::Size &size) {
                    using namespace kb;
                    string text = "FPS: " + std::to_string(fps);
                    nvg::beginPath();
                    nvg::roundedRect(10, 10, 30 * text.size() + 10, 60, 10);
                    nvg::fillColor(cv::Scalar(255, 255, 255, 180));
                    nvg::fill();

                    nvg::beginPath();
                    nvg::fontSize(60.0f);
                    nvg::fontFace("mono");
                    nvg::fillColor(cv::Scalar(90, 90, 90, 255));
                    nvg::textAlign(NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
                    nvg::text(22, 37, text.c_str(), nullptr);
                });
            }
            cnt = 0;
        }
    }

    tick.start();
    ++cnt;
}

cv::Scalar convert(const cv::Scalar& src, cv::ColorConversionCodes code) {
    static cv::Mat tmpIn(1,1,CV_8UC3);
    static cv::Mat tmpOut(1,1,CV_8UC3);

    tmpIn.at<cv::Vec3b>(0,0) = cv::Vec3b(src[0], src[1], src[2]);

    cvtColor(tmpIn, tmpOut, code);
    const cv::Vec3b& vdst = tmpOut.at<cv::Vec3b>(0,0);
    cv::Scalar dst(vdst[0],vdst[1],vdst[2], src[3]);
    return dst;
}

}
}
