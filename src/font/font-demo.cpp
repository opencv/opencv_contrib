#define CL_TARGET_OPENCL_VERSION 120

constexpr unsigned long WIDTH = 1920;
constexpr unsigned long HEIGHT = 1080;
constexpr bool OFFSCREEN = false;
constexpr const char *OUTPUT_FILENAME = "font-demo.mkv";
constexpr const int VA_HW_DEVICE_INDEX = 0;
constexpr double FPS = 60;

#include "../common/subsystems.hpp"

#include <string>
#include <algorithm>
#include <vector>
#include <sstream>

using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::stringstream;

int main(int argc, char **argv) {
    using namespace kb;

    //Initialize VP9 HW encoding using VAAPI
    cv::VideoWriter writer(OUTPUT_FILENAME, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), FPS, cv::Size(WIDTH, HEIGHT), {
            cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
            cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1
    });

    //Copy OpenCL Context for VAAPI. Must be called right after VideoWriter/VideoCapture initialization.
    va::init();


    if (!OFFSCREEN)
        x11::init();
    egl::init();
    gl::init();
    nvg::init();

    cerr << "EGL Version: " << egl::get_info() << endl;
    cerr << "OpenGL Version: " << gl::get_info() << endl;
    cerr << "OpenCL Platforms: " << endl << cl::get_info() << endl;

    cv::UMat frameBuffer;
    cv::UMat videoFrame;

    //Bind the OpenCL context for VAAPI
    string text = cv::getBuildInformation();
    size_t numLines = std::count(text.begin(), text.end(), '\n');
    vector<cv::Point2f> src = {{0,0},{WIDTH,0},{WIDTH,HEIGHT},{0,HEIGHT}};
    vector<cv::Point2f> dst = {{WIDTH/3,0},{WIDTH/1.5,0},{WIDTH,HEIGHT},{0,HEIGHT}};
    cv::Mat M = cv::getPerspectiveTransform(src, dst);

    float cnt = 1;
    while (true) {
        //Render using nanovg
        gl::bind();
        nvg::begin();
        nvg::clear();
        {
            using kb::nvg::vg;

            float lineh;
            float y = 0;

            nvgFontSize(vg, 40.0f);
            nvgFontFace(vg, "serif");
            nvgFillColor(vg, nvgHSLA(0.15, 1, 0.5, 255));
            nvgTextAlign(vg, NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
            nvgTextMetrics(vg, NULL, NULL, &lineh);
            size_t displayLines = ceil(cnt / lineh);
            size_t skipLines = numLines - displayLines;
            size_t maxLines = ceil(HEIGHT / lineh) + 1;
            nvgTranslate(vg, 0, cnt - ((displayLines * lineh)));

            std::istringstream iss(text);
            for (std::string line; std::getline(iss, line); ) {
                if(skipLines == 0) {
                    if((y / lineh) < maxLines) {
                        nvgText(vg, WIDTH/2.0, y, line.c_str(), line.c_str() + line.size());
                        y += lineh;
                    }
                } else {
                   --skipLines;
                }
            }
        }
        nvg::end();

        //Aquire frame buffer from OpenGL
        gl::acquire_from_gl(frameBuffer);
        cv::warpPerspective(frameBuffer, frameBuffer, M, videoFrame.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
        //Color-conversion from BGRA to RGB. OpenCV/OpenCL.
        cv::cvtColor(frameBuffer, videoFrame, cv::COLOR_BGRA2RGB);
        //Transfer buffer ownership back to OpenGL
        gl::release_to_gl(frameBuffer);

        //if x11 is enabled it displays the framebuffer in the native window. returns false if the window was closed.
        if(!gl::display())
            break;

        //Activate the OpenCL context for VAAPI
        va::bind();
        //Encode the frame using VAAPI on the GPU.
        writer << videoFrame;

        print_fps();
        ++cnt;
        if(cnt > 1000000)
            cnt = 1;
    }

    return 0;
}
