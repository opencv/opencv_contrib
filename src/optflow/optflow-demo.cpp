#define CL_TARGET_OPENCL_VERSION 120

constexpr unsigned long WIDTH = 1920;
constexpr unsigned long HEIGHT = 1080;
constexpr bool OFFSCREEN = true;
constexpr const int VA_HW_DEVICE_INDEX = 0;

#include "../common/subsystems.hpp"
#include <stdio.h>
#include <csignal>
#include <cstdint>
#include <iomanip>
#include <string>

#include <opencv2/optflow.hpp>

using std::cerr;
using std::endl;
using std::vector;
using std::string;

static bool done = false;
static void finish(int ignore) {
    std::cerr << endl;
    done = true;
}

int main(int argc, char **argv) {
    signal(SIGINT, finish);
    using namespace kb;

    if (argc != 2) {
        std::cerr << "Usage: optflow <input-video-file>" << endl;
        exit(1);
    }

    va::init();
    cv::VideoCapture cap(argv[1], cv::CAP_FFMPEG, { cv::CAP_PROP_HW_DEVICE, VA_HW_DEVICE_INDEX, cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI, cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL, 1 });
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera" << endl;
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter encoder("optflow.mkv", cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, cv::Size(WIDTH, HEIGHT), { cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI, cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1 });

    if (!OFFSCREEN)
        x11::init();
    egl::init();
    gl::init();
    nvg::init();

    cerr << "VA Version: " << va::get_info() << endl;
    cerr << "EGL Version: " << egl::get_info() << endl;
    cerr << "OpenGL Version: " << gl::get_info() << endl;
    cerr << "OpenCL Platforms: " << endl << cl::get_info() << endl;

    cv::UMat frameBuffer;
    cv::UMat background;
    cv::UMat foreground(HEIGHT, WIDTH, CV_8UC4, cv::Scalar::all(0));
    cv::UMat videoFrame, nextVideoFrameGray, prevVideoFrameGray, foregroundMaskGrey;

    cv::Ptr<cv::BackgroundSubtractor> bgSubtractor = cv::createBackgroundSubtractorMOG2(100, 32.0, false);

    vector<cv::Point2f> allPoints, prevPoints, nextPoints, newPoints;
    vector<vector<cv::Point> > contours;
    vector<cv::Vec4i> hierarchy;
    double avgLength = 1;

    uint64_t cnt = 1;
    int64 start = cv::getTickCount();
    double tickFreq = cv::getTickFrequency();
    double lastFps = fps;

    while (!done) {
        va::bind();
        cap >> videoFrame;
        if (videoFrame.empty())
            break;

        cv::resize(videoFrame, videoFrame, cv::Size(WIDTH, HEIGHT));
        cv::cvtColor(videoFrame, background, cv::COLOR_RGB2BGRA);
        cvtColor(videoFrame, nextVideoFrameGray, cv::COLOR_RGB2GRAY);

        bgSubtractor->apply(videoFrame, foregroundMaskGrey);

        int morph_size = 1;
        cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
        cv::morphologyEx(foregroundMaskGrey, foregroundMaskGrey, cv::MORPH_OPEN, element, cv::Point(-1, -1), 2);
        cv::morphologyEx(foregroundMaskGrey, foregroundMaskGrey, cv::MORPH_CLOSE, element, cv::Point(-1, -1), 2);
        findContours(foregroundMaskGrey, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        allPoints.clear();
        for (const auto &c : contours) {
            for (const auto &pt : c) {
                allPoints.push_back(pt);
            }
        }

        gl::bind();

        if (allPoints.size() > 4) {
            prevPoints = allPoints;
            if (prevVideoFrameGray.empty()) {
               prevVideoFrameGray = nextVideoFrameGray.clone();
            }

            std::vector<uchar> status;
            std::vector<float> err;
            cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
            cv::calcOpticalFlowPyrLK(prevVideoFrameGray, nextVideoFrameGray, prevPoints, nextPoints, status, err, cv::Size(15, 15), 2, criteria);

            nvg::begin();
            nvg::clear();
            newPoints.clear();

            using kb::nvg::vg;
            nvgBeginPath(vg);
            nvgStrokeWidth(vg, std::fmax(3.0, WIDTH/960.0));
            nvgStrokeColor(vg, nvgHSLA(0.1, 1, 0.5, 48));
            for (size_t i = 0; i < prevPoints.size(); i++) {

                if (status[i] == 1 && nextPoints[i].y >= 0 && nextPoints[i].x >= 0 && nextPoints[i].y < foregroundMaskGrey.rows && nextPoints[i].x < foregroundMaskGrey.cols) {
                    double len = hypot(fabs(nextPoints[i].x - prevPoints[i].x), fabs(nextPoints[i].y - prevPoints[i].y));
                    avgLength = ((avgLength * 0.95) + (len * 0.05));
                    if (len > 0 && len < avgLength) {
                        newPoints.push_back(nextPoints[i]);

                        nvgMoveTo(vg, nextPoints[i].x, nextPoints[i].y);
                        nvgLineTo(vg, prevPoints[i].x, prevPoints[i].y);
                    }
                }
            }
            nvgStroke(vg);

            nvg::end();

            prevVideoFrameGray = nextVideoFrameGray.clone();
            prevPoints = nextPoints;
        }

        gl::acquire_from_gl(frameBuffer);

        cv::flip(frameBuffer, frameBuffer, 0);
        cv::addWeighted(foreground, 0.9, frameBuffer, 1.1, 0.0, foreground);
        cv::addWeighted(background, 1.0, foreground, 1.0, 0.0, frameBuffer);
        cv::flip(frameBuffer, frameBuffer, 0);
        cv::cvtColor(frameBuffer, videoFrame, cv::COLOR_BGRA2RGB);

        gl::release_to_gl(frameBuffer);

        va::bind();
        cv::flip(videoFrame, videoFrame, 0);
        encoder.write(videoFrame);

        if (x11::is_initialized()) {
            gl::bind();
            gl::blit_frame_buffer_to_screen();

            if (x11::window_closed()) {
                finish(0);
                break;
            }

            gl::swap_buffers();
        }

        //Measure FPS
        if (cnt % uint64(ceil(lastFps)) == 0) {
            int64 tick = cv::getTickCount();
            lastFps = tickFreq / ((tick - start + 1) / cnt);
            cerr << "FPS : " << lastFps << '\r';
            start = tick;
            cnt = 1;
        }

        ++cnt;
    }

    return 0;
}
