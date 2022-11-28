#define CL_TARGET_OPENCL_VERSION 120

#include "../common/tsafe_queue.hpp"
#include "../common/subsystems.hpp"
#include <csignal>
#include <cstdint>
#include <iomanip>
#include <thread>
#include <string>

#include <opencv2/objdetect/objdetect.hpp>

constexpr unsigned long WIDTH = 1280;
constexpr unsigned long HEIGHT = 720;
constexpr unsigned long DOWNSIZE_WIDTH = 640;
constexpr unsigned long DOWNSIZE_HEIGHT = 360;
constexpr double WIDTH_FACTOR = double(WIDTH) / DOWNSIZE_WIDTH;
constexpr double HEIGHT_FACTOR = double(HEIGHT) / DOWNSIZE_HEIGHT;
constexpr bool OFFSCREEN = false;
constexpr const int VA_HW_DEVICE_INDEX = 0;
constexpr const char *OUTPUT_FILENAME = "pedestrian-demo.mkv";

using std::cerr;
using std::endl;
using std::vector;
using std::string;

static bool done = false;
static void finish(int ignore) {
    std::cerr << endl;
    done = true;
}

SafeQueue<std::pair<std::vector<cv::Rect>, cv::UMat>> task_queue;

int main(int argc, char **argv) {
    signal(SIGINT, finish);
    using namespace kb;

    if (argc != 2) {
        std::cerr << "Usage: pedestrian-demo <input-video-file>" << endl;
        exit(1);
    }

    kb::init(WIDTH, HEIGHT);

    cv::VideoCapture cap(argv[1], cv::CAP_FFMPEG, {
            cv::CAP_PROP_HW_DEVICE, VA_HW_DEVICE_INDEX,
            cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
            cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL, 1
    });

    va::copy();


    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera" << endl;
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    cerr << "Detected FPS: " << fps << endl;

//    cv::VideoWriter encoder(OUTPUT_FILENAME, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, cv::Size(WIDTH, HEIGHT), {
//            cv::VIDEOWRITER_PROP_HW_ACCELERATION,
//            cv::VIDEO_ACCELERATION_VAAPI, cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1
//    });

    if (!OFFSCREEN)
        x11::init();
    egl::init();
    gl::init();
    nvg::init();

    cerr << "EGL Version: " << egl::get_info() << endl;
    cerr << "OpenGL Version: " << gl::get_info() << endl;
    cerr << "OpenCL Platforms: " << endl << cl::get_info() << endl;

    cv::UMat frameBuffer;
    cv::UMat background;

    std::thread producer([&]() {
        cv::UMat videoFrame1, videoFrameUp1, videoFrameDown1, videoFrameDownGrey1;
        std::vector<cv::Rect> locations1;
        cv::HOGDescriptor hog;
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        va::bind();
        while (!done) {
            cap >> videoFrame1;
            if (videoFrame1.empty())
                break;

            cv::resize(videoFrame1, videoFrameUp1, cv::Size(WIDTH, HEIGHT));
            cv::resize(videoFrame1, videoFrameDown1, cv::Size(DOWNSIZE_WIDTH, DOWNSIZE_HEIGHT));
            cv::cvtColor(videoFrameDown1, videoFrameDownGrey1, cv::COLOR_RGB2GRAY);

            hog.detectMultiScale(videoFrameDownGrey1, locations1);
            task_queue.enqueue( { locations1, videoFrameUp1.clone() });
        }
        task_queue.enqueue( { { }, { } });
    });

    uint64_t cnt = 1;
    int64 start = cv::getTickCount();
    double tickFreq = cv::getTickFrequency();
    double lastFps = fps;

    cv::UMat videoFrame2;
    std::vector<cv::Rect> locations2;
    while (!done) {
        auto p = task_queue.dequeue();
        locations2 = p.first;
        videoFrame2 = p.second;

        if (videoFrame2.empty())
            break;

        cv::cvtColor(videoFrame2, background, cv::COLOR_RGB2BGRA);

        gl::bind();
        nvg::begin();
        nvg::clear();
        using kb::nvg::vg;

        nvgBeginPath(vg);
        nvgStrokeWidth(vg, std::fmax(4.0, WIDTH / 960.0));
        nvgStrokeColor(vg, nvgHSLA(0.0, 1, 0.5, 128));

        for (size_t i = 0; i < locations2.size(); i++) {
            nvgRect(vg, locations2[i].x * WIDTH_FACTOR, locations2[i].y * HEIGHT_FACTOR, locations2[i].width * WIDTH_FACTOR, locations2[i].height * HEIGHT_FACTOR);
        }

        nvgStroke(vg);
        nvg::end();

        gl::acquire_from_gl(frameBuffer);

        cv::flip(frameBuffer, frameBuffer, 0);
        cv::addWeighted(background, 1.0, frameBuffer, 1.0, 0.0, frameBuffer);
        cv::flip(frameBuffer, frameBuffer, 0);
        cv::cvtColor(frameBuffer, videoFrame2, cv::COLOR_BGRA2RGB);

        gl::release_to_gl(frameBuffer);

        va::bind();
        cv::flip(videoFrame2, videoFrame2, 0);
//        encoder.write(videoFrame2);

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
