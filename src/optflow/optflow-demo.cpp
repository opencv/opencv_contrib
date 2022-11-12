#define CL_TARGET_OPENCL_VERSION 120

constexpr unsigned long WIDTH = 1920;
constexpr unsigned long HEIGHT = 1080;
constexpr unsigned int DOWN_SCALE = 2;
constexpr bool OFFSCREEN = true;
constexpr int VA_HW_DEVICE_INDEX = 0;
constexpr int MAX_POINTS = 2500;

#include "../common/tsafe_queue.hpp"
#include "../common/subsystems.hpp"
#include <thread>
#include <csignal>
#include <cstdint>
#include <iterator>
#include <iomanip>
#include <string>

#include <opencv2/optflow.hpp>
#include <opencv2/objdetect.hpp>

using std::cerr;
using std::endl;
using std::vector;
using std::string;

static bool done = false;
static void finish(int ignore) {
    std::cerr << endl;
    done = true;
}

SafeQueue<std::tuple<cv::UMat, std::vector<uchar>, std::vector<cv::Point2f>, std::vector<cv::Point2f>>> task_queue;
cv::RNG rng;

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


    std::thread producer([&]() {
        cv::Ptr<cv::BackgroundSubtractor> bgSubtractor = cv::createBackgroundSubtractorMOG2(300, 32.0, true);
        cv::UMat videoFrame, downScaled;
        cv::UMat downPrevGrey, downNextGrey, downMaskGrey, downEqualGrey;
        cv::UMat downNextGreyFloat, downMaskGreyFloat, downTargetGreyFloat;
        vector<cv::Point2f> downPoints, downPrevPoints, downNextPoints, downNewPoints;
        vector<vector<cv::Point> > contours;
        vector<cv::Vec4i> hierarchy;
        std::vector<uchar> status;
        std::vector<float> err;

        va::bind();
        while (!done) {
            cap >> videoFrame;
            if (videoFrame.empty())
                break;

            cv::resize(videoFrame, videoFrame, cv::Size(WIDTH, HEIGHT));
            cv::resize(videoFrame, downScaled, cv::Size(0, 0), 1.0 / DOWN_SCALE, 1.0 / DOWN_SCALE);
            cvtColor(downScaled, downNextGrey, cv::COLOR_RGB2GRAY);
            equalizeHist(downNextGrey, downEqualGrey);
            bgSubtractor->apply(downEqualGrey, downMaskGrey);
            int morph_size = 1;
            cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
            cv::morphologyEx(downMaskGrey, downMaskGrey, cv::MORPH_OPEN, element, cv::Point(-1, -1), 1);
            findContours(downMaskGrey, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

            downPoints.clear();
            for (const auto &c : contours) {
                for (const auto &pt : c) {
                    downPoints.push_back(pt);
                }
            }

            if (downPoints.size() > 4) {
                int copyn = std::min(downPoints.size(), (MAX_POINTS - downPrevPoints.size()));
                std::random_shuffle(downPoints.begin(), downPoints.end());

                if(downPrevPoints.size() < MAX_POINTS) {
                    std::copy(downPoints.begin(), downPoints.begin() + copyn, std::back_inserter(downPrevPoints));
                }

                if (downPrevGrey.empty()) {
                   downPrevGrey = downNextGrey.clone();
                }

                cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
                cv::calcOpticalFlowPyrLK(downPrevGrey, downNextGrey, downPrevPoints, downNextPoints, status, err, cv::Size(15, 15), 2, criteria);

                downNewPoints.clear();
                for(size_t i = 0; i < status.size(); ++i) {
                    if (status[i] == 1) {
                        downNewPoints.push_back(downNextPoints[i]);
                    }
                }

                task_queue.enqueue( { videoFrame.clone(), status, downPrevPoints, downNextPoints });
                downPrevPoints = downNewPoints;
            }
            downPrevGrey = downNextGrey.clone();
        }

        task_queue.enqueue({{},{},{},{}});
    });

    double avgLength = 1;

    uint64_t cnt = 1;
    int64 start = cv::getTickCount();
    double tickFreq = cv::getTickFrequency();
    double lastFps = fps;

    cv::UMat frameBuffer;
    cv::UMat background;
    cv::UMat foreground(HEIGHT, WIDTH, CV_8UC4, cv::Scalar::all(0));
    cv::UMat videoFrame;
    vector<cv::Point2f> downPrevPoints, downNextPoints, upPrevPoints, upNextPoints;
    std::vector<uchar> status;

    while (true) {
        auto tup = task_queue.dequeue();
        videoFrame = std::get<0>(tup);
        status = std::get<1>(tup);
        downPrevPoints = std::get<2>(tup);
        downNextPoints = std::get<3>(tup);

        if(videoFrame.empty())
            break;

        cv::cvtColor(videoFrame, background, cv::COLOR_RGB2BGRA);

        gl::bind();
        if (downPrevPoints.size() > 1 && downNextPoints.size() > 1) {
            upNextPoints.clear();
            upPrevPoints.clear();
            for (cv::Point2f pt : downPrevPoints) {
                upPrevPoints.push_back(pt *= float(DOWN_SCALE));
            }

            for (cv::Point2f pt : downNextPoints) {
                upNextPoints.push_back(pt *= float(DOWN_SCALE));
            }

            nvg::begin();
            nvg::clear();
            using kb::nvg::vg;
            nvgBeginPath(vg);
            nvgStrokeWidth(vg, std::fmax(3.0, WIDTH/960.0));
            nvgStrokeColor(vg, nvgHSLA(0.1, 1, 0.5, 48));
            for (size_t i = 0; i < downPrevPoints.size(); i++) {
                if (status[i] == 1 && upNextPoints[i].y >= 0 && upNextPoints[i].x >= 0 && upNextPoints[i].y < HEIGHT && upNextPoints[i].x < WIDTH) {
                    double len = hypot(fabs(upNextPoints[i].x - upPrevPoints[i].x), fabs(upNextPoints[i].y - upPrevPoints[i].y));
                    avgLength = ((avgLength * 0.95) + (len * 0.05));
                    if (len > 0 && len < avgLength) {
                        nvgMoveTo(vg, upNextPoints[i].x, upNextPoints[i].y);
                        nvgLineTo(vg, upPrevPoints[i].x, upPrevPoints[i].y);
                    }
                }
            }
            nvgStroke(vg);

            nvg::end();
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

    producer.join();

    return 0;
}
