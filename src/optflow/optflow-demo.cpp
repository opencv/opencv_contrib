#define CL_TARGET_OPENCL_VERSION 120

constexpr unsigned long WIDTH = 1920;
constexpr unsigned long HEIGHT = 1080;
constexpr float SCALE_FACTOR = 0.5;
constexpr bool OFFSCREEN = false;
constexpr int VA_HW_DEVICE_INDEX = 0;

#include "../common/subsystems.hpp"
#include <csignal>
#include <list>
#include <vector>
#include <cstdint>
#include <string>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/optflow.hpp>

using std::cerr;
using std::endl;
using std::vector;
using std::list;
using std::string;

float current_max_points = 2000;

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
    cv::VideoCapture cap(argv[1], cv::CAP_FFMPEG, {
            cv::CAP_PROP_HW_DEVICE, VA_HW_DEVICE_INDEX,
            cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
            cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL, 1
    });

    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera" << endl;
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter encoder("optflow.mkv", cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, cv::Size(WIDTH, HEIGHT), {
            cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
            cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1
    });

    if (!OFFSCREEN)
        x11::init();
    egl::init();
    gl::init();
    nvg::init();

    cerr << "VA Version: " << va::get_info() << endl;
    cerr << "EGL Version: " << egl::get_info() << endl;
    cerr << "OpenGL Version: " << gl::get_info() << endl;
    cerr << "OpenCL Platforms: " << endl << cl::get_info() << endl;

    uint64_t cnt = 1;
    int64 start = cv::getTickCount();
    double tickFreq = cv::getTickFrequency();
    double lastFps = fps;

    cv::UMat frameBuffer, videoFrame, downScaled, background, foreground(HEIGHT, WIDTH, CV_8UC4, cv::Scalar::all(0));
    cv::UMat downPrevGrey, downNextGrey, downMaskGrey;

    vector<cv::Point2f> featurePoints, downNewPoints, downPrevPoints, downNextPoints, downHull, upPrevPoints, upNextPoints;
    cv::Ptr<cv::BackgroundSubtractor> bgSubtractor = cv::createBackgroundSubtractorMOG2(100, 16.0, false);
    cv::Ptr<cv::ORB> detector = cv::ORB::create(10000);
    std::vector<uchar> status;
    std::vector<float> err;
    while (!done) {
        va::bind();
        cap >> videoFrame;
        if (videoFrame.empty())
            break;

        cv::resize(videoFrame, videoFrame, cv::Size(WIDTH, HEIGHT));
        cv::cvtColor(videoFrame, background, cv::COLOR_RGB2BGRA);
        cv::resize(videoFrame, downScaled, cv::Size(0, 0), SCALE_FACTOR, SCALE_FACTOR);
        cvtColor(downScaled, downNextGrey, cv::COLOR_RGB2GRAY);

        bgSubtractor->apply(downNextGrey, downMaskGrey);

        if (cv::countNonZero(downMaskGrey) < ((WIDTH*SCALE_FACTOR) * (HEIGHT*SCALE_FACTOR) * 0.25)) {
            int morph_size = 1;
            cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
            cv::morphologyEx(downMaskGrey, downMaskGrey, cv::MORPH_OPEN, element, cv::Point(-1, -1), 2);

            vector<cv::KeyPoint> kps;
            detector->detect(downMaskGrey,kps);
            featurePoints.clear();

            for (const auto &kp : kps) {
                featurePoints.push_back(kp.pt);
            }

            gl::bind();
            nvg::begin();
            nvg::clear();

            if (featurePoints.size() > 12) {
                cv::convexHull(featurePoints, downHull);
                float area = cv::contourArea(downHull);
                float density = (featurePoints.size() / area);
                current_max_points = density * 25000.0;
                size_t copyn = std::min(featurePoints.size(), (size_t(std::ceil(current_max_points)) - downPrevPoints.size()));
                if (downPrevPoints.size() < current_max_points) {
                    std::copy(featurePoints.begin(), featurePoints.begin() + copyn, std::back_inserter(downPrevPoints));
                }

                if (downPrevGrey.empty()) {
                    downPrevGrey = downNextGrey.clone();
                }

                cv::calcOpticalFlowPyrLK(downPrevGrey, downNextGrey, downPrevPoints, downNextPoints, status, err);

                downNewPoints.clear();
                if (downPrevPoints.size() > 1 && downNextPoints.size() > 1) {
                    upNextPoints.clear();
                    upPrevPoints.clear();
                    for (cv::Point2f pt : downPrevPoints) {
                        upPrevPoints.push_back(pt /= SCALE_FACTOR);
                    }

                    for (cv::Point2f pt : downNextPoints) {
                        upNextPoints.push_back(pt /= SCALE_FACTOR);
                    }

                    float wholeArea = WIDTH * HEIGHT;
                    float stroke = 20.0 * sqrt(area / wholeArea);

                    using kb::nvg::vg;
                    nvgBeginPath(vg);
                    nvgStrokeWidth(vg, stroke);
                    nvgStrokeColor(vg, nvgHSLA(0.1, 1, 0.5, 48));

                    for (size_t i = 0; i < downPrevPoints.size(); i++) {
                        if (status[i] == 1 && err[i] < (1000000.0 / area)
                                && upNextPoints[i].y >= 0 && upNextPoints[i].x >= 0
                                && upNextPoints[i].y < HEIGHT && upNextPoints[i].x < WIDTH
                                && !(upPrevPoints[i].x == upNextPoints[i].x && upPrevPoints[i].y == upNextPoints[i].y)) {
                            downNewPoints.push_back(downNextPoints[i]);
                            nvgMoveTo(vg, upNextPoints[i].x, upNextPoints[i].y);
                            nvgLineTo(vg, upPrevPoints[i].x, upPrevPoints[i].y);
                        }
                    }
                    nvgStroke(vg);
                }
                downPrevPoints = downNewPoints;
            }
            nvg::end();
        } else {
            gl::bind();
            nvg::begin();
            nvg::clear();
            nvg::end();
            foreground = cv::Scalar::all(0);
        }

        downPrevGrey = downNextGrey.clone();

        gl::acquire_from_gl(frameBuffer);

        cv::flip(frameBuffer, frameBuffer, 0);
        cv::addWeighted(foreground, 0.9, frameBuffer, 1.1, 0.0, foreground);
        cv::addWeighted(background, 1.0, foreground, 1.0, 0.0, frameBuffer);
        cv::cvtColor(frameBuffer, videoFrame, cv::COLOR_BGRA2RGB);
        cv::flip(frameBuffer, frameBuffer, 0);

        gl::release_to_gl(frameBuffer);

        va::bind();
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
