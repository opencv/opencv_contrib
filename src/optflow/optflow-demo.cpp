#define CL_TARGET_OPENCL_VERSION 120

constexpr unsigned long WIDTH = 1920;
constexpr unsigned long HEIGHT = 1080;
constexpr float SCALE_FACTOR = 0.5;
constexpr unsigned long SCALED_WIDTH = WIDTH * SCALE_FACTOR;
constexpr unsigned long SCALED_HEIGHT = HEIGHT * SCALE_FACTOR;
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

float current_max_points = 5000;

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

    cv::Ptr<cv::BackgroundSubtractor> bgSubtractor = cv::createBackgroundSubtractorMOG2(100, 16.0, false);
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(1, false);

    cv::Size frameBufferSize(WIDTH, HEIGHT);
    cv::UMat frameBuffer, videoFrame, resized, down, background, foreground(frameBufferSize, CV_8UC4, cv::Scalar::all(0));
    cv::UMat backgroundGrey, downPrevGrey, downNextGrey, downMaskGrey;

    vector<cv::Point2f> downNewPoints, downPrevPoints, downNextPoints, downHull;
    vector<cv::Point2f> upPrevPoints, upNextPoints;
    vector<cv::KeyPoint> downPrevKeyPoints, keypoints;

    std::vector<uchar> status;
    std::vector<float> err;

    uint64_t cnt = 1;
    int64 start = cv::getTickCount();
    double tickFreq = cv::getTickFrequency();
    double lastFps = fps;

    va::bind();
    while (!done) {
        cap >> videoFrame;
        if (videoFrame.empty())
            break;
        if(videoFrame.size().width != frameBufferSize.width || videoFrame.size().height != frameBufferSize.height)
            cv::resize(videoFrame, resized, frameBufferSize);
        else
            resized = videoFrame;
        cv::resize(resized, down, cv::Size(0, 0), SCALE_FACTOR, SCALE_FACTOR);
        cv::cvtColor(resized, background, cv::COLOR_RGB2BGRA);
        cv::cvtColor(down, downNextGrey, cv::COLOR_RGB2GRAY);

        bgSubtractor->apply(downNextGrey, downMaskGrey);

        gl::bind();
        nvg::begin();
        nvg::clear();

        if (cv::countNonZero(downMaskGrey) < (SCALED_WIDTH * SCALED_HEIGHT * 0.25)) {
            int morph_size = 1;
            cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
            cv::morphologyEx(downMaskGrey, downMaskGrey, cv::MORPH_OPEN, element, cv::Point(element.cols >> 1, element.rows >> 1), 2, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());

            detector->detect(downMaskGrey, keypoints);
            downPrevKeyPoints.clear();

            for (const auto &kp : keypoints) {
                downPrevKeyPoints.push_back(kp.pt);
            }

            if (downPrevKeyPoints.size() > 4) {
                cv::convexHull(downPrevKeyPoints, downHull);
                float area = cv::contourArea(downHull);
                float density = (downPrevKeyPoints.size() / area);
                float stroke = 30.0 * sqrt(area / (SCALED_WIDTH * SCALED_HEIGHT * 4));
                current_max_points = density * 500000.0;
                size_t copyn = std::min(downPrevKeyPoints.size(), (size_t(std::ceil(current_max_points)) - downPrevPoints.size()));
                if (downPrevPoints.size() < current_max_points) {
                    std::copy(downPrevKeyPoints.begin(), downPrevKeyPoints.begin() + copyn, std::back_inserter(downPrevPoints));
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

                    using kb::nvg::vg;
                    nvgBeginPath(vg);
                    nvgStrokeWidth(vg, stroke);

                    for (size_t i = 0; i < downPrevPoints.size(); i++) {
                        nvgStrokeColor(vg, nvgHSLA(0.1, 1, 0.55, 16));
                        if (status[i] == 1 && err[i] < (1.0 / density)
                                && upNextPoints[i].y >= 0 && upNextPoints[i].x >= 0
                                && upNextPoints[i].y < HEIGHT && upNextPoints[i].x < WIDTH
                                && !(upPrevPoints[i].x == upNextPoints[i].x && upPrevPoints[i].y == upNextPoints[i].y)) {
                            float len = hypot(fabs(upPrevPoints[i].x - upNextPoints[i].x), fabs(upPrevPoints[i].y - upNextPoints[i].y));
                            if(len < sqrt(area)) {
                                downNewPoints.push_back(downNextPoints[i]);
                                nvgMoveTo(vg, upNextPoints[i].x, upNextPoints[i].y);
                                nvgLineTo(vg, upPrevPoints[i].x, upPrevPoints[i].y);
                            }
                        }
                    }
                    nvgStroke(vg);
                }
                downPrevPoints = downNewPoints;
            }
        } else {
            foreground = cv::Scalar::all(0);
        }
        nvg::end();

        downPrevGrey = downNextGrey.clone();

        gl::acquire_from_gl(frameBuffer);

        cv::flip(frameBuffer, frameBuffer, 0);
        cv::subtract(foreground, cv::Scalar::all(8), foreground);
        cv::add(foreground, frameBuffer, foreground);
        cv::cvtColor(background, backgroundGrey, cv::COLOR_BGRA2GRAY);
        cv::cvtColor(backgroundGrey, background, cv::COLOR_GRAY2BGRA);
        cv::add(background, foreground, frameBuffer);

        cv::cvtColor(frameBuffer, videoFrame, cv::COLOR_BGRA2RGB);
        cv::flip(frameBuffer, frameBuffer, 0);

        gl::release_to_gl(frameBuffer);

        if (x11::is_initialized()) {
            gl::blit_frame_buffer_to_screen();

            if (x11::window_closed()) {
                finish(0);
                break;
            }

            gl::swap_buffers();
        }

        va::bind();
        encoder.write(videoFrame);

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
