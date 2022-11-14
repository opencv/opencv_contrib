#define CL_TARGET_OPENCL_VERSION 120

constexpr unsigned long WIDTH = 1920;
constexpr unsigned long HEIGHT = 1080;
constexpr float SCALE_FACTOR = 0.5f;
constexpr unsigned long SCALED_WIDTH = WIDTH * SCALE_FACTOR;
constexpr unsigned long SCALED_HEIGHT = HEIGHT * SCALE_FACTOR;
constexpr bool OFFSCREEN = false;
constexpr int VA_HW_DEVICE_INDEX = 0;
constexpr float SCENE_CHANGE_THRESH = 0.29f;
constexpr float SCENE_CHANGE_THRESH_DIFF = 0.1f;

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
float last_movement = 0;
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

    vector<cv::Point2f> downFeaturePoints, downNewPoints, downPrevPoints, downNextPoints, downHull;
    vector<cv::Point2f> upPrevPoints, upNextPoints;
    vector<cv::KeyPoint> keypoints;

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

        int morph_size = 1;
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
        cv::morphologyEx(downMaskGrey, downMaskGrey, cv::MORPH_OPEN, element, cv::Point(element.cols >> 1, element.rows >> 1), 2, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());

        detector->detect(downMaskGrey, keypoints);

        float movement = cv::countNonZero(downMaskGrey) / double(SCALED_WIDTH * SCALED_HEIGHT);
        float relation = movement > 0 && last_movement > 0 ? std::max(movement, last_movement) / std::min(movement, last_movement) : 0;
        float relM = relation * log10(1.0f + (movement * 9.0));
        float relLM = relation * log10(1.0f + (last_movement * 9.0));

        gl::bind();
        nvg::begin();
        nvg::clear();

        if ((movement > 0 && last_movement > 0 && relation > 0)
                && (relM < SCENE_CHANGE_THRESH && relLM < SCENE_CHANGE_THRESH && fabs(relM - relLM) < SCENE_CHANGE_THRESH_DIFF)) {
            downFeaturePoints.clear();
            for (const auto &kp : keypoints) {
                downFeaturePoints.push_back(kp.pt);
            }

            if (downFeaturePoints.size() > 4) {
                cv::convexHull(downFeaturePoints, downHull);
                float downArea = cv::contourArea(downHull);
                float downDensity = (downFeaturePoints.size() / downArea);
                float stroke = 30.0 * sqrt(downArea / (SCALED_WIDTH * SCALED_HEIGHT * 4));
                current_max_points = downDensity * 500000.0;

                size_t copyn = std::min(downFeaturePoints.size(), (size_t(std::ceil(current_max_points)) - downPrevPoints.size()));
                if (downPrevPoints.size() < current_max_points) {
                    std::copy(downFeaturePoints.begin(), downFeaturePoints.begin() + copyn, std::back_inserter(downPrevPoints));
                }

                if (downPrevGrey.empty()) {
                    downPrevGrey = downNextGrey.clone();
                } else {
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
                        nvgStrokeColor(vg, nvgHSLA(0.1, 1, 0.55, 8));

                        for (size_t i = 0; i < downPrevPoints.size(); i++) {
                            if (status[i] == 1 && err[i] < (1.0 / downDensity) && upNextPoints[i].y >= 0 && upNextPoints[i].x >= 0 && upNextPoints[i].y < HEIGHT && upNextPoints[i].x < WIDTH && !(upPrevPoints[i].x == upNextPoints[i].x && upPrevPoints[i].y == upNextPoints[i].y)) {
                                float len = hypot(fabs(upPrevPoints[i].x - upNextPoints[i].x), fabs(upPrevPoints[i].y - upNextPoints[i].y));
                                if (len < sqrt(downArea)) {
                                    downNewPoints.push_back(downNextPoints[i]);
                                    nvgMoveTo(vg, upNextPoints[i].x, upNextPoints[i].y);
                                    nvgLineTo(vg, upPrevPoints[i].x, upPrevPoints[i].y);
                                }
                            }
                        }
                        nvgStroke(vg);
                    }
                }
                downPrevPoints = downNewPoints;
            }
        }
        nvg::end();

        last_movement = (last_movement + movement) / 2.0f;
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
