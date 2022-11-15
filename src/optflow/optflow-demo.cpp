#define CL_TARGET_OPENCL_VERSION 120

constexpr unsigned long WIDTH = 1920;
constexpr unsigned long HEIGHT = 1080;
constexpr float SCALE_FACTOR = 0.5f;
constexpr bool OFFSCREEN = false;
constexpr int VA_HW_DEVICE_INDEX = 0;
constexpr float SCENE_CHANGE_THRESH = 0.29f;
constexpr float SCENE_CHANGE_THRESH_DIFF = 0.1f;
constexpr float MAX_POINTS = 500000.0;

#include "../common/subsystems.hpp"
#include <vector>
#include <string>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/optflow.hpp>

using std::cerr;
using std::endl;
using std::vector;
using std::string;

void prepare_background_mask(const cv::UMat& srcGrey, cv::UMat& mask) {
    static cv::Ptr<cv::BackgroundSubtractor> bg_subtrator = cv::createBackgroundSubtractorMOG2(100, 16.0, false);

    bg_subtrator->apply(srcGrey, mask);

    int morph_size = 1;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element, cv::Point(element.cols >> 1, element.rows >> 1), 2, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());
}

void detect_points(const cv::UMat& srcMaskGrey, vector<cv::Point2f>& points) {
    static cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(1, false);
    static vector<cv::KeyPoint> tmpKeyPoints;

    tmpKeyPoints.clear();
    detector->detect(srcMaskGrey, tmpKeyPoints);

    points.clear();
    for (const auto &kp : tmpKeyPoints) {
        points.push_back(kp.pt);
    }
}

bool detect_scene_change(const cv::UMat& srcMaskGrey) {
    static float last_movement = 0;

    float movement = cv::countNonZero(srcMaskGrey) / double(srcMaskGrey.cols * srcMaskGrey.rows);
    float relation = movement > 0 && last_movement > 0 ? std::max(movement, last_movement) / std::min(movement, last_movement) : 0;
    float relM = relation * log10(1.0f + (movement * 9.0));
    float relLM = relation * log10(1.0f + (last_movement * 9.0));
    bool result = !((movement > 0 && last_movement > 0 && relation > 0)
            && (relM < SCENE_CHANGE_THRESH && relLM < SCENE_CHANGE_THRESH && fabs(relM - relLM) < SCENE_CHANGE_THRESH_DIFF));
    last_movement = (last_movement + movement) / 2.0f;
    return result;
}

void visualize_sparse_optical_flow(const cv::UMat& prevGrey, const cv::UMat &nextGrey, vector<cv::Point2f> &detectedPoints, const double scaleFactor) {
    static vector<cv::Point2f> hull, prevPoints, nextPoints, newPoints;
    static vector<cv::Point2f> upPrevPoints, upNextPoints;
    static std::vector<uchar> status;
    static std::vector<float> err;

    if (detectedPoints.size() > 4) {
        cv::convexHull(detectedPoints, hull);
        float area = cv::contourArea(hull);
        float density = (detectedPoints.size() / area);
        float stroke = 30.0 * sqrt(area / (nextGrey.cols * nextGrey.rows * 4));
        size_t currentMaxPoints = density * MAX_POINTS;

        size_t copyn = std::min(detectedPoints.size(), (size_t(std::ceil(currentMaxPoints)) - prevPoints.size()));
        if (prevPoints.size() < currentMaxPoints) {
            std::copy(detectedPoints.begin(), detectedPoints.begin() + copyn, std::back_inserter(prevPoints));
        }

        cv::calcOpticalFlowPyrLK(prevGrey, nextGrey, prevPoints, nextPoints, status, err);
        newPoints.clear();
        if (prevPoints.size() > 1 && nextPoints.size() > 1) {
            upNextPoints.clear();
            upPrevPoints.clear();
            for (cv::Point2f pt : prevPoints) {
                upPrevPoints.push_back(pt /= scaleFactor);
            }

            for (cv::Point2f pt : nextPoints) {
                upNextPoints.push_back(pt /= scaleFactor);
            }

            using kb::nvg::vg;
            nvgBeginPath(vg);
            nvgStrokeWidth(vg, stroke);
            nvgStrokeColor(vg, nvgHSLA(0.1, 1, 0.7, 3));

            for (size_t i = 0; i < prevPoints.size(); i++) {
                if (status[i] == 1 && err[i] < (1.0 / density) && upNextPoints[i].y >= 0 && upNextPoints[i].x >= 0 && upNextPoints[i].y < HEIGHT && upNextPoints[i].x < WIDTH && !(upPrevPoints[i].x == upNextPoints[i].x && upPrevPoints[i].y == upNextPoints[i].y)) {
                    float len = hypot(fabs(upPrevPoints[i].x - upNextPoints[i].x), fabs(upPrevPoints[i].y - upNextPoints[i].y));
                    if (len < sqrt(area)) {
                        newPoints.push_back(nextPoints[i]);
                        nvgMoveTo(vg, upNextPoints[i].x, upNextPoints[i].y);
                        nvgLineTo(vg, upPrevPoints[i].x, upPrevPoints[i].y);
                    }
                }
            }
            nvgStroke(vg);
        }
        prevPoints = newPoints;
    }
}

int main(int argc, char **argv) {
    using namespace kb;

    if (argc != 2) {
        std::cerr << "Usage: optflow <input-video-file>" << endl;
        exit(1);
    }

    va::init();
    cv::VideoCapture capture(argv[1], cv::CAP_FFMPEG, {
            cv::CAP_PROP_HW_DEVICE, VA_HW_DEVICE_INDEX,
            cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI,
            cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL, 1
    });

    if (!capture.isOpened()) {
        cerr << "ERROR! Unable to open video input" << endl;
        return -1;
    }

    double fps = capture.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer("optflow.mkv", cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, cv::Size(WIDTH, HEIGHT), {
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

    cv::Size frameBufferSize(WIDTH, HEIGHT);
    cv::UMat frameBuffer, videoFrame, resized, down, background, foreground(frameBufferSize, CV_8UC4, cv::Scalar::all(0));
    cv::UMat backgroundGrey, downPrevGrey, downNextGrey, downMaskGrey;
    vector<cv::Point2f> detectedPoints;

    va::bind();
    while (true) {
        capture >> videoFrame;
        if (videoFrame.empty())
            break;

        cv::resize(videoFrame, resized, frameBufferSize);
        cv::resize(videoFrame, down, cv::Size(WIDTH * SCALE_FACTOR, HEIGHT * SCALE_FACTOR));
        cv::cvtColor(resized, background, cv::COLOR_RGB2BGRA);
        cv::cvtColor(down, downNextGrey, cv::COLOR_RGB2GRAY);

        prepare_background_mask(downNextGrey, downMaskGrey);
        detect_points(downMaskGrey, detectedPoints);

        gl::bind();
        nvg::begin();
        nvg::clear();
        if (!downPrevGrey.empty()) {
            if (!detect_scene_change(downMaskGrey)) {
                visualize_sparse_optical_flow(downPrevGrey, downNextGrey, detectedPoints, SCALE_FACTOR);
            }
        }
        nvg::end();

        downPrevGrey = downNextGrey.clone();

        gl::acquire_from_gl(frameBuffer);
        cv::flip(frameBuffer, frameBuffer, 0);
        cv::subtract(foreground, cv::Scalar::all(9), foreground);
        cv::add(foreground, frameBuffer, foreground);
        cv::cvtColor(background, backgroundGrey, cv::COLOR_BGRA2GRAY);
        cv::cvtColor(backgroundGrey, background, cv::COLOR_GRAY2BGRA);
        cv::add(background, foreground, frameBuffer);
        cv::cvtColor(frameBuffer, videoFrame, cv::COLOR_BGRA2RGB);
        cv::flip(frameBuffer, frameBuffer, 0);
        gl::release_to_gl(frameBuffer);

        //if x11 is enabled it displays the framebuffer in the native window. returns false if the window was closed.
        if(!gl::display())
            break;

        va::bind();
        writer << videoFrame;

        print_fps();
    }

    return 0;
}
