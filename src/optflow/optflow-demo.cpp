#define CL_TARGET_OPENCL_VERSION 120

#include "../common/subsystems.hpp"
#include <cmath>
#include <vector>
#include <string>
#include <thread>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/optflow.hpp>

using std::cerr;
using std::endl;
using std::vector;
using std::string;
using namespace std::literals::chrono_literals;

/** Application parameters **/

constexpr unsigned int WIDTH = 1920;
constexpr unsigned int HEIGHT = 1080;
constexpr unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));
constexpr const char* OUTPUT_FILENAME = "optflow-demo.mkv";
constexpr bool OFFSCREEN = false;
constexpr int VA_HW_DEVICE_INDEX = 0;

/** Visualization parameters **/

// Generate the foreground at this scale.
float fg_scale = 0.5f;
// On every frame the foreground loses on brightness. specifies the loss in percent.
float fg_loss = 2.5;
// Peak thresholds for the scene change detection. Lowering them makes the detection more sensitive but
// the default should be fine.
float scene_change_thresh = 0.29f;
float scene_change_thresh_diff = 0.1f;
// The theoretical maximum number of points to track which is scaled by the density of detected points
// and therefor is usually much smaller.
int max_points = 250000;
// How many of the tracked points to lose intentionally, in percent.
float point_loss = 25;
// The theoretical maximum size of the drawing stroke which is scaled by the area of the convex hull
// of tracked points and therefor is usually much smaller.
int max_stroke = 17;
// Intensity of glow defined by kernel size. The default scales with the image diagonal.
int glow_kernel_size = std::max(int(DIAG / 138 % 2 == 0 ? DIAG / 138  + 1 : DIAG / 138), 1);
// Keep alpha separate for the GUI
float alpha = 0.1f;
// Red, green, blue and alpha. All from 0.0f to 1.0f
nanogui::Color effect_color(1.0f, 0.75f, 0.4f, 1.0f);
//display on-screen FPS
bool show_fps = true;
//Use OpenCL or not
bool use_opencl = true;

void prepare_motion_mask(const cv::UMat& srcGrey, cv::UMat& motionMaskGrey) {
    static cv::Ptr<cv::BackgroundSubtractor> bg_subtrator = cv::createBackgroundSubtractorMOG2(100, 16.0, false);

    bg_subtrator->apply(srcGrey, motionMaskGrey);

    int morph_size = 1;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
    cv::morphologyEx(motionMaskGrey, motionMaskGrey, cv::MORPH_OPEN, element, cv::Point(element.cols >> 1, element.rows >> 1), 2, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());
}

void detect_points(const cv::UMat& srcMotionMaskGrey, vector<cv::Point2f>& points) {
    static cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(1, false);
    static vector<cv::KeyPoint> tmpKeyPoints;

    tmpKeyPoints.clear();
    detector->detect(srcMotionMaskGrey, tmpKeyPoints);

    points.clear();
    for (const auto &kp : tmpKeyPoints) {
        points.push_back(kp.pt);
    }
}

bool detect_scene_change(const cv::UMat& srcMotionMaskGrey, const float thresh, const float theshDiff) {
    static float last_movement = 0;

    float movement = cv::countNonZero(srcMotionMaskGrey) / double(srcMotionMaskGrey.cols * srcMotionMaskGrey.rows);
    float relation = movement > 0 && last_movement > 0 ? std::max(movement, last_movement) / std::min(movement, last_movement) : 0;
    float relM = relation * log10(1.0f + (movement * 9.0));
    float relLM = relation * log10(1.0f + (last_movement * 9.0));

    bool result = !((movement > 0 && last_movement > 0 && relation > 0)
            && (relM < thresh && relLM < thresh && fabs(relM - relLM) < theshDiff));
    last_movement = (last_movement + movement) / 2.0f;
    return result;
}

void visualize_sparse_optical_flow(NVGcontext* vg, const cv::UMat &prevGrey, const cv::UMat &nextGrey, vector<cv::Point2f> &detectedPoints,
        const float scaleFactor, const int maxStrokeSize, const cv::Scalar color, const int maxPoints, const float pointLossPercent) {
    static vector<cv::Point2f> hull, prevPoints, nextPoints, newPoints;
    static vector<cv::Point2f> upPrevPoints, upNextPoints;
    static std::vector<uchar> status;
    static std::vector<float> err;

    if (detectedPoints.size() > 4) {
        cv::convexHull(detectedPoints, hull);
        float area = cv::contourArea(hull);
        float density = (detectedPoints.size() / area);
        float stroke = maxStrokeSize * pow(area / (nextGrey.cols * nextGrey.rows), 0.33f);
        size_t currentMaxPoints = density * maxPoints;

        std::random_shuffle(prevPoints.begin(), prevPoints.end());
        prevPoints.resize(ceil(prevPoints.size() * (1.0f - (pointLossPercent / 100.0f))));

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

            nvgBeginPath(vg);
            nvgStrokeWidth(vg, stroke);
            nvgStrokeColor(vg, nvgRGBA(color[0], color[1], color[2], color[3]));

            for (size_t i = 0; i < prevPoints.size(); i++) {
                if (status[i] == 1 && err[i] < (1.0 / density)
                        && upNextPoints[i].y >= 0 && upNextPoints[i].x >= 0
                        && upNextPoints[i].y < nextGrey.rows / scaleFactor
                            && upNextPoints[i].x < nextGrey.cols / scaleFactor) {
                    float len = hypot(fabs(upPrevPoints[i].x - upNextPoints[i].x), fabs(upPrevPoints[i].y - upNextPoints[i].y));
                    if (len > 0 && len < sqrt(area)) {
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

void glow_effect(const cv::UMat &src, cv::UMat &dst, const int ksize) {
    static cv::UMat resize;
    static cv::UMat blur;
    static cv::UMat dst16;

    cv::bitwise_not(src, dst);

    //Resize for some extra performance
    cv::resize(dst, resize, cv::Size(), 0.5, 0.5);
    //Cheap blur
    cv::boxFilter(resize, resize, -1, cv::Size(ksize, ksize), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
    //Back to original size
    cv::resize(resize, blur, src.size());

    //Multiply the src image with a blurred version of itself
    cv::multiply(dst, blur, dst16, 1, CV_16U);
    //Normalize and convert back to CV_8U
    cv::divide(dst16, cv::Scalar::all(255.0), dst, 1, CV_8U);

    cv::bitwise_not(dst, dst);
}

void composite_layers(const cv::UMat background, const cv::UMat foreground, const cv::UMat frameBuffer, cv::UMat dst, int glowKernelSize, float fgLossPercent) {
    static cv::UMat glow;
    static cv::UMat backgroundGrey;

    cv::subtract(foreground, cv::Scalar::all(255.0f * (fgLossPercent / 100.0f)), foreground);
    cv::add(foreground, frameBuffer, foreground);
    glow_effect(foreground, glow, glowKernelSize);
    cv::cvtColor(background, backgroundGrey, cv::COLOR_BGRA2GRAY);
    cv::cvtColor(backgroundGrey, background, cv::COLOR_GRAY2BGRA);
    cv::add(background, glow, dst);
}

void setup_gui(cv::Ptr<kb::Window> window) {
    window->makeWindow(6, 45, "Settings");

    auto useOpenCL = window->makeFormVariable("Use OpenCL", use_opencl, "Enable or disable OpenCL acceleration");

    window->makeGroup("Foreground");
    window->makeFormVariable("Scale", fg_scale, 0.1f, 4.0f, true, "", "Generate the foreground at this scale");
    window->makeFormVariable("Loss", fg_loss, 0.1f, 99.9f, true, "%", "On every frame the foreground loses on brightness");

    window->makeGroup("Scene Change Detection");
    window->makeFormVariable("Threshold", scene_change_thresh, 0.1f, 1.0f, true, "", "Peak threshold. Lowering it makes detection more sensitive");
    window->makeFormVariable("Threshold Diff", scene_change_thresh_diff, 0.1f, 1.0f, true, "", "Difference of peak thresholds. Lowering it makes detection more sensitive");

    window->makeGroup("Points");
    window->makeFormVariable("Max. Points", max_points, 10, 1000000, true, "", "The theoretical maximum number of points to track which is scaled by the density of detected points and therefor is usually much smaller");
    window->makeFormVariable("Point Loss", point_loss, 0.0f, 100.0f, true, "%", "How many of the tracked points to lose intentionally");

    window->makeGroup("Effect");
    window->makeFormVariable("Max. Stroke Size", max_stroke, 1, 100, true, "px", "The theoretical maximum size of the drawing stroke which is scaled by the area of the convex hull of tracked points and therefor is usually much smaller");
    auto glowKernel = window->makeFormVariable("Glow Kernel Size", glow_kernel_size, 1, 63, true, "", "Intensity of glow defined by kernel size");
    glowKernel->set_callback([](const int& k) {
        glow_kernel_size = std::max(int(k % 2 == 0 ? k + 1 : k), 1);
    });
    auto color = window->form()->add_variable("Color", effect_color);
    color->set_tooltip("The effect color");
    color->set_final_callback([](const nanogui::Color &c) {
        effect_color[0] = c[0];
        effect_color[1] = c[1];
        effect_color[2] = c[2];
    });
    window->makeFormVariable("Alpha", alpha, 0.0f, 1.0f, true, "", "The opacity of the effect");


    window->makeWindow(240, 45, "Display");
    window->makeGroup("Display");
    window->makeFormVariable("Show FPS", show_fps, "Enable or disable the On-screen FPS display");
    window->form()->add_button("Fullscreen", [=]() {
        window->setFullscreen(!window->isFullscreen());
    });

    if(!window->isOffscreen())
        window->setVisible(true);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: optflow <input-video-file>" << endl;
        exit(1);
    }
    cv::Ptr<kb::Window> window = new kb::Window(cv::Size(WIDTH, HEIGHT), OFFSCREEN, "Sparse Optical Flow Demo");

    std::thread worker([&]() {
        window->initialize();
        kb::print_system_info();
        setup_gui(window);

        auto capture = window->makeVACapture(argv[1], VA_HW_DEVICE_INDEX);

        if (!capture.isOpened()) {
            cerr << "ERROR! Unable to open video input" << endl;
            exit(-1);
        }

        float fps = capture.get(cv::CAP_PROP_FPS);

        window->makeVAWriter(OUTPUT_FILENAME, cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, window->getSize(), VA_HW_DEVICE_INDEX);

        //BGRA
        cv::UMat background, foreground(window->getSize(), CV_8UC4, cv::Scalar::all(0));
        //RGB
        cv::UMat rgb, down;
        //GREY
        cv::UMat backgroundGrey, downPrevGrey, downNextGrey, downMotionMaskGrey;
        vector<cv::Point2f> detectedPoints;

        while (true) {
            if(cv::ocl::useOpenCL() != use_opencl)
                window->setUseOpenCL(use_opencl);

            if(!window->captureVA())
                break;

            window->compute([&](cv::UMat& frameBuffer){
                cv::resize(frameBuffer, down, cv::Size(window->getSize().width * fg_scale, window->getSize().height * fg_scale));
                cv::cvtColor(frameBuffer, background, cv::COLOR_RGB2BGRA);
            });

            cv::cvtColor(down, downNextGrey, cv::COLOR_RGB2GRAY);
            //Subtract the background to create a motion mask
            prepare_motion_mask(downNextGrey, downMotionMaskGrey);
            //Detect trackable points in the motion mask
            detect_points(downMotionMaskGrey, detectedPoints);

            window->renderNVG([&](NVGcontext* vg, const cv::Size& sz) {
                window->clear();
                if (!downPrevGrey.empty()) {
                    //We don't want the algorithm to get out of hand when there is a scene change, so we suppress it when we detect one.
                    if (!detect_scene_change(downMotionMaskGrey, scene_change_thresh, scene_change_thresh_diff)) {
                        //Visualize the sparse optical flow using nanovg
                        cv::Scalar color = cv::Scalar(effect_color.r() * 255.0f, effect_color.g() * 255.0f, effect_color.b() * 255.0f, alpha * 255.0f);
                        visualize_sparse_optical_flow(vg, downPrevGrey, downNextGrey, detectedPoints, fg_scale, max_stroke, color, max_points, point_loss);
                    }
                }
            });

            downPrevGrey = downNextGrey.clone();

            window->compute([&](cv::UMat& frameBuffer){
                //Put it all together (OpenCL)
                composite_layers(background, foreground, frameBuffer, frameBuffer, glow_kernel_size, fg_loss);
            });

            window->writeVA();

            update_fps(window, show_fps);
            //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
            if(!window->display())
                break;
        }
        window->close();
    });

    while(!window->isClosed()) {
        window->pollEvents();
        std::this_thread::sleep_for(10ms);
    }

    worker.join();

    return 0;
}
