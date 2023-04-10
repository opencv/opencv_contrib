// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/viz2d/viz2d.hpp"
#include "opencv2/viz2d/nvg.hpp"
#include "opencv2/viz2d/util.hpp"

#include <cmath>
#include <vector>
#include <set>
#include <string>
#include <thread>
#include <random>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/core/ocl.hpp>

using std::cerr;
using std::endl;
using std::vector;
using std::string;
using namespace std::literals::chrono_literals;

enum BackgroundModes {
    GREY,
    COLOR,
    VALUE,
    BLACK
};

enum PostProcModes {
    GLOW,
    BLOOM,
    NONE
};

/** Application parameters **/

constexpr unsigned int WIDTH = 1920;
constexpr unsigned int HEIGHT = 1080;
const unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));
constexpr const char* OUTPUT_FILENAME = "optflow-demo.mkv";
constexpr bool OFFSCREEN = false;

static cv::Ptr<cv::viz::Viz2D> v2d = cv::viz::Viz2D::make(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), OFFSCREEN, "Sparse Optical Flow Demo");
#ifndef __EMSCRIPTEN__
static cv::Ptr<cv::viz::Viz2D> v2dMenu = cv::viz::Viz2D::make(cv::Size(240, 360), cv::Size(240,360), false, "Display Settings");
#endif

/** Visualization parameters **/

// Generate the foreground at this scale.
#ifndef __EMSCRIPTEN__
float fg_scale = 0.5f;
#else
float fg_scale = 0.5f;
#endif
// On every frame the foreground loses on brightness. specifies the loss in percent.
#ifndef __EMSCRIPTEN__
float fg_loss = 2.5;
#else
float fg_loss = 10.0;
#endif
//Convert the background to greyscale
BackgroundModes background_mode = GREY;
// Peak thresholds for the scene change detection. Lowering them makes the detection more sensitive but
// the default should be fine.
float scene_change_thresh = 0.29f;
float scene_change_thresh_diff = 0.1f;
// The theoretical maximum number of points to track which is scaled by the density of detected points
// and therefor is usually much smaller.
#ifndef __EMSCRIPTEN__
int max_points = 250000;
#else
int max_points = 10000;
#endif
// How many of the tracked points to lose intentionally, in percent.
#ifndef __EMSCRIPTEN__
float point_loss = 25;
#else
float point_loss = 10;
#endif
// The theoretical maximum size of the drawing stroke which is scaled by the area of the convex hull
// of tracked points and therefor is usually much smaller.
#ifndef __EMSCRIPTEN__
int max_stroke = 14;
#else
int max_stroke = 2;
#endif
// Keep alpha separate for the GUI
#ifndef __EMSCRIPTEN__
float alpha = 0.1f;
#else
float alpha = 1.0f;
#endif

// Red, green, blue and alpha. All from 0.0f to 1.0f
nanogui::Color effect_color(1.0f, 0.75f, 0.4f, 1.0f);
//display on-screen FPS
bool show_fps = true;
//Stretch frame buffer to window size
bool stretch = false;
//Use OpenCL or not
bool use_acceleration = true;
//The post processing mode
#ifndef __EMSCRIPTEN__
PostProcModes post_proc_mode = GLOW;
#else
PostProcModes post_proc_mode = NONE;
#endif
// Intensity of glow or bloom defined by kernel size. The default scales with the image diagonal.
int GLOW_KERNEL_SIZE = std::max(int(DIAG / 100 % 2 == 0 ? DIAG / 100 + 1 : DIAG / 100), 1);
//The lightness selection threshold
int bloom_thresh = 210;
//The intensity of the bloom filter
float bloom_gain = 3;

void prepare_motion_mask(const cv::UMat& srcGrey, cv::UMat& motionMaskGrey) {
    static cv::Ptr<cv::BackgroundSubtractor> bg_subtrator = cv::createBackgroundSubtractorMOG2(100, 16.0, false);
    static int morph_size = 1;
    static cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));

    bg_subtrator->apply(srcGrey, motionMaskGrey);
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

    float movement = cv::countNonZero(srcMotionMaskGrey) / float(srcMotionMaskGrey.cols * srcMotionMaskGrey.rows);
    float relation = movement > 0 && last_movement > 0 ? std::max(movement, last_movement) / std::min(movement, last_movement) : 0;
    float relM = relation * log10(1.0f + (movement * 9.0));
    float relLM = relation * log10(1.0f + (last_movement * 9.0));

    bool result = !((movement > 0 && last_movement > 0 && relation > 0)
            && (relM < thresh && relLM < thresh && fabs(relM - relLM) < theshDiff));
    last_movement = (last_movement + movement) / 2.0f;
    return result;
}

void visualize_sparse_optical_flow(const cv::UMat &prevGrey, const cv::UMat &nextGrey, const vector<cv::Point2f> &detectedPoints, const float scaleFactor, const int maxStrokeSize, const cv::Scalar color, const int maxPoints, const float pointLossPercent) {
    static vector<cv::Point2f> hull, prevPoints, nextPoints, newPoints;
    static vector<cv::Point2f> upPrevPoints, upNextPoints;
    static std::vector<uchar> status;
    static std::vector<float> err;
    static std::random_device rd;
    static std::mt19937 g(rd());

    if (detectedPoints.size() > 4) {
        cv::convexHull(detectedPoints, hull);
        float area = cv::contourArea(hull);
        if (area > 0) {
            float density = (detectedPoints.size() / area);
            float strokeSize = maxStrokeSize * pow(area / (nextGrey.cols * nextGrey.rows), 0.33f);
            size_t currentMaxPoints = ceil(density * maxPoints);

            std::shuffle(prevPoints.begin(), prevPoints.end(), g);
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

                using namespace cv::viz::nvg;
                beginPath();
                strokeWidth(strokeSize);
                strokeColor(color);

                for (size_t i = 0; i < prevPoints.size(); i++) {
                    if (status[i] == 1 && err[i] < (1.0 / density) && upNextPoints[i].y >= 0 && upNextPoints[i].x >= 0 && upNextPoints[i].y < nextGrey.rows / scaleFactor && upNextPoints[i].x < nextGrey.cols / scaleFactor) {
                        float len = hypot(fabs(upPrevPoints[i].x - upNextPoints[i].x), fabs(upPrevPoints[i].y - upNextPoints[i].y));
                        if (len > 0 && len < sqrt(area)) {
                            newPoints.push_back(nextPoints[i]);
                            moveTo(upNextPoints[i].x, upNextPoints[i].y);
                            lineTo(upPrevPoints[i].x, upPrevPoints[i].y);
                        }
                    }
                }
                stroke();
            }
            prevPoints = newPoints;
        }
    }
}

void bloom(const cv::UMat& src, cv::UMat &dst, int ksize = 3, int threshValue = 235, float gain = 4) {
    static cv::UMat bgr;
    static cv::UMat hls;
    static cv::UMat ls16;
    static cv::UMat ls;
    static cv::UMat blur;
    static std::vector<cv::UMat> hlsChannels;

    cv::cvtColor(src, bgr, cv::COLOR_BGRA2RGB);
    cv::cvtColor(bgr, hls, cv::COLOR_BGR2HLS);
    cv::split(hls, hlsChannels);
    cv::bitwise_not(hlsChannels[2], hlsChannels[2]);

    cv::multiply(hlsChannels[1], hlsChannels[2], ls16, 1, CV_16U);
    cv::divide(ls16, cv::Scalar(255.0), ls, 1, CV_8U);
    cv::threshold(ls, blur, threshValue, 255, cv::THRESH_BINARY);

    cv::boxFilter(blur, blur, -1, cv::Size(ksize, ksize), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
    cv::cvtColor(blur, blur, cv::COLOR_GRAY2BGRA);

    addWeighted(src, 1.0, blur, gain, 0, dst);
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

void composite_layers(cv::UMat& background, const cv::UMat& foreground, const cv::UMat& frameBuffer, cv::UMat& dst, int kernelSize, float fgLossPercent, BackgroundModes bgMode, PostProcModes ppMode) {
    static cv::UMat tmp;
    static cv::UMat post;
    static cv::UMat backgroundGrey;
    static vector<cv::UMat> channels;

    cv::subtract(foreground, cv::Scalar::all(255.0f * (fgLossPercent / 100.0f)), foreground);
    cv::add(foreground, frameBuffer, foreground);

    switch (bgMode) {
    case GREY:
        cv::cvtColor(background, backgroundGrey, cv::COLOR_BGRA2GRAY);
        cv::cvtColor(backgroundGrey, background, cv::COLOR_GRAY2BGRA);
        break;
    case VALUE:
        cv::cvtColor(background, tmp, cv::COLOR_BGRA2BGR);
        cv::cvtColor(tmp, tmp, cv::COLOR_BGR2HSV);
        split(tmp, channels);
        cv::cvtColor(channels[2], background, cv::COLOR_GRAY2BGRA);
        break;
    case COLOR:
        break;
    case BLACK:
        background = cv::Scalar::all(0);
        break;
    default:
        break;
    }

    switch (ppMode) {
    case GLOW:
        glow_effect(foreground, post, kernelSize);
        break;
    case BLOOM:
        bloom(foreground, post, kernelSize, bloom_thresh, bloom_gain);
        break;
    case NONE:
        foreground.copyTo(post);
        break;
    default:
        break;
    }

    cv::add(background, post, dst);
}

void setup_gui(cv::Ptr<cv::viz::Viz2D> v2d, cv::Ptr<cv::viz::Viz2D> v2dMenu) {
    v2d->nanogui([&](cv::viz::FormHelper& form){
        form.makeDialog(5, 30, "Effects");

        form.makeGroup("Foreground");
        form.makeFormVariable("Scale", fg_scale, 0.1f, 4.0f, true, "", "Generate the foreground at this scale");
        form.makeFormVariable("Loss", fg_loss, 0.1f, 99.9f, true, "%", "On every frame the foreground loses on brightness");

        form.makeGroup("Background");
        form.makeComboBox("Mode",background_mode, {"Grey", "Color", "Value", "Black"});

        form.makeGroup("Points");
        form.makeFormVariable("Max. Points", max_points, 10, 1000000, true, "", "The theoretical maximum number of points to track which is scaled by the density of detected points and therefor is usually much smaller");
        form.makeFormVariable("Point Loss", point_loss, 0.0f, 100.0f, true, "%", "How many of the tracked points to lose intentionally");

        form.makeGroup("Optical flow");
        form.makeFormVariable("Max. Stroke Size", max_stroke, 1, 100, true, "px", "The theoretical maximum size of the drawing stroke which is scaled by the area of the convex hull of tracked points and therefor is usually much smaller");
        form.makeColorPicker("Color", effect_color, "The primary effect color",[&](const nanogui::Color &c) {
            effect_color[0] = c[0];
            effect_color[1] = c[1];
            effect_color[2] = c[2];
        });
        form.makeFormVariable("Alpha", alpha, 0.0f, 1.0f, true, "", "The opacity of the effect");

        form.makeDialog(220, 30, "Post Processing");
        auto* postPocMode = form.makeComboBox("Mode",post_proc_mode, {"Glow", "Bloom", "None"});
        auto* kernelSize = form.makeFormVariable("Kernel Size", GLOW_KERNEL_SIZE, 1, 63, true, "", "Intensity of glow defined by kernel size");
        kernelSize->set_callback([=](const int& k) {
            static int lastKernelSize = GLOW_KERNEL_SIZE;

            if(k == lastKernelSize)
                return;

            if(k <= lastKernelSize) {
                GLOW_KERNEL_SIZE = std::max(int(k % 2 == 0 ? k - 1 : k), 1);
            } else if(k > lastKernelSize)
                GLOW_KERNEL_SIZE = std::max(int(k % 2 == 0 ? k + 1 : k), 1);

            lastKernelSize = k;
            kernelSize->set_value(GLOW_KERNEL_SIZE);
        });
        auto* thresh = form.makeFormVariable("Threshold", bloom_thresh, 1, 255, true, "", "The lightness selection threshold", true, false);
        auto* gain = form.makeFormVariable("Gain", bloom_gain, 0.1f, 20.0f, true, "", "Intensity of the effect defined by gain", true, false);
        postPocMode->set_callback([=](const int& m) {
            switch(m) {
            case GLOW:
                kernelSize->set_enabled(true);
                thresh->set_enabled(false);
                gain->set_enabled(false);
            break;
            case BLOOM:
                kernelSize->set_enabled(true);
                thresh->set_enabled(true);
                gain->set_enabled(true);
            break;
            case NONE:
                kernelSize->set_enabled(false);
                thresh->set_enabled(false);
                gain->set_enabled(false);
            break;

            }
            postPocMode->set_selected_index(m);
        });

        form.makeDialog(220, 175, "Settings");

        form.makeGroup("Scene Change Detection");
        form.makeFormVariable("Threshold", scene_change_thresh, 0.1f, 1.0f, true, "", "Peak threshold. Lowering it makes detection more sensitive");
        form.makeFormVariable("Threshold Diff", scene_change_thresh_diff, 0.1f, 1.0f, true, "", "Difference of peak thresholds. Lowering it makes detection more sensitive");
    });

    v2dMenu->nanogui([&](cv::viz::FormHelper& form){
        form.makeDialog(8, 16, "Display");

        form.makeGroup("Display");
        form.makeFormVariable("Show FPS", show_fps, "Enable or disable the On-screen FPS display");
        form.makeFormVariable("Stetch", stretch, "Stretch the frame buffer to the window size")->set_callback([=](const bool &s) {
            v2d->setStretching(s);
        });

#ifndef __EMSCRIPTEN__
        form.makeButton("Fullscreen", [=]() {
            v2d->setFullscreen(!v2d->isFullscreen());
        });

        form.makeButton("Offscreen", [=]() {
            v2d->setOffscreen(!v2d->isOffscreen());
        });
#endif
    });
}

void iteration() {
    //BGRA
    static cv::UMat background, down;
    static cv::UMat foreground(v2d->getFrameBufferSize(), CV_8UC4, cv::Scalar::all(0));
    //RGB
    static cv::UMat menuFrame;
    //GREY
    static cv::UMat downPrevGrey, downNextGrey, downMotionMaskGrey;
    static vector<cv::Point2f> detectedPoints;

    if(!v2d->capture())
        exit(0);

    v2d->fb([=](cv::UMat& frameBuffer) {
        cv::resize(frameBuffer, down, cv::Size(v2d->getFrameBufferSize().width * fg_scale, v2d->getFrameBufferSize().height * fg_scale));
        frameBuffer.copyTo(background);
    });

    cv::cvtColor(down, downNextGrey, cv::COLOR_RGBA2GRAY);
    //Subtract the background to create a motion mask
    prepare_motion_mask(downNextGrey, downMotionMaskGrey);
    //Detect trackable points in the motion mask
    detect_points(downMotionMaskGrey, detectedPoints);

    v2d->nvg([=](const cv::Size& sz) {
        v2d->clear();
        if (!downPrevGrey.empty()) {
            //We don't want the algorithm to get out of hand when there is a scene change, so we suppress it when we detect one.
            if (!detect_scene_change(downMotionMaskGrey, scene_change_thresh, scene_change_thresh_diff)) {
                //Visualize the sparse optical flow using nanovg
                cv::Scalar color = cv::Scalar(effect_color.b() * 255.0f, effect_color.g() * 255.0f, effect_color.r() * 255.0f, alpha * 255.0f);
                visualize_sparse_optical_flow(downPrevGrey, downNextGrey, detectedPoints, fg_scale, max_stroke, color, max_points, point_loss);
            }
        }
    });

    downPrevGrey = downNextGrey.clone();

    v2d->fb([=](cv::UMat& frameBuffer){
        //Put it all together (OpenCL)
        composite_layers(background, foreground, frameBuffer, frameBuffer, GLOW_KERNEL_SIZE, fg_loss, background_mode, post_proc_mode);
#ifndef __EMSCRIPTEN__
        cvtColor(frameBuffer, menuFrame, cv::COLOR_BGRA2RGB);
#endif
    });

    updateFps(v2d, show_fps);

#ifndef __EMSCRIPTEN__
    v2d->write();

    v2dMenu->capture([=](cv::OutputArray& videoFrame) {
        cv::UMat m = videoFrame.getUMat();
        menuFrame.copyTo(m);
    });

    if(!v2dMenu->display())
        exit(0);
#endif

    //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
    if(!v2d->display())
        exit(0);
}
int main(int argc, char **argv) {
    using namespace cv::viz;
#ifndef __EMSCRIPTEN__
    if (argc != 2) {
        std::cerr << "Usage: optflow <input-video-file>" << endl;
        exit(1);
    }
#endif
    printSystemInfo();

    if(!v2d->isOffscreen()) {
#ifndef __EMSCRIPTEN__
        setup_gui(v2d, v2dMenu);
        v2dMenu->setResizable(false);
        v2dMenu->setVisible(true);
#else
        setup_gui(v2d, v2d);
#endif
        v2d->setVisible(true);
    }

#ifndef __EMSCRIPTEN__
    Source src = makeCaptureSource(argv[1]);
    v2d->setSource(src);

    Sink sink = makeWriterSink(OUTPUT_FILENAME, cv::VideoWriter::fourcc('V', 'P', '9', '0'), src.fps(), cv::Size(WIDTH, HEIGHT));
    v2d->setSink(sink);
#else
    Source src = makeCaptureSource(WIDTH, HEIGHT);
    v2d->setSource(src);
#endif

    v2d->run(iteration);

    return 0;
}
