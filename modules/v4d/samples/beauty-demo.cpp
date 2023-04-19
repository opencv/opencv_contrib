// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/v4d.hpp"
#include "opencv2/v4d/nvg.hpp"

#include <vector>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/face.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/tracking.hpp>

using std::cerr;
using std::endl;
using std::vector;
using std::string;

#ifdef __EMSCRIPTEN__
//enables KCF tracking instead of continuous detection.
#define USE_TRACKER 1;
#endif

/** Application parameters **/
constexpr unsigned int WIDTH = 1920;
constexpr unsigned int HEIGHT = 1080;
constexpr double SCALE = 0.125; //Scale at which face detection is performed
constexpr bool OFFSCREEN = false;
constexpr const char *OUTPUT_FILENAME = "beauty-demo.mkv";
const unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));

/** Effect parameters **/
constexpr int BLUR_DIV = 500;
int blur_skin_kernel_size = std::max(int(DIAG / BLUR_DIV % 2 == 0 ? DIAG / BLUR_DIV + 1 : DIAG / BLUR_DIV), 1);
float eyes_and_lips_saturation = 1.5f; //Saturation boost factor for eyes and lips
float skin_saturation = 1.2f; //Saturation boost factor for skin
float skin_contrast = 0.6f; //Contrast factor skin
#ifndef __EMSCRIPTEN__
bool side_by_side = true; //Show input and output side by side
bool stretch = true; //Stretch the video to the window size
#else
bool side_by_side = false;
bool stretch = false;
#endif

static cv::Ptr<cv::viz::V4D> v4d = cv::viz::V4D::make(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), OFFSCREEN, "Beauty Demo");
static cv::Ptr<cv::face::Facemark> facemark = cv::face::createFacemarkLBF(); //Face landmark detection
#ifdef USE_TRACKER
static cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create(); //Instead of continues face detection we can use a tracker
#endif

/*!
 * Data structure holding the points for all face landmarks
 */
struct FaceFeatures {
    cv::Rect faceRect_;
    vector<cv::Point2f> chin_;
    vector<cv::Point2f> top_nose_;
    vector<cv::Point2f> bottom_nose_;
    vector<cv::Point2f> left_eyebrow_;
    vector<cv::Point2f> right_eyebrow_;
    vector<cv::Point2f> left_eye_;
    vector<cv::Point2f> right_eye_;
    vector<cv::Point2f> outer_lips_;
    vector<cv::Point2f> inside_lips_;

    FaceFeatures(const cv::Rect &faceRect, const vector<cv::Point2f> &shape, double scale) {
        //calculate the face rectangle
        faceRect_ = cv::Rect(faceRect.x / scale, faceRect.y / scale, faceRect.width / scale, faceRect.height / scale);

        /** Copy all features **/
        size_t i = 0;
        // Around Chin. Ear to Ear
        for (i = 0; i <= 16; ++i)
            chin_.push_back(shape[i] / scale);
        // left eyebrow
        for (; i <= 21; ++i)
            left_eyebrow_.push_back(shape[i] / scale);
        // Right eyebrow
        for (; i <= 26; ++i)
            right_eyebrow_.push_back(shape[i] / scale);
        // Line on top of nose
        for (; i <= 30; ++i)
            top_nose_.push_back(shape[i] / scale);
        // Bottom part of the nose
        for (; i <= 35; ++i)
            bottom_nose_.push_back(shape[i] / scale);
        // Left eye
        for (; i <= 41; ++i)
            left_eye_.push_back(shape[i] / scale);
        // Right eye
        for (; i <= 47; ++i)
            right_eye_.push_back(shape[i] / scale);
        // Lips outer part
        for (; i <= 59; ++i)
            outer_lips_.push_back(shape[i] / scale);
        // Lips inside part
        for (; i <= 67; ++i)
            inside_lips_.push_back(shape[i] / scale);
    }

    //concatinates all feature points
    vector<cv::Point2f> points() const {
        vector<cv::Point2f> allPoints;
        allPoints.insert(allPoints.begin(), chin_.begin(), chin_.end());
        allPoints.insert(allPoints.begin(), top_nose_.begin(), top_nose_.end());
        allPoints.insert(allPoints.begin(), bottom_nose_.begin(), bottom_nose_.end());
        allPoints.insert(allPoints.begin(), left_eyebrow_.begin(), left_eyebrow_.end());
        allPoints.insert(allPoints.begin(), right_eyebrow_.begin(), right_eyebrow_.end());
        allPoints.insert(allPoints.begin(), left_eye_.begin(), left_eye_.end());
        allPoints.insert(allPoints.begin(), right_eye_.begin(), right_eye_.end());
        allPoints.insert(allPoints.begin(), outer_lips_.begin(), outer_lips_.end());
        allPoints.insert(allPoints.begin(), inside_lips_.begin(), inside_lips_.end());

        return allPoints;
    }

    //returns all face features points in fixed order
    vector<vector<cv::Point2f>> features() const {
        return {chin_,
            top_nose_,
            bottom_nose_,
            left_eyebrow_,
            right_eyebrow_,
            left_eye_,
            right_eye_,
            outer_lips_,
            inside_lips_};
    }

    size_t empty() const {
        return points().empty();
    }
};

//based on the detected FaceFeatures guesses a decent face oval and draws a mask.
void draw_face_oval_mask(const vector<FaceFeatures> &lm) {
    using namespace cv::viz::nvg;
    for (size_t i = 0; i < lm.size(); i++) {
        vector<vector<cv::Point2f>> features = lm[i].features();
        cv::RotatedRect rotRect = cv::fitEllipse(features[0]);

        beginPath();
        fillColor(cv::Scalar(255, 255, 255, 255));
        ellipse(rotRect.center.x, rotRect.center.y * 1, rotRect.size.width / 2, rotRect.size.height / 2.5);
        rotate(rotRect.angle);
        fill();
    }
}

//Draws a mask consisting of eyes and lips areas (deduced from FaceFeatures)
void draw_face_eyes_and_lips_mask(const vector<FaceFeatures> &lm) {
    using namespace cv::viz::nvg;
    for (size_t i = 0; i < lm.size(); i++) {
        vector<vector<cv::Point2f>> features = lm[i].features();
        for (size_t j = 5; j < 8; ++j) {
            beginPath();
            fillColor(cv::Scalar(255, 255, 255, 255));
            moveTo(features[j][0].x, features[j][0].y);
            for (size_t k = 1; k < features[j].size(); ++k) {
                lineTo(features[j][k].x, features[j][k].y);
            }
            closePath();
            fill();
        }

        beginPath();
        fillColor(cv::Scalar(0, 0, 0, 255));
        moveTo(features[8][0].x, features[8][0].y);
        for (size_t k = 1; k < features[8].size(); ++k) {
            lineTo(features[8][k].x, features[8][k].y);
        }
        closePath();
        fill();
    }
}

//adjusts the saturation of a UMat
void adjust_saturation(const cv::UMat &srcBGR, cv::UMat &dstBGR, float factor) {
    static vector<cv::UMat> channels;
    static cv::UMat hls;

    cvtColor(srcBGR, hls, cv::COLOR_BGR2HLS);
    split(hls, channels);
    cv::multiply(channels[2], factor, channels[2]);
    merge(channels, hls);
    cvtColor(hls, dstBGR, cv::COLOR_HLS2BGR);
}

//Setup the gui
void setup_gui(cv::Ptr<cv::viz::V4D> v4d) {
    v4d->nanogui([&](cv::viz::FormHelper& form){
        form.makeDialog(5, 30, "Effect");

        form.makeGroup("Display");
        form.makeFormVariable("Side by side", side_by_side, "Enable or disable side by side view");
        form.makeFormVariable("Stretch", stretch, "Enable or disable stetching to the window size");
    #ifndef __EMSCRIPTEN__
        form.makeButton("Fullscreen", [=]() {
            v4d->setFullscreen(!v4d->isFullscreen());
        });
    #endif
        form.makeButton("Offscreen", [=]() {
            v4d->setOffscreen(!v4d->isOffscreen());
        });

        form.makeGroup("Face Skin");
        auto* kernelSize = form.makeFormVariable("Blur", blur_skin_kernel_size, 0, 256, true, "", "use this kernel size to blur the face skin");
        kernelSize->set_callback([=](const int& k) {
            static int lastKernelSize = blur_skin_kernel_size;

            if(k == lastKernelSize)
                return;

            if(k <= lastKernelSize) {
                blur_skin_kernel_size = std::max(int(k % 2 == 0 ? k - 1 : k), 1);
            } else if(k > lastKernelSize)
                blur_skin_kernel_size = std::max(int(k % 2 == 0 ? k + 1 : k), 1);

            lastKernelSize = k;
            kernelSize->set_value(blur_skin_kernel_size);
        });
        form.makeFormVariable("Saturation", skin_saturation, 0.0f, 100.0f, true, "", "adjust the skin saturation by this amount");
        form.makeFormVariable("Contrast", skin_contrast, 0.0f, 1.0f, true, "", "contrast amount of the face skin");

        form.makeGroup("Eyes and Lips");
        form.makeFormVariable("Saturation", eyes_and_lips_saturation, 0.0f, 100.0f, true, "", "adjust the saturation of the eyes and the lips by this amount");
    });
}

bool iteration() {
    try {
#ifdef USE_TRACKER
        static bool trackerInitalized = false;
#endif
        //Face detector
        static cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create("face_detection_yunet_2022mar.onnx", "", cv::Size(v4d->getFrameBufferSize().width * SCALE, v4d->getFrameBufferSize().height * SCALE), 0.9, 0.3, 5000, cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_OPENCL);
        //Blender (used to put the different face parts back together)
        static cv::detail::MultiBandBlender blender(false, 5);
        blender.prepare(cv::Rect(0, 0, WIDTH, HEIGHT));
        //BGR
        static cv::UMat input, down, blurred, contrast, faceOval, eyesAndLips, skin;
        static cv::UMat frameOut(HEIGHT, WIDTH, CV_8UC3);
        static cv::UMat lhalf(HEIGHT * SCALE, WIDTH * SCALE, CV_8UC3);
        static cv::UMat rhalf(lhalf.size(), lhalf.type());
        //GREY
        static cv::UMat faceSkinMaskGrey, eyesAndLipsMaskGrey, backgroundMaskGrey;
        //BGR-Float
        static cv::UMat frameOutFloat;

        //detected faces
        static cv::Mat faces;
        //the rectangle of the tracked face
        static cv::Rect trackedFace;
        //rectangles of all faces found
        static vector<cv::Rect> faceRects;
        //list all of shapes (face features) found
        static vector<vector<cv::Point2f>> shapes;
        //FaceFeatures of all faces found
        static vector<FaceFeatures> featuresList;

        if (!v4d->capture())
            return false;

        v4d->fb([&](cv::UMat &frameBuffer) {
            cvtColor(frameBuffer, input, cv::COLOR_BGRA2BGR);
        });

        //Downscale the input for face detection
        cv::resize(input, down, cv::Size(0, 0), SCALE, SCALE);

        shapes.clear();
        faceRects.clear();
#ifdef USE_TRACKER
        //Use the KCF tracker to track the face "on past experience"
        if (trackerInitalized && tracker->update(down, trackedFace)) {
            faceRects.push_back(trackedFace);
        } else
#endif
        {
            //Detect faces in the down-scaled image
            detector->detect(down, faces);

            //Collect face bounding rectangles thought we will only use the first
            for (int i = 0; i < faces.rows; i++) {
                faceRects.push_back(cv::Rect(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))));
            }
        }

        //find landmarks if faces have been detected
        if (!faceRects.empty() && facemark->fit(down, faceRects, shapes)) {
#ifdef USE_TRACKER
            if(!trackerInitalized) {
                //on first time init the tracker
                tracker->init(down, faceRects[0]);
                trackerInitalized = true;
            }
#endif
            featuresList.clear();
            //a FaceFeatures instance for each face
            for (size_t i = 0; i < faceRects.size(); ++i) {
                featuresList.push_back(FaceFeatures(faceRects[i], shapes[i], float(down.size().width) / WIDTH));
            }

            v4d->nvg([&](const cv::Size &sz) {
                v4d->clear();
                //Draw the face oval of the first face
                draw_face_oval_mask(featuresList);
            });

            v4d->fb([&](cv::UMat &frameBuffer) {
                //Convert/Copy the mask
                cvtColor(frameBuffer, faceOval, cv::COLOR_BGRA2GRAY);
            });

            v4d->nvg([&](const cv::Size &sz) {
                v4d->clear();
                //Draw eyes eyes and lips areas of the first face
                draw_face_eyes_and_lips_mask(featuresList);
            });

            v4d->fb([&](cv::UMat &frameBuffer) {
                //Convert/Copy the mask
                cvtColor(frameBuffer, eyesAndLipsMaskGrey, cv::COLOR_BGRA2GRAY);
            });

            //Create the skin mask
            cv::subtract(faceOval, eyesAndLipsMaskGrey, faceSkinMaskGrey);
            //Create the background mask
            cv::bitwise_not(eyesAndLipsMaskGrey,backgroundMaskGrey);

            //boost saturation of eyes and lips
            adjust_saturation(input,eyesAndLips, eyes_and_lips_saturation);
            //reduce skin contrast
            multiply(input, cv::Scalar::all(skin_contrast), contrast);
            //fix skin brightness
            add(contrast, cv::Scalar::all((1.0 - skin_contrast) / 2.0) * 255.0, contrast);
            //blur the skin
            cv::boxFilter(contrast, blurred, -1, cv::Size(blur_skin_kernel_size, blur_skin_kernel_size), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
            //boost skin saturation
            adjust_saturation(blurred,skin, skin_saturation);

            //piece it all together
            blender.feed(skin, faceSkinMaskGrey, cv::Point(0, 0));
            blender.feed(input, backgroundMaskGrey, cv::Point(0, 0));
            blender.feed(eyesAndLips, eyesAndLipsMaskGrey, cv::Point(0, 0));
            blender.blend(frameOutFloat, cv::UMat());
            frameOutFloat.convertTo(frameOut, CV_8U, 1.0);

            if (side_by_side) {
                cv::resize(input, lhalf, cv::Size(0, 0), 0.5, 0.5);
                cv::resize(frameOut, rhalf, cv::Size(0, 0), 0.5, 0.5);

                frameOut = cv::Scalar::all(0);
                lhalf.copyTo(frameOut(cv::Rect(0, 0, lhalf.size().width, lhalf.size().height)));
                rhalf.copyTo(frameOut(cv::Rect(rhalf.size().width, 0, rhalf.size().width, rhalf.size().height)));
            }

            v4d->fb([&](cv::UMat &frameBuffer) {
                cvtColor(frameOut, frameBuffer, cv::COLOR_BGR2BGRA);
            });
        } else {
            if (side_by_side) {
                frameOut = cv::Scalar::all(0);
                cv::resize(input, lhalf, cv::Size(0, 0), 0.5, 0.5);
                lhalf.copyTo(frameOut(cv::Rect(0, 0, lhalf.size().width, lhalf.size().height)));
                lhalf.copyTo(frameOut(cv::Rect(lhalf.size().width, 0, lhalf.size().width, lhalf.size().height)));
            } else {
                input.copyTo(frameOut);
            }

            v4d->fb([&](cv::UMat &frameBuffer) {
                cvtColor(frameOut, frameBuffer, cv::COLOR_BGR2BGRA);
            });
        }
        updateFps(v4d, true);

#ifndef __EMSCRIPTEN__
        v4d->write();
#endif

        //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
        if (!v4d->display())
            return false;
    } catch (std::exception &ex) {
        cerr << ex.what() << endl;
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    using namespace cv::viz;
#ifndef __EMSCRIPTEN__
    if (argc != 2) {
        std::cerr << "Usage: beauty-demo <input-video-file>" << endl;
        exit(1);
    }
#endif
    facemark->loadModel("lbfmodel.yaml");

    printSystemInfo();

    v4d->setStretching(stretch);

    if (!v4d->isOffscreen()) {
        setup_gui(v4d);
        v4d->setVisible(true);
    }

#ifndef __EMSCRIPTEN__
    Source src = makeCaptureSource(argv[1]);
    v4d->setSource(src);
    Sink sink = makeWriterSink(OUTPUT_FILENAME, cv::VideoWriter::fourcc('V', 'P', '9', '0'), src.fps(), cv::Size(WIDTH, HEIGHT));
    v4d->setSink(sink);
#endif

    v4d->run(iteration);

    return 0;
}
