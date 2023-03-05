#define CL_TARGET_OPENCL_VERSION 120

#include "../common/viz2d.hpp"
#include "../common/nvg.hpp"
#include "../common/util.hpp"

#include <vector>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/face.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

/** Application parameters **/

constexpr unsigned int WIDTH = 1920;
constexpr unsigned int HEIGHT = 1080;
constexpr double SCALE = 0.125;
constexpr bool OFFSCREEN = false;
constexpr const char *OUTPUT_FILENAME = "beauty-demo.mkv";
constexpr int VA_HW_DEVICE_INDEX = 0;
const unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));

/** Effect parameters **/

constexpr int BLUR_DIV = 200;
const int BLUR_KERNEL_SIZE = std::max(int(DIAG / BLUR_DIV % 2 == 0 ? DIAG / BLUR_DIV + 1 : DIAG / BLUR_DIV), 1);
constexpr int REDUCE_SHADOW = 10; //percent
constexpr uchar BOOST_LIP_AND_EYE_SATURATION = 40; //0-255
constexpr int DILATE_ITERATIONS = 1;
#ifndef __EMSCRIPTEN__
constexpr bool SIDE_BY_SIDE = true;
constexpr bool STRETCH = true;
#else
constexpr bool SIDE_BY_SIDE = false;
constexpr bool STRETCH = false;
#endif

static cv::Ptr<kb::viz2d::Viz2D> v2d = new kb::viz2d::Viz2D(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), OFFSCREEN, "Beauty Demo");

using std::cerr;
using std::endl;
using std::vector;
using std::string;

#ifdef __EMSCRIPTEN__
#  include <emscripten.h>
#  include <emscripten/bind.h>
#  include <fstream>

using namespace emscripten;

std::string pushImage(std::string filename){
    try {
        std::ifstream fs(filename, std::fstream::in | std::fstream::binary);
        fs.seekg(0, std::ios::end);
        auto length = fs.tellg();
        fs.seekg(0, std::ios::beg);

        v2d->capture([&](cv::UMat &videoFrame) {
            if(videoFrame.empty())
                videoFrame.create(HEIGHT, WIDTH, CV_8UC3);
            if (length == (videoFrame.elemSize() + 1) * videoFrame.total()) {
                cv::Mat tmp;
                cv::Mat v = videoFrame.getMat(cv::ACCESS_RW);
                cvtColor(v, tmp, cv::COLOR_RGB2BGRA);
                fs.read((char*)(tmp.data), tmp.elemSize() * tmp.total());
                cvtColor(tmp, v, cv::COLOR_BGRA2RGB);
                v.release();
                tmp.release();
            } else {
                cerr << "mismatch" << endl;
            }
        });
        return "success";
    } catch(std::exception& ex) {
        return string(ex.what());
    }
}

EMSCRIPTEN_BINDINGS(my_module)
{
    function("push_image", &pushImage);
}
#endif

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
        faceRect_ = cv::Rect(faceRect.x / scale, faceRect.y / scale, faceRect.width / scale, faceRect.height / scale);
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

void draw_face_bg_mask(const vector<FaceFeatures> &lm) {
    using namespace kb::viz2d::nvg;
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

void draw_face_fg_mask(const vector<FaceFeatures> &lm) {
    using namespace kb::viz2d::nvg;
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
    }
}

void boost_saturation(const cv::UMat &srcBGR, cv::UMat &dstBGR, uchar by) {
    static vector<cv::UMat> channels;
    static cv::UMat hls;

    cvtColor(srcBGR, hls, cv::COLOR_BGR2HLS);
    split(hls, channels);
    cv::add(channels[2], by, channels[2]);
    merge(channels, hls);
    cvtColor(hls, dstBGR, cv::COLOR_HLS2BGR);
}

void lighten_shadows(const cv::UMat &srcBGR, cv::UMat &dstBGR, double to_percent) {
    assert(srcBGR.type() == CV_8UC3);
    static cv::UMat hsv;
    static vector<cv::UMat> hsvChannels;
    static cv::UMat valueFloat;

    cvtColor(srcBGR, hsv, cv::COLOR_BGR2HSV);
    cv::split(hsv, hsvChannels);
    hsvChannels[2].convertTo(valueFloat, CV_32F, 1.0 / 255.0);

    double minIn, maxIn;
    cv::minMaxLoc(valueFloat, &minIn, &maxIn);
    cv::subtract(valueFloat, minIn, valueFloat);
    cv::divide(valueFloat, cv::Scalar::all(maxIn - minIn), valueFloat);
    double minOut = (minIn + (1.0 * (to_percent / 100.0)));
    cv::multiply(valueFloat, cv::Scalar::all(1.0 - minOut), valueFloat);
    cv::add(valueFloat, cv::Scalar::all(minOut), valueFloat);

    valueFloat.convertTo(hsvChannels[2], CV_8U, 255.0);
    cv::merge(hsvChannels, hsv);
    cvtColor(hsv, dstBGR, cv::COLOR_HSV2BGR);
}

static cv::Ptr<cv::face::Facemark> facemark = cv::face::createFacemarkLBF();

void iteration() {
    try {
        static cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create("assets/face_detection_yunet_2022mar.onnx", "", cv::Size(v2d->getFrameBufferSize().width * SCALE, v2d->getFrameBufferSize().height * SCALE), 0.9, 0.3, 5000, cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_OPENCL);
        //TODO try FeatherBlender
        static cv::detail::MultiBandBlender blender(true);
        //BGR
        static cv::UMat bgr, down, faceBgMask, blurred, reduced, saturated;
        static cv::UMat frameOut(HEIGHT, WIDTH, CV_8UC3);
        static cv::UMat lhalf(HEIGHT * SCALE, WIDTH * SCALE, CV_8UC3);
        static cv::UMat rhalf(lhalf.size(), lhalf.type());
        //GREY
        static cv::UMat downGrey, faceBgMaskGrey, faceFgMaskGrey, faceFgMaskInvGrey;
        //BGR-Float
        static cv::UMat frameOutFloat;

        static cv::Mat faces;
        static vector<cv::Rect> faceRects;
        static vector<vector<cv::Point2f>> shapes;
        static vector<FaceFeatures> featuresList;

#ifndef __EMSCRIPTEN__
        if (!v2d->capture())
            exit(0);
#endif

        v2d->clgl([&](cv::UMat &frameBuffer) {
            cvtColor(frameBuffer, bgr, cv::COLOR_BGRA2BGR);
        });

        cv::resize(bgr, down, cv::Size(0, 0), SCALE, SCALE);
        cvtColor(down, downGrey, cv::COLOR_BGRA2GRAY);
        detector->detect(down, faces);

        faceRects.clear();
        for (int i = 0; i < faces.rows; i++) {
            faceRects.push_back(cv::Rect(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))));
        }

        shapes.clear();

        if (!faceRects.empty() && facemark->fit(downGrey, faceRects, shapes)) {
            featuresList.clear();
            for (size_t i = 0; i < faceRects.size(); ++i) {
                featuresList.push_back(FaceFeatures(faceRects[i], shapes[i], float(down.size().width) / WIDTH));
            }

            v2d->nvg([&](const cv::Size &sz) {
                v2d->clear();
                //Draw the face background mask (= face oval)
                draw_face_bg_mask(featuresList);
            });

            v2d->clgl([&](cv::UMat &frameBuffer) {
                //Convert/Copy the mask
                cvtColor(frameBuffer, faceBgMask, cv::COLOR_BGRA2BGR);
                cvtColor(frameBuffer, faceBgMaskGrey, cv::COLOR_BGRA2GRAY);
            });

            v2d->nvg([&](const cv::Size &sz) {
                v2d->clear();
                //Draw the face forground mask -> eyes and outer lips
                draw_face_fg_mask(featuresList);
            });

            v2d->clgl([&](cv::UMat &frameBuffer) {
                //Convert/Copy the mask
                cvtColor(frameBuffer, faceFgMaskGrey, cv::COLOR_BGRA2GRAY);
            });

            //Dilate the face forground mask to make eyes and mouth areas wider
            int morph_size = 1;
            cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
            cv::morphologyEx(faceFgMaskGrey, faceFgMaskGrey, cv::MORPH_DILATE, element, cv::Point(element.cols >> 1, element.rows >> 1), DILATE_ITERATIONS, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());

            cv::subtract(faceBgMaskGrey, faceFgMaskGrey, faceBgMaskGrey);

            cv::bitwise_not(faceFgMaskGrey,faceFgMaskInvGrey);

            boost_saturation(bgr,saturated,BOOST_LIP_AND_EYE_SATURATION);
            lighten_shadows(bgr, reduced, REDUCE_SHADOW);

            cv::boxFilter(reduced, blurred, -1, cv::Size(BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
            blender.prepare(cv::Rect(0, 0, WIDTH, HEIGHT));
            blender.feed(blurred, faceBgMaskGrey, cv::Point(0, 0));
            blender.feed(bgr, faceFgMaskInvGrey, cv::Point(0, 0));
            blender.feed(saturated, faceFgMaskGrey, cv::Point(0, 0));
            blender.blend(frameOutFloat, cv::UMat());
            frameOutFloat.convertTo(frameOut, CV_8U, 1.0);

            if (SIDE_BY_SIDE) {
                cv::resize(bgr, lhalf, cv::Size(0, 0), 0.5, 0.5);
                cv::resize(frameOut, rhalf, cv::Size(0, 0), 0.5, 0.5);

                frameOut = cv::Scalar::all(0);
                lhalf.copyTo(frameOut(cv::Rect(0, 0, lhalf.size().width, lhalf.size().height)));
                rhalf.copyTo(frameOut(cv::Rect(rhalf.size().width, 0, rhalf.size().width, rhalf.size().height)));
            }

            v2d->clgl([&](cv::UMat &frameBuffer) {
                cvtColor(frameOut, frameBuffer, cv::COLOR_BGR2BGRA);
            });
        } else {
            if (SIDE_BY_SIDE) {
                frameOut = cv::Scalar::all(0);
                cv::resize(bgr, lhalf, cv::Size(0, 0), 0.5, 0.5);
                lhalf.copyTo(frameOut(cv::Rect(0, 0, lhalf.size().width, lhalf.size().height)));
                lhalf.copyTo(frameOut(cv::Rect(lhalf.size().width, 0, lhalf.size().width, lhalf.size().height)));
            } else {
                bgr.copyTo(frameOut);
            }

            v2d->clgl([&](cv::UMat &frameBuffer) {
                cvtColor(frameOut, frameBuffer, cv::COLOR_BGR2BGRA);
            });
        }
        update_fps(v2d, true);

#ifndef __EMSCRIPTEN__
        v2d->write();
#endif

        //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
        if (!v2d->display())
            exit(0);
    } catch (std::exception &ex) {
        cerr << ex.what() << endl;
        exit(1);
    }
}

int main(int argc, char **argv) {
    using namespace kb::viz2d;
#ifndef __EMSCRIPTEN__
    if (argc != 2) {
        std::cerr << "Usage: beauty-demo <input-video-file>" << endl;
        exit(1);
    }
#endif
    facemark->loadModel("assets/lbfmodel.yaml");

    print_system_info();

    v2d->setStretching(STRETCH);

    if (!v2d->isOffscreen())
        v2d->setVisible(true);

#ifndef __EMSCRIPTEN__
    auto capture = v2d->makeVACapture(argv[1], VA_HW_DEVICE_INDEX);

    if (!capture.isOpened()) {
        cerr << "ERROR! Unable to open video input" << endl;
        exit(-1);
    }

    float fps = capture.get(cv::CAP_PROP_FPS);
    float width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    float height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    v2d->makeVAWriter(OUTPUT_FILENAME, cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, v2d->getFrameBufferSize(), VA_HW_DEVICE_INDEX);

    while (true)
        iteration();
#else
    emscripten_set_main_loop(iteration, -1, false);
#endif
    return 0;
}
