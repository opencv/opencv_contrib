#define CL_TARGET_OPENCL_VERSION 120

#include "../common/subsystems.hpp"

#include <vector>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/face.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

constexpr unsigned long WIDTH = 1920;
constexpr unsigned long HEIGHT = 1080;
constexpr double SCALE = 0.125;
constexpr bool OFFSCREEN = false;
constexpr int VA_HW_DEVICE_INDEX = 0;
constexpr unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));

constexpr int BLUR_KERNEL_SIZE = DIAG / 413 % 2 == 0 ? DIAG / 413 + 1 : DIAG / 413;

using std::cerr;
using std::endl;
using std::vector;
using std::string;

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
    kb::nvg::begin();
    kb::nvg::clear();
    using kb::nvg::vg;

    for (int i = 0; i < lm.size(); i++) {
        vector<vector<cv::Point2f>> features = lm[i].features();
        cv::RotatedRect rotRect = cv::fitEllipse(features[0]);

        nvgBeginPath(vg);
        nvgFillColor(vg, nvgRGBA(255, 255, 255, 255));
        nvgEllipse(vg, rotRect.center.x, rotRect.center.y, rotRect.size.width / 2, rotRect.size.height / 2);
        nvgRotate(vg, rotRect.angle);
        nvgFill(vg);
    }
    kb::nvg::end();
}

void draw_face_fg_mask(const vector<FaceFeatures> &lm) {
    kb::nvg::begin();
    kb::nvg::clear();
    using kb::nvg::vg;

    for (int i = 0; i < lm.size(); i++) {
        vector<vector<cv::Point2f>> features = lm[i].features();
        for (size_t j = 5; j < 8; ++j) {
            nvgBeginPath(vg);
            nvgFillColor(vg, nvgRGBA(255, 255, 255, 255));
            nvgMoveTo(vg, features[j][0].x, features[j][0].y);
            for (size_t k = 1; k < features[j].size(); ++k) {
                nvgLineTo(vg, features[j][k].x, features[j][k].y);
            }
            nvgClosePath(vg);
            nvgFill(vg);
        }
    }
    kb::nvg::end();
}

void reduce_shadows(const cv::UMat& srcBGR, cv::UMat& dstBGR, double to_percent) {
    assert(srcBGR.type() == CV_8UC3);
    static cv::UMat hsv;
    static vector<cv::UMat> hsvChannels;
    static cv::UMat valueFloat;


    cvtColor(srcBGR, hsv, cv::COLOR_BGR2HSV);
    cv::split(hsv, hsvChannels);
    hsvChannels[2].convertTo(valueFloat, CV_32F, 1.0/255.0);

    double minIn, maxIn;
    cv::minMaxLoc(valueFloat, &minIn, &maxIn);
    cv::subtract(valueFloat, minIn, valueFloat);
    cv::divide(valueFloat, cv::Scalar::all(maxIn - minIn), valueFloat);
    double minOut = (minIn * (1.0f - (to_percent / 100.0)));
    cv::multiply(valueFloat, cv::Scalar::all(1.0 - minOut), valueFloat);
    cv::add(valueFloat, cv::Scalar::all(minOut), valueFloat);

    valueFloat.convertTo(hsvChannels[2], CV_8U, 255.0);
    cv::merge(hsvChannels, hsv);
    cvtColor(hsv, dstBGR, cv::COLOR_HSV2BGR);
}

int main(int argc, char **argv) {
    using namespace kb;

    if (argc != 2) {
        std::cerr << "Usage: beauty-demo <input-video-file>" << endl;
        exit(1);
    }

    kb::init(WIDTH, HEIGHT);

    cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create("assets/face_detection_yunet_2022mar.onnx", "", cv::Size(320, 320), 0.9, 0.3, 5000, cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_OPENCL);
    cv::Ptr<cv::face::Facemark> facemark = cv::face::createFacemarkLBF();
    cv::detail::MultiBandBlender blender(true);

    facemark->loadModel("assets/lbfmodel.yaml");

    cv::VideoCapture capture(argv[1], cv::CAP_FFMPEG, { cv::CAP_PROP_HW_DEVICE, VA_HW_DEVICE_INDEX, cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI, cv::CAP_PROP_HW_ACCELERATION_USE_OPENCL, 1 });
    //Copy OpenCL Context for VAAPI. Must be called right after VideoWriter/VideoCapture initialization.
    va::init();

    if (!capture.isOpened()) {
        cerr << "ERROR! Unable to open video input" << endl;
        return -1;
    }

    double fps = capture.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer("beauty-demo.mkv", cv::CAP_FFMPEG, cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, cv::Size(WIDTH, HEIGHT), { cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_VAAPI, cv::VIDEOWRITER_PROP_HW_ACCELERATION_USE_OPENCL, 1 });

    if (!OFFSCREEN)
        x11::init();
    egl::init();
    gl::init();
    nvg::init();

    cerr << "EGL Version: " << egl::get_info() << endl;
    cerr << "OpenGL Version: " << gl::get_info() << endl;
    cerr << "OpenCL Platforms: " << endl << cl::get_info() << endl;

    //BGRA
    cv::UMat frameBuffer;
    //BGR
    cv::UMat videoFrameIn, resized, down, faceBgMask, diff, blurred, reduced, masked;
    cv::UMat videoFrameOut(HEIGHT, WIDTH, CV_8UC3);
    cv::UMat lhalf(HEIGHT * SCALE, WIDTH * SCALE, CV_8UC3);
    cv::UMat rhalf(lhalf.size(), lhalf.type());
    //GREY
    cv::UMat downGrey, faceBgMaskGrey, faceBgMaskInvGrey, faceFgMaskGrey, resMaskGrey;
    //BGR-Float
    cv::UMat videoFrameOutFloat;
    cv::Mat faces;
    vector<cv::Rect> faceRects;
    vector<vector<cv::Point2f>> shapes;
    vector<FaceFeatures> featuresList;

    va::bind();
    while (true) {
        capture >> videoFrameIn;
        if (videoFrameIn.empty())
            break;

        cv::resize(videoFrameIn, resized, cv::Size(WIDTH, HEIGHT));
        cv::resize(videoFrameIn, down, cv::Size(0, 0), SCALE, SCALE);
        cvtColor(down, downGrey, cv::COLOR_BGRA2GRAY);

        gl::bind();

        detector->setInputSize(down.size());
        detector->detect(down, faces);

        faceRects.clear();
        for (int i = 0; i < faces.rows; i++) {
            faceRects.push_back(cv::Rect(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))));
        }

        shapes.clear();

        if (!faceRects.empty() && facemark->fit(downGrey, faceRects, shapes)) {
            featuresList.clear();
            for (size_t i = 0; i < faceRects.size(); ++i) {
                featuresList.push_back(FaceFeatures(faceRects[i], shapes[i], float(down.size().width) / frameBuffer.size().width));
            }

            draw_face_bg_mask(featuresList);

            gl::acquire_from_gl(frameBuffer);
            cvtColor(frameBuffer, faceBgMask, cv::COLOR_BGRA2BGR);
            cvtColor(frameBuffer, faceBgMaskGrey, cv::COLOR_BGRA2GRAY);
            gl::release_to_gl(frameBuffer);

            draw_face_fg_mask(featuresList);

            gl::acquire_from_gl(frameBuffer);
            cvtColor(frameBuffer, faceFgMaskGrey, cv::COLOR_BGRA2GRAY);
            int morph_size = 1;
            cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
            cv::morphologyEx(faceFgMaskGrey, faceFgMaskGrey, cv::MORPH_DILATE, element, cv::Point(element.cols >> 1, element.rows >> 1), 1, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());
            cv::subtract(faceBgMaskGrey, faceFgMaskGrey, faceBgMaskGrey);
            cv::bitwise_not(faceBgMaskGrey, faceBgMaskInvGrey);

            reduce_shadows(resized, reduced, 10);
            blender.prepare(cv::Rect(0,0, WIDTH,HEIGHT));
            blender.feed(reduced, faceBgMaskGrey, cv::Point(0,0));
            blender.feed(resized, faceBgMaskInvGrey, cv::Point(0,0));
            blender.blend(videoFrameOutFloat, resMaskGrey);
            videoFrameOutFloat.convertTo(videoFrameOut, CV_8U, 1.0);

            cv::boxFilter(videoFrameOut, blurred, -1, cv::Size(BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
            cv::subtract(blurred, resized, diff);
            bitwise_and(diff, faceBgMask, masked);
            cv::add(videoFrameOut, masked, videoFrameOut);

            cv::resize(resized, lhalf, cv::Size(0, 0), 0.5, 0.5);
            cv::resize(reduced, rhalf, cv::Size(0, 0), 0.5, 0.5);
        } else {
            gl::acquire_from_gl(frameBuffer);
            cv::resize(resized, lhalf, cv::Size(0, 0), 0.5, 0.5);
        }

        videoFrameOut = cv::Scalar::all(0);
        lhalf.copyTo(videoFrameOut(cv::Rect(0, 0, lhalf.size().width, lhalf.size().height)));
        rhalf.copyTo(videoFrameOut(cv::Rect(rhalf.size().width, 0, rhalf.size().width, rhalf.size().height)));
        cvtColor(videoFrameOut, frameBuffer, cv::COLOR_BGR2RGBA);
        gl::release_to_gl(frameBuffer);

        if (!gl::display())
            break;

        va::bind();
        writer << videoFrameOut;

        print_fps();
    }

    return 0;
}
