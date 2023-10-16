// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>
#include <opencv2/v4d/v4d.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/face.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/tracking.hpp>

#include <vector>
#include <string>

using std::cerr;
using std::endl;
using std::vector;
using std::string;

/* Demo parameters */
#ifndef __EMSCRIPTEN__
constexpr long unsigned int WIDTH = 1280;
constexpr long unsigned int HEIGHT = 720;
#else
constexpr long unsigned int WIDTH = 960;
constexpr long unsigned int HEIGHT = 960;
#endif
constexpr unsigned int DOWNSIZE_WIDTH = 960;
constexpr unsigned int DOWNSIZE_HEIGHT = 540;
constexpr bool OFFSCREEN = false;
#ifndef __EMSCRIPTEN__
constexpr const char *OUTPUT_FILENAME = "beauty-demo.mkv";
#endif
const unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));

/* Visualization parameters */
constexpr int BLUR_DIV = 500;
static int blur_skin_kernel_size = std::max(int(DIAG / BLUR_DIV % 2 == 0 ? DIAG / BLUR_DIV + 1 : DIAG / BLUR_DIV), 1);
//Saturation boost factor for eyes and lips
static float eyes_and_lips_saturation = 2.0f;
//Saturation boost factor for skin
static float skin_saturation = 1.7f;
//Contrast factor skin
static float skin_contrast = 0.7f;
#ifndef __EMSCRIPTEN__
//Show input and output side by side
static bool side_by_side = true;
//Scale the video to the window size
static bool stretch = true;
#else
static bool side_by_side = false;
static bool stretch = false;
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
    FaceFeatures() {};
    FaceFeatures(const cv::Rect &faceRect, const vector<cv::Point2f> &shape, double local_scale) {
        //calculate the face rectangle
        faceRect_ = cv::Rect(faceRect.x / local_scale, faceRect.y / local_scale, faceRect.width / local_scale, faceRect.height / local_scale);

        /** Copy all features **/
        size_t i = 0;
        // Around Chin. Ear to Ear
        for (i = 0; i <= 16; ++i)
            chin_.push_back(shape[i] / local_scale);
        // left eyebrow
        for (; i <= 21; ++i)
            left_eyebrow_.push_back(shape[i] / local_scale);
        // Right eyebrow
        for (; i <= 26; ++i)
            right_eyebrow_.push_back(shape[i] / local_scale);
        // Line on top of nose
        for (; i <= 30; ++i)
            top_nose_.push_back(shape[i] / local_scale);
        // Bottom part of the nose
        for (; i <= 35; ++i)
            bottom_nose_.push_back(shape[i] / local_scale);
        // Left eye
        for (; i <= 41; ++i)
            left_eye_.push_back(shape[i] / local_scale);
        // Right eye
        for (; i <= 47; ++i)
            right_eye_.push_back(shape[i] / local_scale);
        // Lips outer part
        for (; i <= 59; ++i)
            outer_lips_.push_back(shape[i] / local_scale);
        // Lips inside part
        for (; i <= 67; ++i)
            inside_lips_.push_back(shape[i] / local_scale);
    }

    //Concatenates all feature points
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

    //Returns all feature points in fixed order
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

//based on the detected FaceFeatures it guesses a decent face oval and draws a mask for it.
static void draw_face_oval_mask(const FaceFeatures &ff) {
    using namespace cv::v4d::nvg;
    clear();

    vector<vector<cv::Point2f>> features = ff.features();
    cv::RotatedRect rotRect = cv::fitEllipse(features[0]);

    beginPath();
    fillColor(cv::Scalar(255, 255, 255, 255));
    ellipse(rotRect.center.x, rotRect.center.y * 1, rotRect.size.width / 2, rotRect.size.height / 2.5);
    rotate(rotRect.angle);
    fill();
}

//Draws a mask consisting of eyes and lips areas (deduced from FaceFeatures)
static void draw_face_eyes_and_lips_mask(const FaceFeatures &ff) {
    using namespace cv::v4d::nvg;
    clear();
    vector<vector<cv::Point2f>> features = ff.features();
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

//adjusts the saturation of a UMat
static void adjust_saturation(const cv::UMat &srcBGR, cv::UMat &dstBGR, float factor) {
    thread_local vector<cv::UMat> channels;
    thread_local cv::UMat hls;

    cvtColor(srcBGR, hls, cv::COLOR_BGR2HLS);
    split(hls, channels);
    cv::multiply(channels[2], factor, channels[2]);
    merge(channels, hls);
    cvtColor(hls, dstBGR, cv::COLOR_HLS2BGR);
}

using namespace cv::v4d;

class BeautyDemoPlan : public Plan {
	cv::Ptr<cv::face::Facemark> facemark_ = cv::face::createFacemarkLBF();
	//Blender (used to put the different face parts back together)
	cv::Ptr<cv::detail::MultiBandBlender> blender_ = new cv::detail::MultiBandBlender(false, 5);
	//Face detector
	#ifndef __EMSCRIPTEN__
	cv::Ptr<cv::FaceDetectorYN> detector_ = cv::FaceDetectorYN::create("modules/v4d/assets/models/face_detection_yunet_2023mar.onnx", "", cv::Size(DOWNSIZE_WIDTH, DOWNSIZE_HEIGHT), 0.9, 0.3, 5000, cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_OPENCL);
	#else
	cv::Ptr<cv::FaceDetectorYN> detector_ = cv::FaceDetectorYN::create("assets/models/face_detection_yunet_2023mar.onnx", "", cv::Size(DOWNSIZE_WIDTH, DOWNSIZE_HEIGHT), 0.9, 0.3, 5000, cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU);
	#endif
	//BGR
	cv::UMat input_, down_, contrast_, faceOval_, eyesAndLips_, skin_;
	cv::UMat lhalf_ = cv::UMat(DOWNSIZE_HEIGHT, DOWNSIZE_WIDTH, CV_8UC3);
	cv::UMat rhalf_ = cv::UMat(DOWNSIZE_HEIGHT, DOWNSIZE_WIDTH, CV_8UC3);
	cv::UMat frameOut_ = cv::UMat(HEIGHT, WIDTH, CV_8UC3);
	//GREY
	cv::UMat faceSkinMaskGrey_, eyesAndLipsMaskGrey_, backgroundMaskGrey_;
	//BGR-Float
	//list all of shapes (face features) found
	vector<vector<cv::Point2f>> shapes_;
	std::vector<cv::Rect> faceRects_;
	bool faceFound_ = false;
	FaceFeatures features_;
public:
	void setup(cv::Ptr<V4D> window) override {
		window->parallel([](cv::Ptr<cv::face::Facemark>& facemark){
#ifndef __EMSCRIPTEN__
			facemark->loadModel("modules/v4d/assets/models/lbfmodel.yaml");
#else
			facemark->loadModel("assets/models/lbfmodel.yaml");
#endif
			cerr << "Loading finished" << endl;
		}, facemark_);
	}
	void infer(cv::Ptr<V4D> window) override {
		auto always = [](){ return true; };
		auto isTrue = [](bool& ff){ return ff; };
		auto isFalse = [](bool& ff){ return !ff; };

		try {
			window->branch(always);
			{
				window->capture();

				//Save the video frame as BGR
				window->fb([](const cv::UMat &framebuffer, cv::UMat& in, cv::UMat& d) {
					cvtColor(framebuffer, in, cv::COLOR_BGRA2BGR);
					//Downscale the video frame for face detection
					cv::resize(in, d, cv::Size(DOWNSIZE_WIDTH, DOWNSIZE_HEIGHT));
				}, input_, down_);

				window->parallel([](cv::Ptr<cv::FaceDetectorYN>& de, cv::Ptr<cv::face::Facemark>& fm, vector<vector<cv::Point2f>>& sh, const cv::UMat& d, std::vector<cv::Rect>& fr, bool& ff, FaceFeatures& ft) {
					sh.clear();
					cv::Mat faces;
					//Detect faces in the down-scaled image
					de->detect(d, faces);
					//Only add the first face
					cv::Rect faceRect;
					if(!faces.empty())
						faceRect = cv::Rect(int(faces.at<float>(0, 0)), int(faces.at<float>(0, 1)), int(faces.at<float>(0, 2)), int(faces.at<float>(0, 3)));
					fr = {faceRect};
					//find landmarks if faces have been detected
					ff = !faceRect.empty() && fm->fit(d, fr, sh);
					if(ff)
						ft = FaceFeatures(fr[0], sh[0], float(d.size().width) / WIDTH);
				}, detector_, facemark_, shapes_, down_, faceRects_, faceFound_, features_);
			}
			window->endbranch(always);

			window->branch(isTrue, faceFound_);
			{
				window->nvg([](const FaceFeatures& f) {
					//Draw the face oval of the first face
					draw_face_oval_mask(f);
				}, features_);

				window->fb([](const cv::UMat& frameBuffer, cv::UMat& fo) {
					//Convert/Copy the mask
					cvtColor(frameBuffer, fo, cv::COLOR_BGRA2GRAY);
				}, faceOval_);

				window->nvg([](const FaceFeatures& f) {
					//Draw eyes eyes and lips areas of the first face
					draw_face_eyes_and_lips_mask(f);
				}, features_);

				window->fb([](const cv::UMat &frameBuffer, cv::UMat& ealmg) {
					//Convert/Copy the mask
					cvtColor(frameBuffer, ealmg, cv::COLOR_BGRA2GRAY);
				}, eyesAndLipsMaskGrey_);

				window->parallel([](const cv::UMat& fo, const cv::UMat& ealmg, cv::UMat& fsmg, cv::UMat& bmg) {
					//Create the skin mask
					cv::subtract(fo, ealmg, fsmg);
					//Create the background mask
					cv::bitwise_not(ealmg, bmg);
				}, faceOval_, eyesAndLipsMaskGrey_, faceSkinMaskGrey_, backgroundMaskGrey_);

				window->parallel([](const cv::UMat& in, cv::UMat& eal, float& eals,  cv::UMat& c, cv::UMat& s) {
					//boost saturation of eyes and lips
					adjust_saturation(in,  eal, eals);
					//reduce skin contrast
					multiply(in, cv::Scalar::all(skin_contrast), c);
					//fix skin brightness
					add(c, cv::Scalar::all((1.0 - skin_contrast) / 2.0) * 255.0, c);
					//blur the skin_
					cv::boxFilter(c, c, -1, cv::Size(blur_skin_kernel_size, blur_skin_kernel_size), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
					//boost skin saturation
					adjust_saturation(c, s, skin_saturation);
				}, input_, eyesAndLips_, eyes_and_lips_saturation, contrast_, skin_);


				window->parallel([](cv::Ptr<cv::detail::MultiBandBlender>& bl,
						const cv::UMat& s, const cv::UMat& fsmg,
						const cv::UMat& in, const cv::UMat& bmg,
						const cv::UMat& eal, const cv::UMat& ealmg,
						cv::UMat& fout) {
					cv:: UMat foFloat;
					//FIXME do it once?

					CV_Assert(!s.empty());
					CV_Assert(!in.empty());
					CV_Assert(!eal.empty());
					//piece it all together
					//FIXME prepare only once?
					bl->prepare(cv::Rect(0, 0, WIDTH, HEIGHT));
					bl->feed(s, fsmg, cv::Point(0, 0));
					bl->feed(in, bmg, cv::Point(0, 0));
					bl->feed(eal, ealmg, cv::Point(0, 0));
					bl->blend(foFloat, cv::UMat());
					CV_Assert(!foFloat.empty());
					foFloat.convertTo(fout, CV_8U, 1.0);
				}, blender_, skin_, faceSkinMaskGrey_, input_, backgroundMaskGrey_, eyesAndLips_, eyesAndLipsMaskGrey_, frameOut_);

				window->parallel([](cv::UMat& fout, const cv::UMat& in, cv::UMat& lh, cv::UMat& rh) {
					if (side_by_side) {
						//create side-by-side view with a result
						cv::resize(in, lh, cv::Size(0, 0), 0.5, 0.5);
						cv::resize(fout, rh, cv::Size(0, 0), 0.5, 0.5);

						fout = cv::Scalar::all(0);
						lh.copyTo(fout(cv::Rect(0, 0, lh.size().width, lh.size().height)));
						rh.copyTo(fout(cv::Rect(rh.size().width, 0, rh.size().width, rh.size().height)));
					}
				}, frameOut_, input_, lhalf_, rhalf_);
			}
			window->endbranch(isTrue, faceFound_);

			window->branch(isFalse, faceFound_);
			{
				window->parallel([](cv::UMat& fout, const cv::UMat& in, cv::UMat& lh) {
					if (side_by_side) {
						//create side-by-side view without a result (using the input image for both sides)
						fout = cv::Scalar::all(0);
						cv::resize(in, lh, cv::Size(0, 0), 0.5, 0.5);
						lh.copyTo(fout(cv::Rect(0, 0, lh.size().width, lh.size().height)));
						lh.copyTo(fout(cv::Rect(lh.size().width, 0, lh.size().width, lh.size().height)));
					} else {
						in.copyTo(fout);
					}
				}, frameOut_, input_, lhalf_);
			}
			window->endbranch(isFalse, faceFound_);

			window->branch(always);
			{
				//write the result to the framebuffer
				window->fb([](cv::UMat &frameBuffer, const cv::UMat& f) {
					cvtColor(f, frameBuffer, cv::COLOR_BGR2BGRA);
				}, frameOut_);
			}
			window->endbranch(always);

		} catch (std::exception &ex) {
			cerr << ex.what() << endl;
		}
	}
};

#ifndef __EMSCRIPTEN__
int main(int argc, char **argv) {
    if (argc != 2) {
        cerr << "Usage: beauty-demo <input-video-file>" << endl;
        exit(1);
    }
#else
int main() {
#endif
    using namespace cv::v4d;
    cv::Ptr<V4D> window = V4D::make(WIDTH, HEIGHT, "Beautification Demo", ALL, OFFSCREEN);
//    window->printSystemInfo();
    window->setStretching(stretch);

//    if (!OFFSCREEN) {
//        window->imgui([window](ImGuiContext* ctx){
//            using namespace ImGui;
//            SetCurrentContext(ctx);
//            Begin("Effect");
//            Text("Display");
//            Checkbox("Side by side", &side_by_side);
//            if(Checkbox("Stetch", &stretch)) {
//                window->setStretching(true);
//            } else
//                window->setStretching(false);
//
//    #ifndef __EMSCRIPTEN__
//            if(Button("Fullscreen")) {
//                window->setFullscreen(!window->isFullscreen());
//            };
//    #endif
//
//            if(Button("Offscreen")) {
//                window->setVisible(!window->isVisible());
//            };
//
//            Text("Face Skin");
//            SliderInt("Blur", &blur_skin_kernel_size, 0, 128);
//            SliderFloat("Saturation", &skin_saturation, 0.0f, 100.0f);
//            SliderFloat("Contrast", &skin_contrast, 0.0f, 1.0f);
//            Text("Eyes and Lips");
//            SliderFloat("Saturation ", &eyes_and_lips_saturation, 0.0f, 100.0f);
//            End();
//        });
//    }
#ifndef __EMSCRIPTEN__
    auto src = makeCaptureSource(window, argv[1]);
    window->setSource(src);
//    Sink sink = makeWriterSink(window, OUTPUT_FILENAME, src.fps(), cv::Size(WIDTH, HEIGHT));
//    window->setSink(sink);
#else
    auto src = makeCaptureSource(window);
    window->setSource(src);
#endif

    window->run<BeautyDemoPlan>(0);

    return 0;
}
