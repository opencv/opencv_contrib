// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/optflow.hpp>

#include <cmath>
#include <vector>
#include <set>
#include <string>
#include <random>
#include <tuple>
#include <array>
#include <utility>

using std::vector;
using std::string;

using namespace cv::v4d;

class OptflowDemoPlan : public Plan {
public:
	using Plan::Plan;
private:
	//How the background will be visualized
	enum BackgroundModes {
	    GREY,
	    COLOR,
	    VALUE,
	    BLACK
	};

	//Post-processing modes for the foreground
	enum PostProcModes {
	    GLOW,
	    BLOOM,
	    DISABLED
	};

	static struct Params {
		// Generate the foreground at this scale.
		float fgScale_ = 0.5f;
		// On every frame the foreground loses on brightness. Specifies the loss in percent.
		float fgLoss_ = 1;
		//Convert the background to greyscale
		BackgroundModes backgroundMode_ = GREY;
		// Peak thresholds for the scene change detection. Lowering them makes the detection more sensitive but
		// the default should be fine.
		float sceneChangeThresh_ = 0.29f;
		float sceneChangeThreshDiff_ = 0.1f;
		// The theoretical maximum number of points to track which is scaled by the density of detected points
		// and therefor is usually much smaller.
		int maxPoints_ = 300000;
		// How many of the tracked points to lose intentionally, in percent.
		float pointLoss_ = 20;
		// The theoretical maximum size of the drawing stroke which is scaled by the area of the convex hull
		// of tracked points and therefor is usually much smaller.
		int maxStroke_ = 6;
		// Blue, green, red and alpha. All from 0.0f to 1.0f
		cv::Scalar_<float> effectColor_ = {0.4f, 0.75f, 1.0f, 0.15f};
		//display on-screen FPS
		bool showFps_ = true;
		//Stretch frame buffer to window size
		bool stretch_ = false;
		//The post processing mode
		PostProcModes postProcMode_ = GLOW;
		// Intensity of glow or bloom defined by kernel size. The default scales with the image diagonal.
		int glowKernelSize_ = 0;
		//The lightness selection threshold
		int bloomThresh_ = 210;
		//The intensity of the bloom filter
		float bloomGain_ = 3;
	} params_;

	struct Cache {
		cv::Mat element_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(1, 1));

		vector<cv::KeyPoint> tmpKeyPoints_;

		float last_movement_ = 0;

		vector<cv::Point2f> hull_, prevPoints_, nextPoints_, newPoints_;
		vector<cv::Point2f> upPrevPoints_, upNextPoints_;
		std::vector<uchar> status_;
		std::vector<float> err_;
		std::random_device rd_;
		std::mt19937 rng_;

		cv::UMat bgr_;
		cv::UMat hls_;
		cv::UMat ls16_;
		cv::UMat ls_;
		cv::UMat bblur_;
		std::vector<cv::UMat> hlsChannels_;

		cv::UMat high_;
		cv::UMat low_;
		cv::UMat gblur_;
		cv::UMat dst16_;

	    cv::UMat tmp_;
	    cv::UMat post_;
	    cv::UMat backgroundGrey_;
	    vector<cv::UMat> channels_;
	    cv::UMat localFg_;
	} cache_;

	//BGRA
	cv::UMat background_, down_, frame_;
    inline static cv::UMat foreground_;
	//BGR
	cv::UMat result_;
	//GREY
	cv::UMat downPrevGrey_, downNextGrey_, downMotionMaskGrey_;
	vector<cv::Point2f> detectedPoints_;

	cv::Ptr<cv::BackgroundSubtractor> bg_subtractor_ = cv::createBackgroundSubtractorMOG2(100, 16.0, false);
	cv::Ptr<cv::FastFeatureDetector> detector_ = cv::FastFeatureDetector::create(1, false);

    //Uses background subtraction to generate a "motion mask"
	static void prepare_motion_mask(const cv::UMat& srcGrey, cv::UMat& motionMaskGrey, cv::Ptr<cv::BackgroundSubtractor> bg_subtractor, Cache& cache) {
	    bg_subtractor->apply(srcGrey, motionMaskGrey);
	    //Surpress speckles
	    cv::morphologyEx(motionMaskGrey, motionMaskGrey, cv::MORPH_OPEN, cache.element_, cv::Point(cache.element_.cols >> 1, cache.element_.rows >> 1), 2, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());
	}

	//Detect points to track
	static void detect_points(const cv::UMat& srcMotionMaskGrey, vector<cv::Point2f>& points, cv::Ptr<cv::FastFeatureDetector> detector, Cache& cache) {
	    detector->detect(srcMotionMaskGrey, cache.tmpKeyPoints_);

	    points.clear();
	    for (const auto &kp : cache.tmpKeyPoints_) {
	        points.push_back(kp.pt);
	    }
	}

	//Detect extrem changes in scene content and report it
	static bool detect_scene_change(const cv::UMat& srcMotionMaskGrey, const Params& params, Cache& cache) {
	    float movement = cv::countNonZero(srcMotionMaskGrey) / float(srcMotionMaskGrey.cols * srcMotionMaskGrey.rows);
	    float relation = movement > 0 && cache.last_movement_ > 0 ? std::max(movement, cache.last_movement_) / std::min(movement, cache.last_movement_) : 0;
	    float relM = relation * log10(1.0f + (movement * 9.0));
	    float relLM = relation * log10(1.0f + (cache.last_movement_ * 9.0));

	    bool result = !((movement > 0 && cache.last_movement_ > 0 && relation > 0)
	            && (relM < params.sceneChangeThresh_ && relLM < params.sceneChangeThresh_ && fabs(relM - relLM) < params.sceneChangeThreshDiff_));
	    cache.last_movement_ = (cache.last_movement_ + movement) / 2.0f;
	    return result;
	}

	//Visualize the sparse optical flow
	static void visualize_sparse_optical_flow(const cv::UMat &prevGrey, const cv::UMat &nextGrey, const vector<cv::Point2f> &detectedPoints, const Params& params, Cache& cache) {
	    //less then 5 points is a degenerate case (e.g. the corners of a video frame)
	    if (detectedPoints.size() > 4) {
	        cv::convexHull(detectedPoints, cache.hull_);
	        float area = cv::contourArea(cache.hull_);
	        //make sure the area of the point cloud is positive
	        if (area > 0) {
	            float density = (detectedPoints.size() / area);
	            //stroke size is biased by the area of the point cloud
	            float strokeSize = params.maxStroke_ * pow(area / (nextGrey.cols * nextGrey.rows), 0.33f);
	            //max points is biased by the densitiy of the point cloud
	            size_t currentMaxPoints = ceil(density * params.maxPoints_);

	            //lose a number of random points specified by pointLossPercent
	            std::shuffle(cache.prevPoints_.begin(), cache.prevPoints_.end(), cache.rng_);
	            cache.prevPoints_.resize(ceil(cache.prevPoints_.size() * (1.0f - (params.pointLoss_ / 100.0f))));

	            //calculate how many newly detected points to add
	            size_t copyn = std::min(detectedPoints.size(), (size_t(std::ceil(currentMaxPoints)) - cache.prevPoints_.size()));
	            if (cache.prevPoints_.size() < currentMaxPoints) {
	                std::copy(detectedPoints.begin(), detectedPoints.begin() + copyn, std::back_inserter(cache.prevPoints_));
	            }

	            //calculate the sparse optical flow
	            cv::calcOpticalFlowPyrLK(prevGrey, nextGrey, cache.prevPoints_, cache.nextPoints_, cache.status_, cache.err_);
	            cache.newPoints_.clear();
	            if (cache.prevPoints_.size() > 1 && cache.nextPoints_.size() > 1) {
	                //scale the points to original size
	            	cache.upNextPoints_.clear();
	            	cache.upPrevPoints_.clear();
	                for (cv::Point2f pt : cache.prevPoints_) {
	                	cache.upPrevPoints_.push_back(pt /= params.fgScale_);
	                }

	                for (cv::Point2f pt : cache.nextPoints_) {
	                	cache.upNextPoints_.push_back(pt /= params.fgScale_);
	                }

	                using namespace cv::v4d::nvg;
	                //start drawing
	                beginPath();
	                strokeWidth(strokeSize);
	                strokeColor(params.effectColor_ * 255.0);

	                for (size_t i = 0; i < cache.prevPoints_.size(); i++) {
	                    if (cache.status_[i] == 1 //point was found in prev and new set
	                            && cache.err_[i] < (1.0 / density) //with a higher density be more sensitive to the feature error
	                            && cache.upNextPoints_[i].y >= 0 && cache.upNextPoints_[i].x >= 0 //check bounds
	                            && cache.upNextPoints_[i].y < nextGrey.rows / params.fgScale_ && cache.upNextPoints_[i].x < nextGrey.cols / params.fgScale_ //check bounds
	                            ) {
	                        float len = hypot(fabs(cache.upPrevPoints_[i].x - cache.upNextPoints_[i].x), fabs(cache.upPrevPoints_[i].y - cache.upNextPoints_[i].y));
	                        //upper and lower bound of the flow vector length
	                        if (len > 0 && len < sqrt(area)) {
	                            //collect new points
	                        	cache.newPoints_.push_back(cache.nextPoints_[i]);
	                            //the actual drawing operations
	                            moveTo(cache.upNextPoints_[i].x, cache.upNextPoints_[i].y);
	                            lineTo(cache.upPrevPoints_[i].x, cache.upPrevPoints_[i].y);
	                        }
	                    }
	                }
	                //end drawing
	                stroke();
	            }
	            cache.prevPoints_ = cache.newPoints_;
	        }
	    }
	}

	//Bloom post-processing effect
	static void bloom(const cv::UMat& src, cv::UMat &dst, Cache& cache, int ksize = 3, int threshValue = 235, float gain = 4) {
	    //remove alpha channel
	    cv::cvtColor(src, cache.bgr_, cv::COLOR_BGRA2RGB);
	    //convert to hls
	    cv::cvtColor(cache.bgr_, cache.hls_, cv::COLOR_BGR2HLS);
	    //split channels
	    cv::split(cache.hls_, cache.hlsChannels_);
	    //invert lightness
	    cv::bitwise_not(cache.hlsChannels_[2], cache.hlsChannels_[2]);
	    //multiply lightness and saturation
	    cv::multiply(cache.hlsChannels_[1], cache.hlsChannels_[2], cache.ls16_, 1, CV_16U);
	    //normalize
	    cv::divide(cache.ls16_, cv::Scalar(255.0), cache.ls_, 1, CV_8U);
	    //binary threhold according to threshValue
	    cv::threshold(cache.ls_, cache.bblur_, threshValue, 255, cv::THRESH_BINARY);
	    //blur
	    cv::boxFilter(cache.bblur_, cache.bblur_, -1, cv::Size(ksize, ksize), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
	    //convert to BGRA
	    cv::cvtColor(cache.bblur_, cache.bblur_, cv::COLOR_GRAY2BGRA);
	    //add src and the blurred L-S-product according to gain
	    addWeighted(src, 1.0, cache.bblur_, gain, 0, dst);
	}

	//Glow post-processing effect
	static void glow_effect(const cv::UMat &src, cv::UMat &dst, const int ksize, Cache& cache) {
	    cv::bitwise_not(src, dst);

	    //Resize for some extra performance
	    cv::resize(dst, cache.low_, cv::Size(), 0.5, 0.5);
	    //Cheap blur
	    cv::boxFilter(cache.low_, cache.gblur_, -1, cv::Size(ksize, ksize), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
	    //Back to original size
	    cv::resize(cache.gblur_, cache.high_, src.size());

	    //Multiply the src image with a blurred version of itself
	    cv::multiply(dst, cache.high_, cache.dst16_, 1, CV_16U);
	    //Normalize and convert back to CV_8U
	    cv::divide(cache.dst16_, cv::Scalar::all(255.0), dst, 1, CV_8U);

	    cv::bitwise_not(dst, dst);
	}

	//Compose the different layers into the final image
	static void composite_layers(cv::UMat& background, cv::UMat& foreground, const cv::UMat& frameBuffer, cv::UMat& dst, const Params& params, Cache& cache) {
	    //Lose a bit of foreground brightness based on fgLossPercent
	    cv::subtract(foreground, cv::Scalar::all(255.0f * (params.fgLoss_ / 100.0f)), foreground);
	    //Add foreground an the current framebuffer into foregound
	    cv::add(foreground, frameBuffer, foreground);

	    //Dependin on bgMode prepare the background in different ways
	    switch (params.backgroundMode_) {
	    case GREY:
	        cv::cvtColor(background, cache.backgroundGrey_, cv::COLOR_BGRA2GRAY);
	        cv::cvtColor(cache.backgroundGrey_, background, cv::COLOR_GRAY2BGRA);
	        break;
	    case VALUE:
	        cv::cvtColor(background, cache.tmp_, cv::COLOR_BGRA2BGR);
	        cv::cvtColor(cache.tmp_, cache.tmp_, cv::COLOR_BGR2HSV);
	        split(cache.tmp_, cache.channels_);
	        cv::cvtColor(cache.channels_[2], background, cv::COLOR_GRAY2BGRA);
	        break;
	    case COLOR:
	        break;
	    case BLACK:
	        background = cv::Scalar::all(0);
	        break;
	    default:
	        break;
	    }

	    //Depending on ppMode perform post-processing
	    switch (params.postProcMode_) {
	    case GLOW:
	        glow_effect(foreground, cache.post_, params.glowKernelSize_, cache);
	        break;
	    case BLOOM:
	        bloom(foreground, cache.post_, cache, params.glowKernelSize_, params.bloomThresh_, params.bloomGain_);
	        break;
	    case DISABLED:
	        foreground.copyTo(cache.post_);
	        break;
	    default:
	        break;
	    }

	    //Add background and post-processed foreground into dst
	    cv::add(background, cache.post_, dst);
	}
public:
    OptflowDemoPlan(const cv::Rect& viewport) : Plan(viewport) {
		Global::registerShared(params_);
		Global::registerShared(foreground_);
    }

    OptflowDemoPlan(const cv::Size& sz) : OptflowDemoPlan(cv::Rect(0,0, sz.width, sz.height)) {
    }

    virtual void gui(cv::Ptr<V4D> window) override {
		window->imgui([](cv::Ptr<V4D> win, ImGuiContext* ctx, Params& params){
	        using namespace ImGui;
	        SetCurrentContext(ctx);

	        Begin("Effects");
	        Text("Foreground");
	        SliderFloat("Scale", &params.fgScale_, 0.1f, 4.0f);
	        SliderFloat("Loss", &params.fgLoss_, 0.1f, 99.9f);
	        Text("Background");
	        thread_local const char* bgm_items[4] = {"Grey", "Color", "Value", "Black"};
	        thread_local int* bgm = (int*)&params.backgroundMode_;
	        ListBox("Mode", bgm, bgm_items, 4, 4);
	        Text("Points");
	        SliderInt("Max. Points", &params.maxPoints_, 10, 1000000);
	        SliderFloat("Point Loss", &params.pointLoss_, 0.0f, 100.0f);
	        Text("Optical flow");
	        SliderInt("Max. Stroke Size", &params.maxStroke_, 1, 100);
	        ColorPicker4("Color", params.effectColor_.val);
	        End();

	        Begin("Post Processing");
	        thread_local const char* ppm_items[3] = {"Glow", "Bloom", "None"};
	        thread_local int* ppm = (int*)&params.postProcMode_;
	        ListBox("Effect",ppm, ppm_items, 3, 3);
	        SliderInt("Kernel Size",&params.glowKernelSize_, 1, 63);
	        SliderFloat("Gain", &params.bloomGain_, 0.1f, 20.0f);
	        End();

	        Begin("Settings");
	        Text("Scene Change Detection");
	        SliderFloat("Threshold", &params.sceneChangeThresh_, 0.1f, 1.0f);
	        SliderFloat("Threshold Diff", &params.sceneChangeThreshDiff_, 0.1f, 1.0f);
	        End();

			Begin("Window");
			if(Checkbox("Show FPS", &params.showFps_)) {
				win->setShowFPS(params.showFps_);
			}
			if(Checkbox("Stretch", &params.stretch_)) {
				win->setStretching(params.stretch_);
			}

			if(Button("Fullscreen")) {
				win->setFullscreen(!win->isFullscreen());
			};

			if(Button("Offscreen")) {
				win->setVisible(!win->isVisible());
			};

			End();
	    }, params_);
	}

	virtual void setup(cv::Ptr<V4D> window) override {
		cache_.rng_ = std::mt19937(cache_.rd_());
		window->setStretching(params_.stretch_);
		window->once([](const cv::Size& sz, Params& params, cv::UMat& foreground){
			int diag = hypot(double(sz.width), double(sz.height));
			params.glowKernelSize_ = std::max(int(diag / 150 % 2 == 0 ? diag / 150 + 1 : diag / 150), 1);
			params.effectColor_[3] /= (Global::workers_started() - 1);
			foreground.create(sz, CV_8UC4);
			foreground = cv::Scalar::all(0);
		}, size(), params_, foreground_);
	}

	virtual void infer(cv::Ptr<V4D> window) override {
		window->capture();

        window->fb([](const cv::UMat& framebuffer, const cv::Rect& viewport, cv::UMat& frame) {
            framebuffer(viewport).copyTo(frame);
        }, viewport(), frame_);

        window->plain([](const cv::UMat& frame, cv::UMat& background) {
            frame.copyTo(background);
        }, frame_, background_);

		window->fb([](const cv::UMat& framebuffer, const cv::Rect& viewport, cv::UMat& d, cv::UMat& b, const Params& params) {
			Params p = Global::safe_copy(params);
			//resize to foreground scale
			cv::resize(framebuffer(viewport), d, cv::Size(viewport.width * p.fgScale_, viewport.height * p.fgScale_));
			//save video background
			framebuffer(viewport).copyTo(b);
		}, viewport(), down_, background_, params_);

		window->plain([](const cv::UMat& d, cv::UMat& dng, cv::UMat& dmmg, std::vector<cv::Point2f>& dp, cv::Ptr<cv::BackgroundSubtractor>& bg_subtractor, cv::Ptr<cv::FastFeatureDetector>& detector, Cache& cache){
			cv::cvtColor(d, dng, cv::COLOR_RGBA2GRAY);
			//Subtract the background to create a motion mask
			prepare_motion_mask(dng, dmmg, bg_subtractor, cache);
			//Detect trackable points in the motion mask
			detect_points(dmmg, dp, detector, cache);
		}, down_, downNextGrey_, downMotionMaskGrey_, detectedPoints_, bg_subtractor_, detector_, cache_);

		window->nvg([](const cv::UMat& dmmg, const cv::UMat& dpg, const cv::UMat& dng, const std::vector<cv::Point2f>& dp, const Params& params, Cache& cache) {
			const Params p = Global::safe_copy(params);
			cv::v4d::nvg::clear();
			if (!dpg.empty()) {
				//We don't want the algorithm to get out of hand when there is a scene change, so we suppress it when we detect one.
				if (!detect_scene_change(dmmg, p, cache)) {
					//Visualize the sparse optical flow using nanovg
					visualize_sparse_optical_flow(dpg, dng, dp, p, cache);
				}
			}
		}, downMotionMaskGrey_, downPrevGrey_, downNextGrey_, detectedPoints_, params_, cache_);

		window->plain([](cv::UMat& dpg, const cv::UMat& dng) {
			dpg = dng.clone();
		}, downPrevGrey_, downNextGrey_);

        window->fb([](const cv::UMat& framebuffer, const cv::Rect& viewport, cv::UMat& frame) {
            framebuffer(viewport).copyTo(frame);
        }, viewport(), frame_);

        window->plain([](cv::UMat& frame, cv::UMat& background, cv::UMat& foreground, const Params& params, Cache& cache) {
            //Put it all together (OpenCL)
            Global::Scope scope(foreground);
            copy_shared(foreground, cache.localFg_);
            composite_layers(background, cache.localFg_, frame, frame, params, cache);
            copy_shared(cache.localFg_, foreground);
        }, frame_, background_, foreground_, params_, cache_);

        window->fb([](cv::UMat& framebuffer, const cv::Rect& viewport, const cv::UMat& frame) {
            frame.copyTo(framebuffer(viewport));
        }, viewport(), frame_);

        window->write();
	}
};

OptflowDemoPlan::Params OptflowDemoPlan::params_;

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: optflow-demo <input-video-file>" << endl;
        exit(1);
    }

    cv::Ptr<OptflowDemoPlan> plan = new OptflowDemoPlan(cv::Size(1280, 720));
	cv::Ptr<V4D> window = V4D::make(plan->size(), "Sparse Optical Flow Demo", ALL);

	auto src = makeCaptureSource(window, argv[1]);
	auto sink = makeWriterSink(window, "optflow-demo.mkv", src->fps(), plan->size());
	window->setSource(src);
	window->setSink(sink);

	window->run(plan, 5);
    return 0;
}
