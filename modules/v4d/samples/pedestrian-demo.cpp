// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/objdetect.hpp>

#include <string>

using std::vector;
using std::string;

using namespace cv::v4d;

class PedestrianDemoPlan : public Plan {
	unsigned long diag_;
	cv::Size downSize_;
	cv::Size_<float> scale_;
	int blurKernelSize_;

	struct Cache {
		cv::UMat blur_;
	} cache_;
    //BGRA
    cv::UMat background_;
    //RGB
    cv::UMat videoFrame_, videoFrameDown_;
    //GREY
    cv::UMat videoFrameDownGrey_;

    //detected pedestrian locations rectangles
    std::vector<cv::Rect> locations_;
    //detected pedestrian locations as boxes
    vector<vector<double>> boxes_;
    //probability of detected object being a pedestrian - currently always set to 1.0
    vector<double> probs_;
    //Faster tracking parameters
    cv::TrackerKCF::Params params_;
    //KCF tracker used instead of continous detection
    cv::Ptr<cv::Tracker> tracker_;
    cv::Rect tracked_ = cv::Rect(0,0,1,1);
    bool trackerInitialized_ = false;
    //If tracking fails re-detect
    bool redetect_ = true;
    //Descriptor used for pedestrian detection
    cv::HOGDescriptor hog_;

	constexpr static auto doRedect_ = [](const bool& trackerInit, const bool& redetect){ return !trackerInit || redetect; };
	constexpr static auto dontRedect_ = [](const bool& trackerInit, const bool& redetect){ return trackerInit && !redetect; };

	//adapted from cv::dnn_objdetect::InferBbox
	static inline bool pair_comparator(std::pair<double, size_t> l1, std::pair<double, size_t> l2) {
	    return l1.first > l2.first;
	}

	//adapted from cv::dnn_objdetect::InferBbox
	static void intersection_over_union(std::vector<std::vector<double> > *boxes, std::vector<double> *base_box, std::vector<double> *iou) {
	    double g_xmin = (*base_box)[0];
	    double g_ymin = (*base_box)[1];
	    double g_xmax = (*base_box)[2];
	    double g_ymax = (*base_box)[3];
	    double base_box_w = g_xmax - g_xmin;
	    double base_box_h = g_ymax - g_ymin;
	    for (size_t b = 0; b < (*boxes).size(); ++b) {
	        double xmin = std::max((*boxes)[b][0], g_xmin);
	        double ymin = std::max((*boxes)[b][1], g_ymin);
	        double xmax = std::min((*boxes)[b][2], g_xmax);
	        double ymax = std::min((*boxes)[b][3], g_ymax);

	        // Intersection
	        double w = std::max(static_cast<double>(0.0), xmax - xmin);
	        double h = std::max(static_cast<double>(0.0), ymax - ymin);
	        // Union
	        double test_box_w = (*boxes)[b][2] - (*boxes)[b][0];
	        double test_box_h = (*boxes)[b][3] - (*boxes)[b][1];

	        double inter_ = w * h;
	        double union_ = test_box_h * test_box_w + base_box_h * base_box_w - inter_;
	        (*iou)[b] = inter_ / (union_ + 1e-7);
	    }
	}

	//adapted from cv::dnn_objdetect::InferBbox
	static std::vector<bool> non_maximal_suppression(std::vector<std::vector<double> > *boxes, std::vector<double> *probs, const double threshold = 0.1) {
	    std::vector<bool> keep(((*probs).size()));
	    std::fill(keep.begin(), keep.end(), true);
	    std::vector<size_t> prob_args_sorted((*probs).size());

	    std::vector<std::pair<double, size_t> > temp_sort((*probs).size());
	    for (size_t tidx = 0; tidx < (*probs).size(); ++tidx) {
	        temp_sort[tidx] = std::make_pair((*probs)[tidx], static_cast<size_t>(tidx));
	    }
	    std::sort(temp_sort.begin(), temp_sort.end(), pair_comparator);

	    for (size_t idx = 0; idx < temp_sort.size(); ++idx) {
	        prob_args_sorted[idx] = temp_sort[idx].second;
	    }

	    for (std::vector<size_t>::iterator itr = prob_args_sorted.begin(); itr != prob_args_sorted.end() - 1; ++itr) {
	        size_t idx = itr - prob_args_sorted.begin();
	        std::vector<double> iou_(prob_args_sorted.size() - idx - 1);
	        std::vector<std::vector<double> > temp_boxes(iou_.size());
	        for (size_t bb = 0; bb < temp_boxes.size(); ++bb) {
	            std::vector<double> temp_box(4);
	            for (size_t b = 0; b < 4; ++b) {
	                temp_box[b] = (*boxes)[prob_args_sorted[idx + bb + 1]][b];
	            }
	            temp_boxes[bb] = temp_box;
	        }
	        intersection_over_union(&temp_boxes, &(*boxes)[prob_args_sorted[idx]], &iou_);
	        for (std::vector<double>::iterator _itr = iou_.begin(); _itr != iou_.end(); ++_itr) {
	            size_t iou_idx = _itr - iou_.begin();
	            if (*_itr > threshold) {
	                keep[prob_args_sorted[idx + iou_idx + 1]] = false;
	            }
	        }
	    }
	    return keep;
	}
    //post process and add layers together
    static void composite_layers(const cv::UMat background, const cv::UMat foreground, cv::UMat dst, int blurKernelSize, Cache& cache) {
        cv::boxFilter(foreground, cache.blur_, -1, cv::Size(blurKernelSize, blurKernelSize), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
        cv::add(background, cache.blur_, dst);
    }
public:
    PedestrianDemoPlan(const cv::Size& sz) : Plan(sz) {
    	int w = size().width;
    	int h = size().height;
    	diag_ = hypot(double(w), double(h));
    	downSize_ = { w / 2, h / 2 };
    	scale_ = { float(w) / downSize_.width, float(h) / downSize_.height };
    	blurKernelSize_ = std::max(int(diag_ / 200 % 2 == 0 ? diag_ / 200 + 1 : diag_ / 200), 1);
    }

    void setup(cv::Ptr<V4D> window) override {
		window->parallel([](cv::TrackerKCF::Params& params, cv::Ptr<cv::Tracker>& tracker, cv::HOGDescriptor& hog){
			params.desc_pca = cv::TrackerKCF::GRAY;
			params.compress_feature = false;
			params.compressed_size = 1;
			tracker = cv::TrackerKCF::create(params);
			hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
		}, params_, tracker_, hog_);
	}

	void infer(cv::Ptr<V4D> window) override {
		window->branch(always_);
		{
			window->capture();

			window->fb([](const cv::UMat& frameBuffer, cv::UMat& videoFrame){
				//copy video frame
				cvtColor(frameBuffer,videoFrame,cv::COLOR_BGRA2RGB);
				//downsample video frame for hog_ detection
			}, videoFrame_);

			window->parallel([](const cv::Size downSize, const cv::UMat& videoFrame, cv::UMat& videoFrameDown, cv::UMat& videoFrameDownGrey, cv::UMat& background){
				cv::resize(videoFrame, videoFrameDown, downSize);
				cv::cvtColor(videoFrameDown, videoFrameDownGrey, cv::COLOR_RGB2GRAY);
				cv::cvtColor(videoFrame, background, cv::COLOR_RGB2BGRA);
			}, downSize_, videoFrame_, videoFrameDown_, videoFrameDownGrey_, background_);
		}
		window->endbranch(always_);

		//Try to track the pedestrian (if we currently are tracking one), else re-detect using HOG descriptor
		window->branch(doRedect_, trackerInitialized_, redetect_);
		{
			window->parallel([](cv::HOGDescriptor& hog, bool& redetect, cv::UMat& videoFrameDownGrey, std::vector<cv::Rect>& locations, vector<vector<double>>& boxes, vector<double>& probs, cv::Ptr<cv::Tracker>& tracker, cv::Rect& tracked, bool& trackerInitialized){
				redetect = false;
				//Detect pedestrians
				hog.detectMultiScale(videoFrameDownGrey, locations, 0, cv::Size(), cv::Size(), 1.15, 2.0, false);

				if (!locations.empty()) {
					boxes.clear();
					probs.clear();
					//collect all found boxes
					for (const auto &rect : locations) {
						boxes.push_back( { double(rect.x), double(rect.y), double(rect.x + rect.width), double(rect.y + rect.height) });
						probs.push_back(1.0);
					}

					//use nms to filter overlapping boxes (https://medium.com/analytics-vidhya/non-max-suppression-nms-6623e6572536)
					vector<bool> keep = non_maximal_suppression(&boxes, &probs, 0.1);
					for (size_t i = 0; i < keep.size(); ++i) {
						if (keep[i]) {
							//only track the first pedestrian found
							tracked = locations[i];
							break;
						}
					}

					if(!trackerInitialized) {
		//            	initialize the tracker once
						tracker->init(videoFrameDownGrey, tracked);
						trackerInitialized = true;
					}
				}
			}, hog_, redetect_, videoFrameDownGrey_, locations_, boxes_, probs_, tracker_, tracked_, trackerInitialized_);
		}
		window->endbranch(doRedect_, trackerInitialized_, redetect_);
		window->branch(dontRedect_, trackerInitialized_, redetect_);
		{
			window->parallel([](bool& redetect, cv::UMat& videoFrameDownGrey, cv::Ptr<cv::Tracker>& tracker, cv::Rect& tracked){
				if(!tracker->update(videoFrameDownGrey, tracked)) {
					//detection failed - re-detect
					redetect = true;
				}
			}, redetect_, videoFrameDownGrey_, tracker_, tracked_);
		}
		window->endbranch(dontRedect_, trackerInitialized_, redetect_);

		window->branch(always_);
		{
		//Draw an ellipse around the tracked pedestrian
			window->nvg([](const cv::Size& sz, const cv::Size_<float> scale, cv::Rect& tracked) {
				using namespace cv::v4d::nvg;
				clear();
				beginPath();
				strokeWidth(std::fmax(2.0, sz.width / 960.0));
				strokeColor(cv::v4d::colorConvert(cv::Scalar(0, 127, 255, 200), cv::COLOR_HLS2BGR));
				float width = tracked.width * scale.width;
				float height = tracked.height * scale.height;
				float cx = tracked.x * scale.width + (width / 2.0f);
				float cy = tracked.y * scale.height + (height / 2.0f);
				ellipse(cx, cy, width / 2.0f, height / 2.0f);
				stroke();
			}, size(), scale_, tracked_);

			//Put it all together
			window->fb([](cv::UMat& frameBuffer, cv::UMat& bg, int blurKernelSize, Cache& cache){
				composite_layers(bg, frameBuffer, frameBuffer, blurKernelSize, cache);
			}, background_, blurKernelSize_, cache_);

			window->write();
		}
		window->endbranch(always_);
	}
};
int main(int argc, char **argv) {
	CV_UNUSED(argc);
	CV_UNUSED(argv);

    if (argc != 2) {
        std::cerr << "Usage: pedestrian-demo <video-input>" << endl;
        exit(1);
    }
#ifndef __EMSCRIPTEN__
    constexpr const char* OUTPUT_FILENAME = "pedestrian-demo.mkv";
    cv::Ptr<PedestrianDemoPlan> plan = new PedestrianDemoPlan(cv::Size(1280, 720));
#else
    cv::Ptr<PedestrianDemoPlan> plan = new PedestrianDemoPlan(cv::Size(960, 960));
#endif

    using namespace cv::v4d;
    cv::Ptr<V4D> window = V4D::make(plan->size(), "Pedestrian Demo", ALL);

    window->printSystemInfo();

#ifndef __EMSCRIPTEN__
    auto src = makeCaptureSource(window, argv[1]);
    window->setSource(src);

    auto sink = makeWriterSink(window, OUTPUT_FILENAME, src->fps(), plan->size());
    window->setSink(sink);
#else
    auto src = makeCaptureSource(WIDTH, HEIGHT, window);
    window->setSource(src);
#endif

    window->run(plan, 0);

    return 0;
}
