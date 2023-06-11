// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include <opencv2/v4d/v4d.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/objdetect.hpp>

#include <string>

constexpr unsigned int WIDTH = 1280;
constexpr unsigned int HEIGHT = 720;
const unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));
constexpr unsigned int DOWNSIZE_WIDTH = 640;
constexpr unsigned int DOWNSIZE_HEIGHT = 360;
constexpr double WIDTH_SCALE = double(WIDTH) / DOWNSIZE_WIDTH;
constexpr double HEIGHT_SCALE = double(HEIGHT) / DOWNSIZE_HEIGHT;
constexpr bool OFFSCREEN = false;
#ifndef __EMSCRIPTEN__
constexpr const char* OUTPUT_FILENAME = "pedestrian-demo.mkv";
#endif

// Intensity of blur defined by kernel size. The default scales with the image diagonal.
const int BLUR_KERNEL_SIZE = std::max(int(DIAG / 200 % 2 == 0 ? DIAG / 200 + 1 : DIAG / 200), 1);

using std::cerr;
using std::endl;
using std::vector;
using std::string;

cv::Ptr<cv::v4d::V4D> window;
cv::HOGDescriptor hog;

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

static void composite_layers(const cv::UMat background, const cv::UMat frameBuffer, cv::UMat dst, int blurKernelSize) {
    static cv::UMat blur;

    cv::boxFilter(frameBuffer, blur, -1, cv::Size(blurKernelSize, blurKernelSize), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
    cv::add(background, blur, dst);
}

static bool iteration() {
    //BGRA
    static cv::UMat background;
    //RGB
    static cv::UMat videoFrame, videoFrameDown;
    //GREY
    static cv::UMat videoFrameDownGrey;

    static std::vector<cv::Rect> locations;
    static vector<vector<double>> boxes;
    static vector<double> probs;

    static cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
    static bool trackerInitalized = false;
    static cv::Rect trackedFace;
    static cv::Rect lastTracked(0,0,1,1);

    static bool redetect = true;

    if(!window->capture())
        return false;

    window->fb([&](cv::UMat& frameBuffer){
        cvtColor(frameBuffer,videoFrame,cv::COLOR_BGRA2RGB);
        cv::resize(videoFrame, videoFrameDown, cv::Size(DOWNSIZE_WIDTH, DOWNSIZE_HEIGHT));
    });

    cv::cvtColor(videoFrameDown, videoFrameDownGrey, cv::COLOR_RGB2GRAY);
    cv::cvtColor(videoFrame, background, cv::COLOR_RGB2BGRA);

    cv::Rect tracked = cv::Rect(0,0,1,1);

    if (!trackerInitalized || redetect || !tracker->update(videoFrameDown, tracked)) {
        redetect = false;
        tracked = cv::Rect(0,0,1,1);
        hog.detectMultiScale(videoFrameDownGrey, locations, 0, cv::Size(), cv::Size(), 1.025, 2.0, false);

        if (!locations.empty()) {
            boxes.clear();
            probs.clear();
            for (const auto &rect : locations) {
                boxes.push_back( { double(rect.x), double(rect.y), double(rect.x + rect.width), double(rect.y + rect.height) });
                probs.push_back(1.0);
            }

            vector<bool> keep = non_maximal_suppression(&boxes, &probs, 0.1);
            for (size_t i = 0; i < keep.size(); ++i) {
                if (keep[i]) {
                    if(tracked.width * tracked.height < locations[i].width * locations[i].height) {
                        tracked = locations[i];
                    }
                }
            }

            if(!trackerInitalized) {
                tracker->init(videoFrameDown, tracked);
                trackerInitalized = true;
            }

            if(tracked.width == 1 && tracked.height == 1) {
                redetect = true;
            } else {
                lastTracked = tracked;
            }
        }
    }

    window->nvg([&](const cv::Size& sz) {
        using namespace cv::v4d::nvg;
        clear();
        beginPath();
        strokeWidth(std::fmax(2.0, sz.width / 960.0));
        strokeColor(cv::v4d::colorConvert(cv::Scalar(0, 127, 255, 200), cv::COLOR_HLS2BGR));
        float width = tracked.width * WIDTH_SCALE;
        float height = tracked.height * HEIGHT_SCALE;
        float cx = tracked.x * WIDTH_SCALE + (width / 2.0f);
        float cy = tracked.y * HEIGHT_SCALE + (height / 2.0f);
        ellipse(cx, cy, width / 2.0f, height / 2.0f);
        stroke();
    });

    window->fb([&](cv::UMat& frameBuffer){
        //Put it all together
        composite_layers(background, frameBuffer, frameBuffer, BLUR_KERNEL_SIZE);
    });

    window->write();

    //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
    return window->display();
}

#ifndef __EMSCRIPTEN__
int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: pedestrian-demo <video-input>" << endl;
        exit(1);
    }
#else
int main() {
#endif
    using namespace cv::v4d;
    window = V4D::make(cv::Size(WIDTH, HEIGHT), cv::Size(), "Pedestrian Demo", OFFSCREEN);
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    window->printSystemInfo();

#ifndef __EMSCRIPTEN__
    Source src = makeCaptureSource(argv[1]);
    window->setSource(src);

    Sink sink = makeWriterSink(OUTPUT_FILENAME, cv::VideoWriter::fourcc('V', 'P', '9', '0'),
            src.fps(), cv::Size(WIDTH, HEIGHT));
    window->setSink(sink);
#else
    Source src = makeCaptureSource(WIDTH, HEIGHT, window);
    window->setSource(src);
#endif

    window->run(iteration);

    return 0;
}
