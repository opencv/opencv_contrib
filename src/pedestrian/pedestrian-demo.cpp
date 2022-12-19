#define CL_TARGET_OPENCL_VERSION 120

#include "../common/viz2d.hpp"
#include "../common/nvg.hpp"
#include "../common/util.hpp"

#include <string>

#include <opencv2/objdetect/objdetect.hpp>

constexpr unsigned int WIDTH = 1920;
constexpr unsigned int HEIGHT = 1080;
constexpr unsigned long DIAG = hypot(double(WIDTH), double(HEIGHT));
constexpr unsigned int DOWNSIZE_WIDTH = 640;
constexpr unsigned int DOWNSIZE_HEIGHT = 360;
constexpr double WIDTH_FACTOR = double(WIDTH) / DOWNSIZE_WIDTH;
constexpr double HEIGHT_FACTOR = double(HEIGHT) / DOWNSIZE_HEIGHT;
constexpr bool OFFSCREEN = false;
constexpr const int VA_HW_DEVICE_INDEX = 0;
constexpr const char* OUTPUT_FILENAME = "pedestrian-demo.mkv";

// On every frame the foreground loses on brightness. Specifies the loss in percent.
constexpr float fg_loss = 3;
// Intensity of blur defined by kernel size. The default scales with the image diagonal.
constexpr int BLUR_KERNEL_SIZE = std::max(int(DIAG / 200 % 2 == 0 ? DIAG / 200 + 1 : DIAG / 200), 1);

using std::cerr;
using std::endl;
using std::vector;
using std::string;

//adapted from cv::dnn_objdetect::InferBbox
static inline bool pair_comparator(std::pair<double, size_t> l1, std::pair<double, size_t> l2) {
    return l1.first > l2.first;
}

//adapted from cv::dnn_objdetect::InferBbox
void intersection_over_union(std::vector<std::vector<double> > *boxes, std::vector<double> *base_box, std::vector<double> *iou) {
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
std::vector<bool> non_maximal_suppression(std::vector<std::vector<double> > *boxes, std::vector<double> *probs, const double threshold = 0.1) {
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

void composite_layers(const cv::UMat background, const cv::UMat foreground, const cv::UMat frameBuffer, cv::UMat dst, int blurKernelSize, float fgLossPercent) {
    static cv::UMat blur;
    static cv::UMat backgroundGrey;

    cv::subtract(foreground, cv::Scalar::all(255.0f * (fgLossPercent / 100.0f)), foreground);
    cv::add(foreground, frameBuffer, foreground);
    cv::boxFilter(foreground, blur, -1, cv::Size(blurKernelSize, blurKernelSize), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
    cv::add(background, blur, dst);
}

int main(int argc, char **argv) {
    using namespace kb::viz2d;

    if (argc != 2) {
        std::cerr << "Usage: pedestrian-demo <video-input>" << endl;
        exit(1);
    }

    cv::Ptr<Viz2D> v2d = new Viz2D(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), OFFSCREEN, "Beauty Demo");
        print_system_info();
        if (!v2d->isOffscreen())
            v2d->setVisible(true);

        auto capture = v2d->makeVACapture(argv[1], VA_HW_DEVICE_INDEX);

        if (!capture.isOpened()) {
            cerr << "ERROR! Unable to open video input" << endl;
            exit(-1);
        }

        float fps = capture.get(cv::CAP_PROP_FPS);
        float width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
        float height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        v2d->makeVAWriter(OUTPUT_FILENAME, cv::VideoWriter::fourcc('V', 'P', '9', '0'), fps, cv::Size(width, height), VA_HW_DEVICE_INDEX);
        //BGRA
        cv::UMat background, foreground(HEIGHT, WIDTH, CV_8UC4, cv::Scalar::all(0));
        //RGB
        cv::UMat rgb, videoFrameUp, videoFrameDown;
        //GREY
        cv::UMat videoFrameDownGrey;

        cv::HOGDescriptor hog;
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        std::vector<cv::Rect> locations;
        std::vector<cv::Rect> maxLocations;
        vector<vector<double>> boxes;
        vector<double> probs;

        while (true) {
            if(!v2d->capture())
                       break;

            v2d->opencl([&](cv::UMat& frameBuffer){
                cvtColor(frameBuffer,videoFrameUp,cv::COLOR_BGRA2RGB);
                cv::resize(frameBuffer, videoFrameDown, cv::Size(DOWNSIZE_WIDTH, DOWNSIZE_HEIGHT));
                cv::cvtColor(videoFrameDown, videoFrameDownGrey, cv::COLOR_RGB2GRAY);
                cv::cvtColor(videoFrameUp, background, cv::COLOR_RGB2BGRA);
                hog.detectMultiScale(videoFrameDownGrey, locations, 0, cv::Size(), cv::Size(), 1.025, 2.0, false);
            });

            maxLocations.clear();
            if (!locations.empty()) {
                boxes.clear();
                probs.clear();
                for (const auto &rect : locations) {
                    boxes.push_back( { double(rect.x), double(rect.y), double(rect.x + rect.width), double(rect.y + rect.height) });
                    probs.push_back(1.0);
                }

                vector<bool> keep = non_maximal_suppression(&boxes, &probs, 0.1);

                for (size_t i = 0; i < keep.size(); ++i) {
                    if (keep[i])
                        maxLocations.push_back(locations[i]);
                }
            }

            v2d->nanovg([&](const cv::Size& sz) {
                v2d->clear();
                nvg::beginPath();
                nvg::strokeWidth(std::fmax(2.0, WIDTH / 960.0));
                nvg::strokeColor(kb::viz2d::color_convert(cv::Scalar(0, 127, 255, 200), cv::COLOR_HLS2BGR));
                for (size_t i = 0; i < maxLocations.size(); i++) {
                    nvg::rect(maxLocations[i].x * WIDTH_FACTOR, maxLocations[i].y * HEIGHT_FACTOR, maxLocations[i].width * WIDTH_FACTOR, maxLocations[i].height * HEIGHT_FACTOR);
                }
                nvg::stroke();
            });

            v2d->opencl([&](cv::UMat& frameBuffer){
                //Put it all together
                composite_layers(background, foreground, frameBuffer, frameBuffer, BLUR_KERNEL_SIZE, fg_loss);
            });

            update_fps(v2d, true);

            v2d->write();

            //If onscreen rendering is enabled it displays the framebuffer in the native window. Returns false if the window was closed.
            if(!v2d->display())
                break;
        }

    return 0;
}
