#include "util/yaml.hpp"

namespace cv::slam {
namespace util {

std::vector<std::vector<float>> get_rectangles(const cv::FileNode& node) {
    std::vector<std::vector<float>> rectangles;
    if (node.empty() || node.type() != cv::FileNode::SEQ) {
        return rectangles;
    }
    for (cv::FileNodeIterator it = node.begin(); it != node.end(); ++it) {
        cv::FileNode rect_node = *it;
        std::vector<float> rect;
        if (rect_node.type() == cv::FileNode::SEQ) {
            for (cv::FileNodeIterator fit = rect_node.begin(); fit != rect_node.end(); ++fit) {
                rect.push_back(static_cast<float>(*fit));
            }
        }
        if (rect.size() != 4) {
            throw std::runtime_error("mask rectangle must contain four parameters");
        }
        if (rect.at(0) >= rect.at(1)) {
            throw std::runtime_error("x_max must be greater than x_min");
        }
        if (rect.at(2) >= rect.at(3)) {
            throw std::runtime_error("y_max must be greater than x_min");
        }
        rectangles.push_back(rect);
    }
    return rectangles;
}

} // namespace util
} // namespace cv::slam
