#include "util/yaml.hpp"

namespace cv::slam {
namespace util {

std::vector<std::vector<float>> get_rectangles(const YAML::Node& node) {
    auto rectangles = node.as<std::vector<std::vector<float>>>(std::vector<std::vector<float>>());
    for (const auto& v : rectangles) {
        if (v.size() != 4) {
            throw std::runtime_error("mask rectangle must contain four parameters");
        }
        if (v.at(0) >= v.at(1)) {
            throw std::runtime_error("x_max must be greater than x_min");
        }
        if (v.at(2) >= v.at(3)) {
            throw std::runtime_error("y_max must be greater than x_min");
        }
    }
    return rectangles;
}

} // namespace util
} // namespace cv::slam
