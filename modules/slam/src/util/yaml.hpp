#ifndef SLAM_UTIL_YAML_H
#define SLAM_UTIL_YAML_H

#include <string>

#include <yaml-cpp/yaml.h>
#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {
namespace util {

inline YAML::Node yaml_optional_ref(const YAML::Node& ref_node, const std::string& key) {
    return ref_node[key] ? ref_node[key] : YAML::Node();
}

std::vector<std::vector<float>> get_rectangles(const YAML::Node& node);

} // namespace util
} // namespace cv::slam

#endif // SLAM_UTIL_YAML_H
