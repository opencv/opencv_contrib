#ifndef SLAM_UTIL_YAML_H
#define SLAM_UTIL_YAML_H

#include <string>

#include <opencv2/core/persistence.hpp>
#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {
namespace util {

//! OpenCV FileNode compatibility helpers (replaces yaml-cpp .as<T>())

template <typename T>
inline T yaml_get_val(const cv::FileNode& node, const std::string& key, const T& default_val) {
    cv::FileNode child = node[key];
    return child.empty() ? default_val : static_cast<T>(child);
}

template <>
inline bool yaml_get_val<bool>(const cv::FileNode& node, const std::string& key, const bool& default_val) {
    cv::FileNode child = node[key];
    if (child.empty()) return default_val;
    if (child.isInt()) return static_cast<int>(child) != 0;
    if (child.isString()) {
        std::string s = static_cast<std::string>(child);
        return (s == "true" || s == "1" || s == "True" || s == "TRUE");
    }
    return !child.empty();
}

template <>
inline unsigned int yaml_get_val<unsigned int>(const cv::FileNode& node, const std::string& key, const unsigned int& default_val) {
    cv::FileNode child = node[key];
    return child.empty() ? default_val : static_cast<unsigned int>(static_cast<int>(child));
}

template <typename T>
inline std::vector<T> yaml_get_vec(const cv::FileNode& node, const std::string& key) {
    cv::FileNode child = node[key];
    std::vector<T> result;
    if (!child.empty() && child.type() == cv::FileNode::SEQ) {
        for (cv::FileNodeIterator it = child.begin(); it != child.end(); ++it) {
            result.push_back(static_cast<T>(*it));
        }
    }
    return result;
}

template <typename T>
inline T yaml_get(const cv::FileNode& node, const T& default_val) {
    return node.empty() ? default_val : static_cast<T>(node);
}

template <typename T>
inline T yaml_get_req(const cv::FileNode& node, const std::string& key) {
    cv::FileNode child = node[key];
    if (child.empty()) {
        throw std::runtime_error("Missing required config key: " + key);
    }
    return static_cast<T>(child);
}

template <>
inline unsigned int yaml_get_req<unsigned int>(const cv::FileNode& node, const std::string& key) {
    cv::FileNode child = node[key];
    if (child.empty()) {
        throw std::runtime_error("Missing required config key: " + key);
    }
    return static_cast<unsigned int>(static_cast<int>(child));
}

template <>
inline float yaml_get_req<float>(const cv::FileNode& node, const std::string& key) {
    cv::FileNode child = node[key];
    if (child.empty()) {
        throw std::runtime_error("Missing required config key: " + key);
    }
    return static_cast<float>(static_cast<double>(child));
}

inline std::string yaml_get_req_str(const cv::FileNode& node, const std::string& key) {
    cv::FileNode child = node[key];
    if (child.empty()) {
        throw std::runtime_error("Missing required config key: " + key);
    }
    return (std::string)child;
}

inline cv::FileNode yaml_optional_ref(const cv::FileNode& ref_node, const std::string& key) {
    cv::FileNode child = ref_node[key];
    return child.empty() ? cv::FileNode() : child;
}

std::vector<std::vector<float>> get_rectangles(const cv::FileNode& node);

} // namespace util
} // namespace cv::slam

#endif // SLAM_UTIL_YAML_H
