#include "config.hpp"
#include "util/string.hpp"
#include "util/yaml.hpp"

#include <iostream>
#include <memory>
#include <fstream>
#include <sstream>
#include <cstdio>

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);

/**
 * @brief Pre-process a standard YAML file into OpenCV FileStorage compatible format.
 *
 * cv::FileStorage requires %YAML:1.0 header and does not support comments.
 * This function strips comment lines and adds the required header.
 *
 * @param file_path Path to the original YAML file
 * @return Path to a temporary file with OpenCV-compatible YAML content
 */
static std::string preprocess_yaml_for_opencv(const std::string& file_path) {
    std::ifstream ifs(file_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open config file: " + file_path);
    }

    std::stringstream oss;
    oss << "%YAML:1.0\n---\n";

    std::string line;
    bool has_header = false;
    while (std::getline(ifs, line)) {
        // Check for existing %YAML header
        if (!has_header && line.find("%YAML") == 0) {
            has_header = true;
            continue;
        }
        // Strip comment lines (but not inline comments in maps)
        size_t hash_pos = line.find_first_not_of(" \t");
        if (hash_pos != std::string::npos && line[hash_pos] == '#') {
            continue;
        }
        oss << line << "\n";
    }

    // Write to temp file
    std::string tmp_path = file_path + ".tmp_cvfs";
    std::ofstream ofs(tmp_path);
    ofs << oss.str();
    ofs.close();

    CV_LOG_INFO(&g_log_tag, "Pre-processed YAML: " << file_path << " -> " << tmp_path);
    return tmp_path;
}

config::config(const std::string& config_file_path)
    : config_file_path_(config_file_path), yaml_node_() {
    std::string tmp_path = preprocess_yaml_for_opencv(config_file_path);
    fs_.open(tmp_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
    CV_Assert(fs_.isOpened());
    // yaml_node_ is alias to fs_ root node; fs_ lifetime >= yaml_node_ lifetime
    const_cast<cv::FileNode&>(yaml_node_) = fs_.root();
    CV_LOG_DEBUG(&g_log_tag, "CONSTRUCT: config");
    CV_LOG_INFO(&g_log_tag, "config file loaded: " << config_file_path_);

    // Clean up temp file
    std::remove(tmp_path.c_str());
}

config::config(const cv::FileNode& yaml_node, const std::string& config_file_path)
    : config_file_path_(config_file_path), yaml_node_(yaml_node) {
    CV_LOG_DEBUG(&g_log_tag, "CONSTRUCT: config");
    CV_LOG_INFO(&g_log_tag, "config file loaded: " << config_file_path_);
}

std::ostream& operator<<(std::ostream& os, const config& cfg) {
    os << "[config file: " << cfg.config_file_path_ << "]";
    return os;
}

} // namespace cv::slam
