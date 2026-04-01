#include "config.hpp"
#include "util/string.hpp"
#include "util/yaml.hpp"

#include <iostream>
#include <memory>

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);

config::config(const std::string& config_file_path)
    : config(YAML::LoadFile(config_file_path), config_file_path) {}

config::config(const YAML::Node& yaml_node, const std::string& config_file_path)
    : config_file_path_(config_file_path), yaml_node_(yaml_node) {
    CV_LOG_DEBUG(&g_log_tag, "CONSTRUCT: config");

    CV_LOG_INFO(&g_log_tag, "config file loaded: " << config_file_path_);
}

config::~config() {
    CV_LOG_DEBUG(&g_log_tag, "DESTRUCT: config");
}

std::ostream& operator<<(std::ostream& os, const config& cfg) {
    os << cfg.yaml_node_;
    return os;
}

} // namespace cv::slam