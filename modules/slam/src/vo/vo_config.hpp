#ifndef CV_SLAM_VO_CONFIG_HPP
#define CV_SLAM_VO_CONFIG_HPP

#include <string>

namespace cv::vo {

/**
 * @brief VO configuration struct
 *
 * Configuration is loaded from a YAML file. Camera parameters, feature
 * parameters, backend parameters, etc. are all read from the YAML file
 * specified by camera_config_file.
 */
struct VOConfig {
    // Camera configuration file path (YAML format, including Camera/Feature/Mapping/System settings)
    std::string camera_config_file;

    // Vocabulary file path (FBoW format, used for loop detection and global localization)
    std::string vocab_file;
};

} // namespace cv::vo

#endif // CV_SLAM_VO_CONFIG_HPP
