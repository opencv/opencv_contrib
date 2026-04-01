#ifndef SLAM_CONFIG_H
#define SLAM_CONFIG_H

#include "feature/orb_params.hpp"

#include <yaml-cpp/yaml.h>

namespace cv::slam {

namespace marker_model {
class base;
}

class config {
public:
    //! Constructor
    explicit config(const std::string& config_file_path);
    explicit config(const YAML::Node& yaml_node, const std::string& config_file_path = "");

    //! Destructor
    ~config();

    friend std::ostream& operator<<(std::ostream& os, const config& cfg);

    //! path to config YAML file
    const std::string config_file_path_;

    //! YAML node
    const YAML::Node yaml_node_;

    //! Marker model
    std::shared_ptr<marker_model::base> marker_model_ = nullptr;
};

} // namespace cv::slam

#endif // SLAM_CONFIG_H
