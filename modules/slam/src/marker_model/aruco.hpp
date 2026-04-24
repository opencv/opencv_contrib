#ifndef SLAM_MARKER_MODEL_ACURO_H
#define SLAM_MARKER_MODEL_ACURO_H

#include "type.hpp"
#include "marker_model/base.hpp"

#include <string>
#include <limits>

#include <yaml-cpp/yaml.h>
#include <nlohmann/json_fwd.hpp>

namespace cv::slam {
namespace marker_model {

class aruco : public marker_model::base {
public:
    //! Constructor
    aruco(double width, int marker_size, int max_markers);

    //! Destructor
    virtual ~aruco();

    //! marker definition
    int marker_size_;
    int max_markers_;

    //! Encode marker_model information as JSON
    virtual nlohmann::json to_json() const;
};

std::ostream& operator<<(std::ostream& os, const aruco& params);

} // namespace marker_model
} // namespace cv::slam

#endif // SLAM_MARKER_MODEL_ACURO_H
