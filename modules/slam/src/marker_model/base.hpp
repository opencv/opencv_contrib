#ifndef SLAM_MARKER_MODEL_BASE_H
#define SLAM_MARKER_MODEL_BASE_H

#include "type.hpp"

#include <string>
#include <limits>

#include <opencv2/core/persistence.hpp>
#include <nlohmann/json_fwd.hpp>

namespace cv::slam {
namespace marker_model {

class base {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    //! Constructor
    explicit base(double width);

    //! Destructor
    virtual ~base();

    //! marker geometry
    const double width_;
    eigen_alloc_vector<Vec3_t> corners_pos_;

    //! Encode marker_model information as JSON
    virtual nlohmann::json to_json() const;
};

std::ostream& operator<<(std::ostream& os, const base& params);

} // namespace marker_model
} // namespace cv::slam

#endif // SLAM_MARKER_MODEL_BASE_H
