#ifndef SLAM_MARKER_MODEL_ACURONANO_H
#define SLAM_MARKER_MODEL_ACURONANO_H

#include "type.hpp"
#include "marker_model/base.hpp"

#include <string>
#include <limits>

#include <opencv2/core/persistence.hpp>
#include <nlohmann/json_fwd.hpp>

namespace cv::slam {
namespace marker_model {

class aruconano : public marker_model::base {
public:
    //! Constructor
    aruconano(double width, int dict);

    //! Destructor
    virtual ~aruconano();

    int dict_;

    //! Encode marker_model information as JSON
    virtual nlohmann::json to_json() const;
};

std::ostream& operator<<(std::ostream& os, const aruconano& params);

} // namespace marker_model
} // namespace cv::slam

#endif // SLAM_MARKER_MODEL_ACURONANO_H
