#ifndef SLAM_MODULE_MARKER_INITIALIZER_H
#define SLAM_MODULE_MARKER_INITIALIZER_H

#include "type.hpp"

namespace cv::slam {

namespace data {
class marker;
} // namespace data

namespace module {

class marker_initializer {
public:
    static void check_marker_initialization(data::marker& mkr, size_t needed_observations_for_initialization);

private:
    const size_t required_keyframes_for_marker_initialization_ = 3;
};

} // namespace module
} // namespace cv::slam

#endif // SLAM_MODULE_MARKER_INITIALIZER_H
