#ifndef SLAM_MATCH_AREA_H
#define SLAM_MATCH_AREA_H

#include "match/base.hpp"

namespace cv::slam {

namespace data {
class frame;
} // namespace data

namespace match {

class area final : public base {
public:
    area(const float lowe_ratio, const bool check_orientation)
        : base(lowe_ratio, check_orientation) {}

    ~area() final = default;

    unsigned int match_in_consistent_area(data::frame& frm_1, data::frame& frm_2, std::vector<cv::Point2f>& prev_matched_pts,
                                          std::vector<int>& matched_indices_2_in_frm_1, int margin = 10);
};

} // namespace match
} // namespace cv::slam

#endif // SLAM_MATCH_AREA_H
