#ifndef SLAM_MATCH_FUSE_H
#define SLAM_MATCH_FUSE_H

#include "type.hpp"
#include "match/base.hpp"

#include <memory>

namespace cv::slam {

namespace data {
class keyframe;
class landmark;
class map_database;
} // namespace data

namespace match {

class fuse final {
public:
    explicit fuse(float lowe_ratio)
        : lowe_ratio_(lowe_ratio) {}

    virtual ~fuse() = default;


    template<typename T>
    unsigned int detect_duplication(const std::shared_ptr<data::keyframe>& keyfrm,
                                    const Mat33_t& rot_cw,
                                    const Vec3_t& trans_cw,
                                    const T& landmarks_to_check,
                                    const float margin,
                                    std::unordered_map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>>& duplicated_lms_in_keyfrm,
                                    std::unordered_map<unsigned int, std::shared_ptr<data::landmark>>& new_connections,
                                    bool do_reprojection_matching = false) const;

protected:
    const float lowe_ratio_;
};

} // namespace match
} // namespace cv::slam

#endif // SLAM_MATCH_FUSE_H
