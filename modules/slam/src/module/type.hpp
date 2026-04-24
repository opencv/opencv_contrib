#ifndef SLAM_MODULE_TYPE_H
#define SLAM_MODULE_TYPE_H

#include "../type.hpp"
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include <memory>

namespace cv::slam {

namespace data {
class keyframe;
}

namespace module {


typedef std::map<std::shared_ptr<data::keyframe>,
                 g2o::Sim3,
                 std::less<std::shared_ptr<data::keyframe>>,
                 Eigen::aligned_allocator<std::pair<std::shared_ptr<data::keyframe> const, g2o::Sim3>>>
    keyframe_Sim3_pairs_t;


struct keyframe_set {
    keyframe_set(const std::set<std::shared_ptr<data::keyframe>>& keyfrm_set, const std::shared_ptr<data::keyframe>& lead_keyfrm, const unsigned int continuity)
        : keyfrm_set_(keyfrm_set), lead_keyfrm_(lead_keyfrm), continuity_(continuity) {}
    std::set<std::shared_ptr<data::keyframe>> keyfrm_set_;
    std::shared_ptr<data::keyframe> lead_keyfrm_;
    unsigned int continuity_ = 0;

    bool intersection_is_empty(const std::set<std::shared_ptr<data::keyframe>>& other_set) const {
        for (const auto& this_keyfrm : keyfrm_set_) {
            if (static_cast<bool>(other_set.count(this_keyfrm))) {
                return false;
            }
        }
        return true;
    }

    bool intersection_is_empty(const keyframe_set& other_set) const {
        return intersection_is_empty(other_set.keyfrm_set_);
    }
};

using keyframe_sets = eigen_alloc_vector<keyframe_set>;

} // namespace module
} // namespace cv::slam

#endif // SLAM_MODULE_TYPE_H
