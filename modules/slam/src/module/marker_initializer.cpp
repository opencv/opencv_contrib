#include "module/marker_initializer.hpp"
#include "data/keyframe.hpp"
#include "data/marker.hpp"
#include "marker_model/base.hpp"

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace module {

void marker_initializer::check_marker_initialization(data::marker& mkr, size_t needed_observations_for_initialization) {
    if (mkr.initialized_before_) {
        return;
    }

    auto id = mkr.id_;
    if (mkr.observations_.size() < needed_observations_for_initialization) {
        CV_LOG_DEBUG(&g_log_tag, "Not using marker " << id << " yet, not enough keyframes (" << mkr.observations_.size() << ", need " << needed_observations_for_initialization << ")");
        return;
    }

    std::vector<std::shared_ptr<data::keyframe>> valid_keyframes;
    for (const auto& id_keyfrm : mkr.observations_) {
        const auto& keyfrm = id_keyfrm.second;
        if (!keyfrm)
            continue;
        if (keyfrm->markers_2d_.find(id) == keyfrm->markers_2d_.end()) {
            CV_LOG_WARNING(&g_log_tag, "Couldn't find 2D marker " << id << " in keyframe that should have it");
            continue;
        }

        valid_keyframes.push_back(keyfrm);
    }

    if (valid_keyframes.size() < needed_observations_for_initialization) {
        CV_LOG_DEBUG(&g_log_tag, "Not using marker " << id << " yet, not enough valid keyframes (" << valid_keyframes.size() << ", need " << needed_observations_for_initialization << ")");
        return;
    }

    // Ok, initialize the positions
    eigen_alloc_vector<Vec3_t> corners_sum(4);
    for (auto& v : corners_sum) {
        v = {0.0, 0.0, 0.0};
    }

    for (const auto& keyfrm : valid_keyframes) {
        auto kf_pose = keyfrm->get_pose_wc();
        auto m2d_it = keyfrm->markers_2d_.find(id);
        assert(m2d_it != keyfrm->markers_2d_.end()); // We've checked this before
        auto& m2d = m2d_it->second;

        if (!m2d.marker_model_) {
            CV_LOG_ERROR(&g_log_tag, "Need marker model to be set to initialize 3D marker pos from 2D corners");
            return;
        }

        auto corners = m2d.compute_corners_pos_w(kf_pose, m2d.marker_model_->corners_pos_);
        for (size_t i = 0; i < 4; i++) {
            corners_sum[i] += corners[i];
        }
    }

    for (size_t i = 0; i < 4; i++) {
        corners_sum[i] /= valid_keyframes.size();
    }

    mkr.set_corner_pos(corners_sum);
    mkr.initialized_before_ = true;

    CV_LOG_DEBUG(&g_log_tag, "Initialized corner positions for marker " << id << " from " << valid_keyframes.size() << " keyframes");
}

} // namespace module
} // namespace cv::slam
