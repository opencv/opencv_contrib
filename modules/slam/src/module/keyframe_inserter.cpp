#include "mapping_module.hpp"
#include "util/yaml.hpp"
#include "data/landmark.hpp"
#include "data/marker.hpp"
#include "data/map_database.hpp"
#include "marker_model/base.hpp"
#include "module/marker_initializer.hpp"
#include "module/keyframe_inserter.hpp"

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace module {

keyframe_inserter::keyframe_inserter(const double max_interval,
                                     const double min_interval,
                                     const double max_distance,
                                     const double min_distance,
                                     const double lms_ratio_thr_almost_all_lms_are_tracked,
                                     const double lms_ratio_thr_view_changed,
                                     const unsigned int enough_lms_thr,
                                     const bool wait_for_local_bundle_adjustment,
                                     const size_t required_keyframes_for_marker_initialization)
    : max_interval_(max_interval),
      min_interval_(min_interval),
      max_distance_(max_distance),
      min_distance_(min_distance),
      lms_ratio_thr_almost_all_lms_are_tracked_(lms_ratio_thr_almost_all_lms_are_tracked),
      lms_ratio_thr_view_changed_(lms_ratio_thr_view_changed),
      enough_lms_thr_(enough_lms_thr),
      wait_for_local_bundle_adjustment_(wait_for_local_bundle_adjustment),
      required_keyframes_for_marker_initialization_(required_keyframes_for_marker_initialization) {}

keyframe_inserter::keyframe_inserter(const cv::FileNode& yaml_node)
    : keyframe_inserter(util::yaml_get_val<double>(yaml_node, "max_interval", 1.0),
                        util::yaml_get_val<double>(yaml_node, "min_interval", 0.1),
                        util::yaml_get_val<double>(yaml_node, "max_distance", -1.0),
                        util::yaml_get_val<double>(yaml_node, "min_distance", -1.0),
                        util::yaml_get_val<double>(yaml_node, "lms_ratio_thr_almost_all_lms_are_tracked", 0.9),
                        util::yaml_get_val<double>(yaml_node, "lms_ratio_thr_view_changed", 0.5),
                        util::yaml_get_val<unsigned int>(yaml_node, "enough_lms_thr", 100),
                        util::yaml_get_val<bool>(yaml_node, "wait_for_local_bundle_adjustment", false),
                        util::yaml_get_val<unsigned int>(yaml_node, "required_keyframes_for_marker_initialization", 3)) {}

void keyframe_inserter::set_mapping_module(mapping_module* mapper) {
    mapper_ = mapper;
}

void keyframe_inserter::reset() {
}

bool keyframe_inserter::new_keyframe_is_needed(data::map_database* map_db,
                                               const data::frame& curr_frm,
                                               const unsigned int num_tracked_lms,
                                               const unsigned int num_reliable_lms,
                                               const data::keyframe& ref_keyfrm,
                                               const unsigned int min_num_obs_thr) const {
    assert(mapper_);
    // Any keyframes are not able to be added when the mapping module stops
    if (mapper_->is_paused() || mapper_->pause_is_requested()) {
        return false;
    }

    auto last_inserted_keyfrm = map_db->get_last_inserted_keyframe();

    // Count the number of the 3D points that are observed from more than two keyframes
    const auto num_reliable_lms_ref = ref_keyfrm.get_num_tracked_landmarks(min_num_obs_thr);

    // When the mapping module skips localBA, it does not insert keyframes
    const auto mapper_is_skipping_localBA = mapper_->is_skipping_localBA();

    constexpr unsigned int num_enough_keyfrms_thr = 5;
    const bool enough_keyfrms = map_db->get_num_keyframes() > num_enough_keyfrms_thr;

    // New keyframe is needed if the time elapsed since the last keyframe insertion reaches the threshold
    bool max_interval_elapsed = false;
    if (max_interval_ > 0.0) {
        max_interval_elapsed = last_inserted_keyfrm && last_inserted_keyfrm->timestamp_ + max_interval_ <= curr_frm.timestamp_;
    }
    bool min_interval_elapsed = true;
    if (min_interval_ > 0.0) {
        min_interval_elapsed = !last_inserted_keyfrm || last_inserted_keyfrm->timestamp_ + min_interval_ <= curr_frm.timestamp_;
    }
    float distance_traveled = -1.0;
    if (last_inserted_keyfrm) {
        distance_traveled = (last_inserted_keyfrm->get_trans_wc() - curr_frm.get_trans_wc()).norm();
    }
    bool max_distance_traveled = false;
    if (max_distance_ > 0.0) {
        max_distance_traveled = last_inserted_keyfrm && distance_traveled > max_distance_;
    }
    bool min_distance_traveled = true;
    if (min_distance_ > 0.0) {
        min_distance_traveled = !last_inserted_keyfrm || distance_traveled > min_distance_;
    }
    // New keyframe is needed if the field-of-view of the current frame is changed a lot
    bool view_changed = false;
    if (lms_ratio_thr_view_changed_ > 0.0) {
        view_changed = num_reliable_lms < num_reliable_lms_ref * lms_ratio_thr_view_changed_;
    }
    // const bool view_changed = num_tracked_lms < num_tracked_lms_on_ref_keyfrm * lms_ratio_thr_view_changed_;
    const bool not_enough_lms = num_reliable_lms < enough_lms_thr_;

    // (Mandatory for keyframe insertion)
    // New keyframe is needed if the number of 3D points exceeds the threshold,
    // and concurrently the ratio of the reliable 3D points larger than the threshold ratio
    constexpr unsigned int num_tracked_lms_thr_unstable = 15;
    bool tracking_is_unstable = num_tracked_lms < num_tracked_lms_thr_unstable;
    bool almost_all_lms_are_tracked = false;
    if (lms_ratio_thr_almost_all_lms_are_tracked_ > 0.0) {
        almost_all_lms_are_tracked = num_reliable_lms > num_reliable_lms_ref * lms_ratio_thr_almost_all_lms_are_tracked_;
    }
    CV_LOG_DEBUG(&g_log_tag, "keyframe_inserter: num_reliable_lms_ref=" << num_reliable_lms_ref);
    CV_LOG_DEBUG(&g_log_tag, "keyframe_inserter: num_reliable_lms=" << num_reliable_lms);
    CV_LOG_DEBUG(&g_log_tag, "keyframe_inserter: max_interval_elapsed=" << max_interval_elapsed);
    CV_LOG_DEBUG(&g_log_tag, "keyframe_inserter: max_distance_traveled=" << max_distance_traveled);
    CV_LOG_DEBUG(&g_log_tag, "keyframe_inserter: view_changed=" << view_changed);
    CV_LOG_DEBUG(&g_log_tag, "keyframe_inserter: not_enough_lms=" << not_enough_lms);
    CV_LOG_DEBUG(&g_log_tag, "keyframe_inserter: enough_keyfrms=" << enough_keyfrms);
    CV_LOG_DEBUG(&g_log_tag, "keyframe_inserter: min_interval_elapsed=" << min_interval_elapsed);
    CV_LOG_DEBUG(&g_log_tag, "keyframe_inserter: tracking_is_unstable=" << tracking_is_unstable);
    CV_LOG_DEBUG(&g_log_tag, "keyframe_inserter: almost_all_lms_are_tracked=" << almost_all_lms_are_tracked);
    CV_LOG_DEBUG(&g_log_tag, "keyframe_inserter: mapper_is_skipping_localBA=" << mapper_is_skipping_localBA);
    return (max_interval_elapsed || max_distance_traveled || view_changed || not_enough_lms)
           && (!enough_keyfrms || (min_interval_elapsed && min_distance_traveled))
           && !tracking_is_unstable
           && !almost_all_lms_are_tracked
           && !mapper_is_skipping_localBA;
}

std::shared_ptr<data::keyframe> keyframe_inserter::create_new_keyframe(
    data::map_database* map_db,
    data::frame& curr_frm) {
    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

    auto keyfrm = data::keyframe::make_keyframe(map_db->next_keyframe_id_++, curr_frm);
    keyfrm->update_landmarks();

    for (const auto& id_mkr2d : keyfrm->markers_2d_) {
        auto marker = map_db->get_marker(id_mkr2d.first);
        if (!marker) {
            // Create new marker
            auto mkr2d = id_mkr2d.second;
            eigen_alloc_vector<Vec3_t> corners_pos_w = mkr2d.compute_corners_pos_w(keyfrm->get_pose_wc(), mkr2d.marker_model_->corners_pos_);
            marker = std::make_shared<data::marker>(corners_pos_w, id_mkr2d.first, mkr2d.marker_model_);
            // add the marker to the map DB
            map_db->add_marker(marker);
        }
        // Set the association to the new marker
        keyfrm->add_marker(marker);
        marker->observations_.emplace(keyfrm->id_, keyfrm);

        marker_initializer::check_marker_initialization(*marker, required_keyframes_for_marker_initialization_);
    }

    // Queue up the keyframe to the mapping module
    if (!keyfrm->depth_is_available()) {
        return keyfrm;
    }

    // Save the valid depth and index pairs
    std::vector<std::pair<float, unsigned int>> depth_idx_pairs;
    depth_idx_pairs.reserve(curr_frm.frm_obs_.undist_keypts_.size());
    for (unsigned int idx = 0; idx < curr_frm.frm_obs_.undist_keypts_.size(); ++idx) {
        assert(!curr_frm.frm_obs_.depths_.empty());
        const auto depth = curr_frm.frm_obs_.depths_.at(idx);
        // Add if the depth is valid
        if (0 < depth) {
            depth_idx_pairs.emplace_back(std::make_pair(depth, idx));
        }
    }

    // Queue up the keyframe to the mapping module if any valid depth values don't exist
    if (depth_idx_pairs.empty()) {
        return keyfrm;
    }

    // Sort in order of distance to the camera
    std::sort(depth_idx_pairs.begin(), depth_idx_pairs.end());

    // Create 3D points by using a depth parameter
    constexpr unsigned int min_num_to_create = 100;
    for (unsigned int count = 0; count < depth_idx_pairs.size(); ++count) {
        const auto depth = depth_idx_pairs.at(count).first;
        const auto idx = depth_idx_pairs.at(count).second;

        // Stop adding a keyframe if the number of 3D points exceeds the minimal threshold,
        // and concurrently the depth value exceeds the threshold
        if (min_num_to_create < count && keyfrm->camera_->depth_thr_ < depth) {
            break;
        }

        // Stereo-triangulation cannot be performed if the 3D point has been already associated to the keypoint index
        {
            const auto& lm = curr_frm.get_landmark(idx);
            if (lm) {
                assert(lm->has_observation());
                continue;
            }
        }

        // Stereo-triangulation can be performed if the 3D point is not yet associated to the keypoint index
        const Vec3_t pos_w = curr_frm.triangulate_stereo(idx);
        auto lm = std::make_shared<data::landmark>(map_db->next_landmark_id_++, pos_w, keyfrm);

        lm->connect_to_keyframe(keyfrm, idx);
        curr_frm.add_landmark(lm, idx);

        lm->compute_descriptor();
        lm->update_mean_normal_and_obs_scale_variance();

        map_db->add_landmark(lm);
    }

    // Queue up the keyframe to the mapping module
    return keyfrm;
}

void keyframe_inserter::insert_new_keyframe(data::map_database* map_db,
                                            data::frame& curr_frm) {
    CV_LOG_DEBUG(&g_log_tag, "keyframe_inserter: insert_new_keyframe (curr_frm=" << curr_frm.id_ << ")");
    // insert the new keyframe
    const auto ref_keyfrm = create_new_keyframe(map_db, curr_frm);
    auto future_add_keyframe = mapper_->async_add_keyframe(ref_keyfrm);
    if (wait_for_local_bundle_adjustment_) {
        future_add_keyframe.get();
    }
    // set the reference keyframe with the new keyframe
    if (ref_keyfrm) {
        curr_frm.ref_keyfrm_ = ref_keyfrm;
    }
}

} // namespace module
} // namespace cv::slam
