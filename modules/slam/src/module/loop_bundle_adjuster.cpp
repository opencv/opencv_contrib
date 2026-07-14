#include "mapping_module.hpp"
#include "data/keyframe.hpp"
#include "data/landmark.hpp"
#include "data/marker.hpp"
#include "data/map_database.hpp"
#include "module/loop_bundle_adjuster.hpp"
#include "optimize/global_bundle_adjuster.hpp"

#include <thread>

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace module {

loop_bundle_adjuster::loop_bundle_adjuster(data::map_database* map_db,
                                           const unsigned int num_iter,
                                           const bool use_huber_kernel,
                                           const bool verbose)
    : map_db_(map_db),
      num_iter_(num_iter),
      use_huber_kernel_(use_huber_kernel),
      verbose_(verbose) {}

void loop_bundle_adjuster::set_mapping_module(mapping_module* mapper) {
    mapper_ = mapper;
}

void loop_bundle_adjuster::abort() {
    std::lock_guard<std::mutex> lock(mtx_thread_);
    abort_loop_BA_ = true;
}

bool loop_bundle_adjuster::is_running() const {
    std::lock_guard<std::mutex> lock(mtx_thread_);
    return loop_BA_is_running_;
}

void loop_bundle_adjuster::optimize(const std::shared_ptr<data::keyframe>& curr_keyfrm) {
    CV_LOG_INFO(&g_log_tag, "start loop bundle adjustment");

    {
        std::lock_guard<std::mutex> lock(mtx_thread_);
        loop_BA_is_running_ = true;
        abort_loop_BA_ = false;
    }

    std::unordered_set<unsigned int> optimized_keyfrm_ids;
    std::unordered_set<unsigned int> optimized_landmark_ids;
    std::unordered_set<unsigned int> optimized_marker_ids;
    eigen_alloc_unord_map<unsigned int, Vec3_t> lm_to_pos_w_after_global_BA;
    eigen_alloc_unord_map<unsigned int, Mat44_t> keyfrm_to_pose_cw_after_global_BA;
    eigen_alloc_unord_map<unsigned int, std::array<Vec3_t, 4>> marker_to_pos_w_after_global_BA;
    const auto global_BA = optimize::global_bundle_adjuster(num_iter_, use_huber_kernel_, verbose_);
    bool ok = global_BA.optimize(curr_keyfrm->graph_node_->get_keyframes_from_root(),
                                 optimized_keyfrm_ids, optimized_landmark_ids,
                                 optimized_marker_ids,
                                 lm_to_pos_w_after_global_BA,
                                 keyfrm_to_pose_cw_after_global_BA,
                                 marker_to_pos_w_after_global_BA,
                                 &abort_loop_BA_);

    {
        std::lock_guard<std::mutex> lock1(mtx_thread_);

        // if the loop BA was aborted, cannot update the map
        if (!ok) {
            CV_LOG_INFO(&g_log_tag, "abort loop bundle adjustment");
            loop_BA_is_running_ = false;
            abort_loop_BA_ = false;
            return;
        }

        CV_LOG_INFO(&g_log_tag, "finish loop bundle adjustment");
        CV_LOG_INFO(&g_log_tag, "updating the map with pose propagation");

        // stop mapping module
        auto future_pause = mapper_->async_pause();
        CV_LOG_DEBUG(&g_log_tag, "loop_bundle_adjuster::optimize: wait for mapper_->async_pause");
        future_pause.get();

        std::lock_guard<std::mutex> lock2(data::map_database::mtx_database_);

        CV_LOG_DEBUG(&g_log_tag, "update the camera pose along the spanning tree from the root");
        eigen_alloc_unord_map<unsigned int, Mat44_t> keyfrm_to_cam_pose_cw_before_BA;
        std::list<std::shared_ptr<data::keyframe>> keyfrms_to_check;
        keyfrms_to_check.push_back(curr_keyfrm->graph_node_->get_spanning_root());
        while (!keyfrms_to_check.empty()) {
            auto parent = keyfrms_to_check.front();
            const Mat44_t cam_pose_wp = parent->get_pose_wc();

            const auto children = parent->graph_node_->get_spanning_children();
            for (auto child : children) {
                if (!optimized_keyfrm_ids.count(child->id_)) {
                    // if `child` is NOT optimized by the loop BA
                    // propagate the pose correction from the spanning parent

                    // parent->child
                    const Mat44_t cam_pose_cp = child->get_pose_cw() * cam_pose_wp;
                    // world->child AFTER correction = parent->child * world->parent AFTER correction
                    keyfrm_to_pose_cw_after_global_BA[child->id_] = cam_pose_cp * keyfrm_to_pose_cw_after_global_BA.at(parent->id_);
                    // check as `child` has been corrected
                    optimized_keyfrm_ids.insert(child->id_);
                }

                // need updating
                keyfrms_to_check.push_back(child);
            }

            // temporally store the camera pose BEFORE correction (for correction of landmark positions)
            keyfrm_to_cam_pose_cw_before_BA[parent->id_] = parent->get_pose_cw();
            // update the camera pose
            parent->set_pose_cw(keyfrm_to_pose_cw_after_global_BA.at(parent->id_));
            // finish updating
            keyfrms_to_check.pop_front();
        }

        CV_LOG_DEBUG(&g_log_tag, "update the positions of the landmarks");
        auto keyfrms = curr_keyfrm->graph_node_->get_keyframes_from_root();
        std::unordered_set<unsigned int> already_found_landmark_ids;
        std::vector<std::shared_ptr<data::landmark>> lms;
        for (const auto& keyfrm : keyfrms) {
            for (const auto& lm : keyfrm->get_landmarks()) {
                if (!lm) {
                    continue;
                }
                if (lm->will_be_erased()) {
                    continue;
                }
                if (already_found_landmark_ids.count(lm->id_)) {
                    continue;
                }

                already_found_landmark_ids.insert(lm->id_);
                lms.push_back(lm);
            }
        }

        for (const auto& lm : lms) {
            if (lm->will_be_erased()) {
                continue;
            }

            if (optimized_landmark_ids.count(lm->id_)) {
                // if `lm` is optimized by the loop BA

                // update with the optimized position
                lm->set_pos_in_world(lm_to_pos_w_after_global_BA.at(lm->id_));
            }
            else {
                // if `lm` is NOT optimized by the loop BA

                // correct the position according to the move of the camera pose of the reference keyframe
                auto ref_keyfrm = lm->get_ref_keyframe();

                assert(optimized_keyfrm_ids.count(ref_keyfrm->id_));

                // convert the position to the camera-reference using the camera pose BEFORE the correction
                const Mat44_t pose_cw_before_BA = keyfrm_to_cam_pose_cw_before_BA.at(ref_keyfrm->id_);
                const Mat33_t rot_cw_before_BA = pose_cw_before_BA.block<3, 3>(0, 0);
                const Vec3_t trans_cw_before_BA = pose_cw_before_BA.block<3, 1>(0, 3);
                const Vec3_t pos_c = rot_cw_before_BA * lm->get_pos_in_world() + trans_cw_before_BA;

                // convert the position to the world-reference using the camera pose AFTER the correction
                const Mat44_t cam_pose_wc = ref_keyfrm->get_pose_wc();
                const Mat33_t rot_wc = cam_pose_wc.block<3, 3>(0, 0);
                const Vec3_t trans_wc = cam_pose_wc.block<3, 1>(0, 3);
                lm->set_pos_in_world(rot_wc * pos_c + trans_wc);
            }
            lm->update_mean_normal_and_obs_scale_variance();
        }

        CV_LOG_DEBUG(&g_log_tag, "update the positions of the markers");

        std::unordered_set<unsigned int> already_found_marker_ids;
        std::vector<std::shared_ptr<data::marker>> markers;
        for (const auto& keyfrm : keyfrms) {
            for (const auto& mkr : keyfrm->get_markers()) {
                if (!mkr) {
                    continue;
                }
                if (already_found_marker_ids.count(mkr->id_)) {
                    continue;
                }

                already_found_marker_ids.insert(mkr->id_);
                markers.push_back(mkr);
            }
        }

        for (const auto& mkr : markers) {
            if (!optimized_marker_ids.count(mkr->id_)) {
                continue;
            }

            // Update all corners
            const std::array<Vec3_t, 4>& new_corners = marker_to_pos_w_after_global_BA.at(mkr->id_);
            for (size_t corner_idx = 0; corner_idx < 4; corner_idx++) {
                mkr->corners_pos_w_[corner_idx] = new_corners[corner_idx];
            }
        }

        mapper_->resume();
        loop_BA_is_running_ = false;

        CV_LOG_INFO(&g_log_tag, "updated the map");
    }
}

} // namespace module
} // namespace cv::slam
