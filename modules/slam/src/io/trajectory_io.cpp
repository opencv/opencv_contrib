#include "data/keyframe.hpp"
#include "data/map_database.hpp"
#include "io/trajectory_io.hpp"
#include "util/converter.hpp"

#include <iostream>
#include <iomanip>

#include <opencv2/core/utils/logger.hpp>
#include <nlohmann/json.hpp>

#include <fstream>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace io {

trajectory_io::trajectory_io(data::map_database* map_db)
    : map_db_(map_db) {}

void trajectory_io::save_frame_trajectory(const std::string& path, const std::string& format) const {
    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

    // 1. acquire the frame stats

    assert(map_db_);
    const auto frm_stats = map_db_->get_frame_statistics();

    // 2. save the frames

    const auto num_valid_frms = frm_stats.get_num_valid_frames();
    const auto reference_keyframes = frm_stats.get_reference_keyframes();
    const auto rel_cam_poses_from_ref_keyfrms = frm_stats.get_relative_cam_poses();
    const auto timestamps = frm_stats.get_timestamps();
    const auto is_lost_frms = frm_stats.get_lost_frames();

    if (num_valid_frms == 0) {
        CV_LOG_WARNING(&g_log_tag, "there are no valid frames, cannot dump frame trajectory");
        return;
    }

    std::ofstream ofs(path, std::ios::out);
    if (!ofs.is_open()) {
        CV_LOG_FATAL(&g_log_tag, "cannot create a file at " << path);
        throw std::runtime_error("cannot create a file at " + path);
    }

    CV_LOG_INFO(&g_log_tag, "dump frame trajectory in \"" << format << "\" format from frame " << reference_keyframes.begin()->first << " to frame " << reference_keyframes.rbegin()->first << " (" << num_valid_frms << " frames)");

    const auto rk_itr_bgn = reference_keyframes.begin();
    const auto rc_itr_bgn = rel_cam_poses_from_ref_keyfrms.begin();
    const auto rk_itr_end = reference_keyframes.end();
    const auto rc_itr_end = rel_cam_poses_from_ref_keyfrms.end();
    auto rk_itr = rk_itr_bgn;
    auto rc_itr = rc_itr_bgn;

    int offset = rk_itr->first;
    unsigned int prev_frm_id = 0;
    for (unsigned int i = 0; i < num_valid_frms; ++i, ++rk_itr, ++rc_itr) {
        // check frame ID
        assert(rk_itr->first == rc_itr->first);
        const auto frm_id = rk_itr->first;

        // check if the frame was lost or not
        if (is_lost_frms.at(frm_id)) {
            CV_LOG_WARNING(&g_log_tag, "frame " << frm_id << " was lost");
            continue;
        }

        // check if the frame was skipped or not
        if (frm_id != i + offset) {
            CV_LOG_WARNING(&g_log_tag, "frame(s) from " << prev_frm_id + 1 << " to " << frm_id - 1 << " was/were skipped");
            offset = frm_id - i;
        }

        auto ref_keyfrm = rk_itr->second;
        const Mat44_t cam_pose_rw = ref_keyfrm->get_pose_cw();
        const Mat44_t rel_cam_pose_cr = rc_itr->second;

        const Mat44_t cam_pose_cw = rel_cam_pose_cr * cam_pose_rw;
        Mat44_t cam_pose_wc = util::converter::inverse_pose(cam_pose_cw);

        if (format == "KITTI") {
            ofs << std::setprecision(9)
                << cam_pose_wc(0, 0) << " " << cam_pose_wc(0, 1) << " " << cam_pose_wc(0, 2) << " " << cam_pose_wc(0, 3) << " "
                << cam_pose_wc(1, 0) << " " << cam_pose_wc(1, 1) << " " << cam_pose_wc(1, 2) << " " << cam_pose_wc(1, 3) << " "
                << cam_pose_wc(2, 0) << " " << cam_pose_wc(2, 1) << " " << cam_pose_wc(2, 2) << " " << cam_pose_wc(2, 3) << std::endl;
        }
        else if (format == "TUM") {
            const Mat33_t& rot_wc = cam_pose_wc.block<3, 3>(0, 0);
            const Vec3_t& trans_wc = cam_pose_wc.block<3, 1>(0, 3);
            const Quat_t quat_wc = Quat_t(rot_wc);
            ofs << std::setprecision(15)
                << timestamps.at(frm_id) << " "
                << std::setprecision(9)
                << trans_wc(0) << " " << trans_wc(1) << " " << trans_wc(2) << " "
                << quat_wc.x() << " " << quat_wc.y() << " " << quat_wc.z() << " " << quat_wc.w() << std::endl;
        }
        else {
            throw std::runtime_error("Not implemented: trajectory format \"" + format + "\"");
        }

        prev_frm_id = frm_id;
    }

    if (rk_itr != rk_itr_end || rc_itr != rc_itr_end) {
        CV_LOG_ERROR(&g_log_tag, "the sizes of frame statistics are not matched");
    }

    ofs.close();
}

void trajectory_io::save_keyframe_trajectory(const std::string& path, const std::string& format) const {
    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

    // 1. acquire keyframes and sort them

    assert(map_db_);
    auto roots = map_db_->get_spanning_roots();
    if (roots.empty()) {
        CV_LOG_WARNING(&g_log_tag, "empty map");
        return;
    }
    auto keyfrms = roots.back()->graph_node_->get_keyframes_from_root();
    std::sort(keyfrms.begin(), keyfrms.end(), [&](const std::shared_ptr<data::keyframe>& keyfrm_1, const std::shared_ptr<data::keyframe>& keyfrm_2) {
        return *keyfrm_1 < *keyfrm_2;
    });

    // 2. save the keyframes

    if (keyfrms.empty()) {
        CV_LOG_WARNING(&g_log_tag, "there are no valid keyframes, cannot dump keyframe trajectory");
        return;
    }

    std::ofstream ofs(path, std::ios::out);
    if (!ofs.is_open()) {
        CV_LOG_FATAL(&g_log_tag, "cannot create a file at " << path);
        throw std::runtime_error("cannot create a file at " + path);
    }

    CV_LOG_INFO(&g_log_tag, "dump keyframe trajectory in \"" << format << "\" format from keyframe " << (*keyfrms.begin())->id_ << " to keyframe " << (*keyfrms.rbegin())->id_ << " (" << keyfrms.size() << " keyframes)");

    for (const auto& keyfrm : keyfrms) {
        const Mat44_t cam_pose_wc = keyfrm->get_pose_wc();
        const auto timestamp = keyfrm->timestamp_;

        if (format == "KITTI") {
            ofs << std::setprecision(9)
                << cam_pose_wc(0, 0) << " " << cam_pose_wc(0, 1) << " " << cam_pose_wc(0, 2) << " " << cam_pose_wc(0, 3) << " "
                << cam_pose_wc(1, 0) << " " << cam_pose_wc(1, 1) << " " << cam_pose_wc(1, 2) << " " << cam_pose_wc(1, 3) << " "
                << cam_pose_wc(2, 0) << " " << cam_pose_wc(2, 1) << " " << cam_pose_wc(2, 2) << " " << cam_pose_wc(2, 3) << std::endl;
        }
        else if (format == "TUM") {
            const Mat33_t& rot_wc = cam_pose_wc.block<3, 3>(0, 0);
            const Vec3_t& trans_wc = cam_pose_wc.block<3, 1>(0, 3);
            const Quat_t quat_wc = Quat_t(rot_wc);
            ofs << std::setprecision(15)
                << timestamp << " "
                << std::setprecision(9)
                << trans_wc(0) << " " << trans_wc(1) << " " << trans_wc(2) << " "
                << quat_wc.x() << " " << quat_wc.y() << " " << quat_wc.z() << " " << quat_wc.w() << std::endl;
        }
        else {
            throw std::runtime_error("Not implemented: trajectory format \"" + format + "\"");
        }
    }

    ofs.close();
}

} // namespace io
} // namespace cv::slam
