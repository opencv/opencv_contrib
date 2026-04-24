#include "type.hpp"
#include "mapping_module.hpp"
#include "tracking_module.hpp"
#include "global_optimization_module.hpp"
#include "data/keyframe.hpp"
#include "data/landmark.hpp"
#include "data/map_database.hpp"
#include "match/fuse.hpp"
#include "match/robust.hpp"
#include "module/two_view_triangulator.hpp"
#include "optimize/local_bundle_adjuster_factory.hpp"
#include "solve/essential_solver.hpp"

#include <thread>

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam_module", cv::utils::logging::LOG_LEVEL_INFO);

mapping_module::mapping_module(const YAML::Node& yaml_node, data::map_database* map_db, data::bow_database* bow_db, data::bow_vocabulary* bow_vocab)
    : local_map_cleaner_(new module::local_map_cleaner(yaml_node, map_db, bow_db)),
      map_db_(map_db), bow_db_(bow_db), bow_vocab_(bow_vocab),
      local_bundle_adjuster_(optimize::local_bundle_adjuster_factory::create(yaml_node)),
      enable_interruption_of_landmark_generation_(yaml_node["enable_interruption_of_landmark_generation"].as<bool>(true)),
      enable_interruption_before_local_BA_(yaml_node["enable_interruption_before_local_BA"].as<bool>(true)),
      num_covisibilities_for_landmark_generation_(yaml_node["num_covisibilities_for_landmark_generation"].as<unsigned int>(10)),
      num_covisibilities_for_landmark_fusion_(yaml_node["num_covisibilities_for_landmark_fusion"].as<unsigned int>(10)),
      erase_temporal_keyframes_(yaml_node["erase_temporal_keyframes"].as<bool>(false)),
      num_temporal_keyframes_(yaml_node["num_temporal_keyframes"].as<unsigned int>(15)),
      residual_rad_thr_(yaml_node["residual_deg_thr"].as<float>(0.2) * M_PI / 180.0) {
    CV_LOG_DEBUG(&g_log_tag, "CONSTRUCT: mapping_module");

    CV_LOG_DEBUG(&g_log_tag, "load mapping parameters");

    CV_LOG_DEBUG(&g_log_tag, "load monocular mappping parameters");
    if (yaml_node["baseline_dist_thr"]) {
        if (yaml_node["baseline_dist_thr_ratio"]) {
            throw std::runtime_error("Do not set both baseline_dist_thr_ratio and baseline_dist_thr.");
        }
        baseline_dist_thr_ = yaml_node["baseline_dist_thr"].as<double>(1.0);
        use_baseline_dist_thr_ratio_ = false;
        CV_LOG_DEBUG(&g_log_tag, "Use baseline_dist_thr: " << baseline_dist_thr_);
    }
    else {
        baseline_dist_thr_ratio_ = yaml_node["baseline_dist_thr_ratio"].as<double>(0.02);
        use_baseline_dist_thr_ratio_ = true;
        CV_LOG_DEBUG(&g_log_tag, "Use baseline_dist_thr_ratio: " << baseline_dist_thr_ratio_);
    }
}

mapping_module::~mapping_module() {
    CV_LOG_DEBUG(&g_log_tag, "DESTRUCT: mapping_module");
}

void mapping_module::set_tracking_module(tracking_module* tracker) {
    tracker_ = tracker;
}

void mapping_module::set_global_optimization_module(global_optimization_module* global_optimizer) {
    global_optimizer_ = global_optimizer;
}

void mapping_module::run() {
    CV_LOG_INFO(&g_log_tag, "start mapping module");

    is_terminated_ = false;

    while (true) {
        // waiting time for the other threads
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        // check if termination is requested
        if (terminate_is_requested()) {
            // terminate and break
            CV_LOG_DEBUG(&g_log_tag, "mapping_module: terminate");
            terminate();
            break;
        }

        // check if reset is requested
        if (reset_is_requested()) {
            // reset and continue
            reset();
            continue;
        }

        // check if pause is requested and not prevented
        if (pause_is_requested()) {
            CV_LOG_DEBUG(&g_log_tag, "mapping_module: tracker_->is_stopped_keyframe_insertion");
            auto future_stop_keyframe_insertion = tracker_->async_stop_keyframe_insertion();
            future_stop_keyframe_insertion.get();
            if (!keyframe_is_queued()) {
                pause();
                CV_LOG_DEBUG(&g_log_tag, "mapping_module: waiting");
                // check if termination or reset is requested during pause
                while (is_paused() && !terminate_is_requested() && !reset_is_requested()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(3));
                }
                auto future_start_keyframe_insertion = tracker_->async_start_keyframe_insertion();
                future_start_keyframe_insertion.get();
                CV_LOG_DEBUG(&g_log_tag, "mapping_module: resume");
            }
        }

        // if the queue is empty, the following process is not needed
        if (!keyframe_is_queued()) {
            continue;
        }

        // create and extend the map with the new keyframe
        mapping_with_new_keyframe();
        // send the new keyframe to the global optimization module
        if (global_optimizer_ && !cur_keyfrm_->graph_node_->is_spanning_root()) {
            global_optimizer_->queue_keyframe(cur_keyfrm_);
        }
    }

    CV_LOG_INFO(&g_log_tag, "terminate mapping module");
}

std::shared_future<void> mapping_module::async_add_keyframe(const std::shared_ptr<data::keyframe>& keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    keyfrms_queue_.push_back(keyfrm);
    abort_local_BA_ = true;
    promise_add_keyfrm_queue_.emplace_back();
    return promise_add_keyfrm_queue_.back().get_future().share();
}

unsigned int mapping_module::get_num_queued_keyframes() const {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    return keyfrms_queue_.size();
}

bool mapping_module::keyframe_is_queued() const {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    return !keyfrms_queue_.empty();
}

bool mapping_module::is_skipping_localBA() const {
    auto queued_keyframes = get_num_queued_keyframes();
    return queued_keyframes >= queue_threshold_;
}

void mapping_module::abort_local_BA() {
    abort_local_BA_ = true;
}

void mapping_module::set_enable_local_BA(bool enable) {
    enable_local_BA_ = enable;
    CV_LOG_INFO(&g_log_tag, "Local BA " << (enable ? "enabled" : "disabled"));
}

bool mapping_module::is_local_BA_enabled() const {
    return enable_local_BA_;
}

void mapping_module::set_ba_window_size(int size) {
    ba_window_size_ = size;
}

int mapping_module::get_ba_window_size() const {
    return ba_window_size_;
}

void mapping_module::mapping_with_new_keyframe() {
    // dequeue
    {
        std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
        // dequeue -> cur_keyfrm_
        cur_keyfrm_ = keyfrms_queue_.front();
        keyfrms_queue_.pop_front();
    }

    CV_LOG_DEBUG(&g_log_tag, "mapping_module: current keyframe is " << cur_keyfrm_->id_);

    // store the new keyframe to the database
    store_new_keyframe();

    // remove invalid landmarks
    local_map_cleaner_->remove_invalid_landmarks(cur_keyfrm_->id_);

    // triangulate new landmarks between the current frame and each of the covisibilities
    std::atomic<bool> abort_create_new_landmarks{false};
    if (!enable_interruption_of_landmark_generation_) {
        create_new_landmarks(abort_create_new_landmarks);
    }
    else {
        auto future_create_new_landmark = std::async(std::launch::async,
                                                     [this, &abort_create_new_landmarks]() {
                                                         create_new_landmarks(abort_create_new_landmarks);
                                                     });
        while (future_create_new_landmark.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout) {
            if (keyframe_is_queued()) {
                abort_create_new_landmarks = true;
            }
        }
    }

    CV_LOG_DEBUG(&g_log_tag, "mapping_module: update_new_keyframe (current keyframe is " << cur_keyfrm_->id_ << ")");

    // detect and resolve the duplication of the landmarks observed in the current frame
    update_new_keyframe();

    if (enable_interruption_before_local_BA_ && (keyframe_is_queued() || pause_is_requested())) {
        {
            std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
            promise_add_keyfrm_queue_.front().set_value();
            promise_add_keyfrm_queue_.pop_front();
        }
        return;
    }

    CV_LOG_DEBUG(&g_log_tag, "mapping_module: local bundle adjustment (current keyframe is " << cur_keyfrm_->id_ << ")");

    // local bundle adjustment
    abort_local_BA_ = false;
    // Check if local BA is disabled (for ablation experiments)
    if (!enable_local_BA_) {
        CV_LOG_DEBUG(&g_log_tag, "Skipped local BA (disabled for ablation)");
    }
    else if (2 < map_db_->get_num_keyframes()) {
        if (is_skipping_localBA()) {
            CV_LOG_DEBUG(&g_log_tag, "Skipped localBA due to insufficient performance");
        }
        else {
            local_bundle_adjuster_->optimize(map_db_, cur_keyfrm_, &abort_local_BA_);
        }
    }

    if (erase_temporal_keyframes_) {
        for (const auto& keyfrm : map_db_->get_all_keyframes()) {
            if (keyfrm->id_ <= map_db_->get_fixed_keyframe_id_threshold()) {
                continue;
            }

            // erase temporal keyframes after a period of time
            if (keyfrm->id_ > map_db_->get_fixed_keyframe_id_threshold()
                && cur_keyfrm_->id_ > keyfrm->id_ + num_temporal_keyframes_) {
                const auto cur_landmarks = keyfrm->get_landmarks();
                keyfrm->prepare_for_erasing(map_db_, bow_db_);
                for (const auto& lm : cur_landmarks) {
                    if (!lm) {
                        continue;
                    }
                    if (lm->will_be_erased()) {
                        continue;
                    }
                    if (!lm->has_representative_descriptor()) {
                        lm->compute_descriptor();
                    }
                    if (!lm->has_valid_prediction_parameters()) {
                        lm->update_mean_normal_and_obs_scale_variance();
                    }
                }
            }
        }
    }

    local_map_cleaner_->remove_redundant_keyframes(cur_keyfrm_);

    {
        std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
        promise_add_keyfrm_queue_.front().set_value();
        promise_add_keyfrm_queue_.pop_front();
    }
}

void mapping_module::store_new_keyframe() {
    // compute BoW feature vector
    if (bow_vocab_ && !cur_keyfrm_->bow_is_available()) {
        cur_keyfrm_->compute_bow(bow_vocab_);
    }

    // Set landmarks into local_map_cleaner to exclude invalid landmarks
    const auto cur_lms = cur_keyfrm_->get_landmarks();
    for (unsigned int idx = 0; idx < cur_lms.size(); ++idx) {
        auto lm = cur_lms.at(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        local_map_cleaner_->add_fresh_landmark(lm);
    }

    // update graph
    cur_keyfrm_->graph_node_->update_connections(map_db_->get_min_num_shared_lms());

    // store the new keyframe to the map database
    map_db_->add_keyframe(cur_keyfrm_);
}

void mapping_module::create_new_landmarks(std::atomic<bool>& abort_create_new_landmarks) {
    // get the covisibilities of `cur_keyfrm_`
    // in order to triangulate landmarks between `cur_keyfrm_` and each of the covisibilities
    const auto cur_covisibilities = cur_keyfrm_->graph_node_->get_top_n_covisibilities(num_covisibilities_for_landmark_generation_);

    match::bow_tree bow_tree_matcher(0.95, false);
    match::robust robust_matcher(0.95, false);

    // camera center of the current keyframe
    const Vec3_t cur_cam_center = cur_keyfrm_->get_trans_wc();

    for (unsigned int i = 0; i < cur_covisibilities.size(); ++i) {
        // if any keyframe is queued, abort the triangulation
        if (1 < i && abort_create_new_landmarks) {
            return;
        }

        // get the neighbor keyframe
        auto ngh_keyfrm = cur_covisibilities.at(i);

        // camera center of the neighbor keyframe
        const Vec3_t ngh_cam_center = ngh_keyfrm->get_trans_wc();

        // compute the baseline between the current and neighbor keyframes
        const Vec3_t baseline_vec = ngh_cam_center - cur_cam_center;
        const auto baseline_dist = baseline_vec.norm();

        // if the scene scale is much smaller than the baseline, abort the triangulation
        if (use_baseline_dist_thr_ratio_) {
            float median_scale_in_ngh;
            if (ngh_keyfrm->camera_->model_type_ == camera::model_type_t::Equirectangular) {
                median_scale_in_ngh = ngh_keyfrm->compute_median_distance();
            }
            else {
                median_scale_in_ngh = ngh_keyfrm->compute_median_depth(true);
            }
            if (baseline_dist < baseline_dist_thr_ratio_ * median_scale_in_ngh) {
                continue;
            }
        }
        else {
            if (baseline_dist < baseline_dist_thr_) {
                continue;
            }
        }

        // estimate matches between the current and neighbor keyframes,
        // then reject outliers using Essential matrix computed from the two camera poses

        // (cur bearing) * E_ngh_to_cur * (ngh bearing) = 0
        // const Mat33_t E_ngh_to_cur = solve::essential_solver::create_E_21(ngh_keyfrm, cur_keyfrm_);
        const Mat33_t E_ngh_to_cur = solve::essential_solver::create_E_21(ngh_keyfrm->get_rot_cw(), ngh_keyfrm->get_trans_cw(),
                                                                          cur_keyfrm_->get_rot_cw(), cur_keyfrm_->get_trans_cw());

        // vector of matches (idx in the current, idx in the neighbor)
        std::vector<std::pair<unsigned int, unsigned int>> matches;
        if (bow_db_ && bow_vocab_) {
            bow_tree_matcher.match_for_triangulation(cur_keyfrm_, ngh_keyfrm, E_ngh_to_cur, matches, residual_rad_thr_);
        }
        else {
            robust_matcher.match_for_triangulation(cur_keyfrm_, ngh_keyfrm, E_ngh_to_cur, matches, residual_rad_thr_);
        }

        // triangulation
        triangulate_with_two_keyframes(cur_keyfrm_, ngh_keyfrm, matches);
    }
}

void mapping_module::triangulate_with_two_keyframes(const std::shared_ptr<data::keyframe>& keyfrm_1, const std::shared_ptr<data::keyframe>& keyfrm_2,
                                                    const std::vector<std::pair<unsigned int, unsigned int>>& matches) {
    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);
    const module::two_view_triangulator triangulator(keyfrm_1, keyfrm_2, 1.0);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(matches.size()); ++i) {
        const auto idx_1 = matches.at(i).first;
        const auto idx_2 = matches.at(i).second;

        // triangulate between idx_1 and idx_2
        Vec3_t pos_w;
        if (!triangulator.triangulate(idx_1, idx_2, pos_w)) {
            // failed
            continue;
        }
        // succeeded

        // create a landmark object
        auto lm = std::make_shared<data::landmark>(map_db_->next_landmark_id_++, pos_w, keyfrm_1);

        lm->connect_to_keyframe(keyfrm_1, idx_1);
        lm->connect_to_keyframe(keyfrm_2, idx_2);

        lm->compute_descriptor();
        lm->update_mean_normal_and_obs_scale_variance();

        map_db_->add_landmark(lm);
        // wait for redundancy check
#ifdef USE_OPENMP
#pragma omp critical
#endif
        {
            local_map_cleaner_->add_fresh_landmark(lm);
        }
    }
}

void mapping_module::update_new_keyframe() {
    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

    // get the targets to check landmark fusion
    const auto fuse_tgt_keyfrms = cur_keyfrm_->graph_node_->get_top_n_covisibilities(num_covisibilities_for_landmark_fusion_);

    // resolve the duplication of landmarks between the current keyframe and the targets
    nondeterministic::unordered_map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>> replaced_lms;
    fuse_landmark_duplication(fuse_tgt_keyfrms, replaced_lms);
    tracker_->replace_landmarks_in_last_frm(replaced_lms);

    // update the geometries
    const auto cur_landmarks = cur_keyfrm_->get_landmarks();
    for (const auto& lm : cur_landmarks) {
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        if (!lm->has_representative_descriptor()) {
            CV_LOG_WARNING(&g_log_tag, "has not representative descriptor " << lm->id_);
            lm->compute_descriptor();
        }
        if (!lm->has_valid_prediction_parameters()) {
            CV_LOG_WARNING(&g_log_tag, "has not valid prediction parameters");
            lm->update_mean_normal_and_obs_scale_variance();
        }
    }

    // update the graph (Because fuse_landmark_duplication changes the landmark)
    cur_keyfrm_->graph_node_->update_connections(map_db_->get_min_num_shared_lms());
}

void mapping_module::fuse_landmark_duplication(const std::vector<std::shared_ptr<data::keyframe>>& fuse_tgt_keyfrms,
                                               nondeterministic::unordered_map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>>& replaced_lms) {
    match::fuse fuse_matcher(0.6);

    {
        // reproject the landmarks observed in the current keyframe to each of the targets, and acquire
        // - additional matches
        // - duplication of matches
        // then, add matches and solve duplication
        auto cur_landmarks = cur_keyfrm_->get_landmarks();
        for (const auto& fuse_tgt_keyfrm : fuse_tgt_keyfrms) {
            std::unordered_map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>> duplicated_lms_in_keyfrm;
            std::unordered_map<unsigned int, std::shared_ptr<data::landmark>> new_connections;
            const Mat33_t rot_cw = fuse_tgt_keyfrm->get_rot_cw();
            const Vec3_t trans_cw = fuse_tgt_keyfrm->get_trans_cw();
            fuse_matcher.detect_duplication(fuse_tgt_keyfrm, rot_cw, trans_cw, cur_landmarks, 3.0, duplicated_lms_in_keyfrm, new_connections, true);

            // There is association between the 3D point and the keyframe
            // -> Duplication exists
            for (const auto& lms_pair : duplicated_lms_in_keyfrm) {
                auto lm_to_replace = lms_pair.first;
                auto lm_in_keyfrm = lms_pair.second;
                // Replace with more reliable 3D points (= more observable)
                assert(!replaced_lms.count(lm_in_keyfrm));
                assert(!replaced_lms.count(lm_to_replace));
                if (lm_to_replace->num_observations() < lm_in_keyfrm->num_observations()) {
                    std::swap(lm_to_replace, lm_in_keyfrm);
                }
                // Replace lm_in_keyfrm with lm_to_replace
                if (lm_to_replace->id_ != lm_in_keyfrm->id_) {
                    replaced_lms[lm_in_keyfrm] = lm_to_replace;
                    lm_in_keyfrm->replace(lm_to_replace, map_db_);
                    if (!lm_to_replace->has_representative_descriptor()) {
                        lm_to_replace->compute_descriptor();
                    }
                    if (!lm_to_replace->has_valid_prediction_parameters()) {
                        lm_to_replace->update_mean_normal_and_obs_scale_variance();
                    }
                }
            }

            for (const auto& best_idx_lm : new_connections) {
                const auto& best_idx = best_idx_lm.first;
                auto lm = best_idx_lm.second;
                while (replaced_lms.count(lm)) {
                    lm = replaced_lms[lm];
                }
                lm->connect_to_keyframe(fuse_tgt_keyfrm, best_idx);
                lm->update_mean_normal_and_obs_scale_variance();
                lm->compute_descriptor();
            }
        }
    }

    {
        // reproject the landmarks observed in each of the targets to each of the current frame, and acquire
        // - additional matches
        // - duplication of matches
        // then, add matches and solve duplication
        nondeterministic::unordered_set<std::shared_ptr<data::landmark>> candidate_landmarks_to_fuse;

        for (const auto& fuse_tgt_keyfrm : fuse_tgt_keyfrms) {
            const auto fuse_tgt_landmarks = fuse_tgt_keyfrm->get_landmarks();

            for (const auto& lm : fuse_tgt_landmarks) {
                if (!lm) {
                    continue;
                }
                if (lm->will_be_erased()) {
                    continue;
                }

                if (static_cast<bool>(candidate_landmarks_to_fuse.count(lm))) {
                    continue;
                }
                candidate_landmarks_to_fuse.insert(lm);
            }
        }

        std::unordered_map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>> duplicated_lms_in_keyfrm;
        std::unordered_map<unsigned int, std::shared_ptr<data::landmark>> new_connections;
        const Mat33_t rot_cw = cur_keyfrm_->get_rot_cw();
        const Vec3_t trans_cw = cur_keyfrm_->get_trans_cw();
        fuse_matcher.detect_duplication(cur_keyfrm_, rot_cw, trans_cw, candidate_landmarks_to_fuse, 3.0, duplicated_lms_in_keyfrm, new_connections, true);

        // There is association between the 3D point and the keyframe
        // -> Duplication exists
        for (const auto& lms_pair : duplicated_lms_in_keyfrm) {
            auto lm_to_replace = lms_pair.first;
            auto lm_in_keyfrm = lms_pair.second;
            // Replace with more reliable 3D points (= more observable)
            assert(!replaced_lms.count(lm_in_keyfrm));
            assert(!replaced_lms.count(lm_to_replace));
            if (lm_to_replace->num_observations() < lm_in_keyfrm->num_observations()) {
                std::swap(lm_to_replace, lm_in_keyfrm);
            }
            // Replace lm_in_keyfrm with lm_to_replace
            if (lm_to_replace->id_ != lm_in_keyfrm->id_) {
                replaced_lms[lm_in_keyfrm] = lm_to_replace;
                lm_in_keyfrm->replace(lm_to_replace, map_db_);
                if (!lm_to_replace->has_representative_descriptor()) {
                    lm_to_replace->compute_descriptor();
                }
                if (!lm_to_replace->has_valid_prediction_parameters()) {
                    lm_to_replace->update_mean_normal_and_obs_scale_variance();
                }
            }
        }

        for (const auto& best_idx_lm : new_connections) {
            const auto& best_idx = best_idx_lm.first;
            auto lm = best_idx_lm.second;
            while (replaced_lms.count(lm)) {
                lm = replaced_lms[lm];
            }
            lm->connect_to_keyframe(cur_keyfrm_, best_idx);
            lm->update_mean_normal_and_obs_scale_variance();
            lm->compute_descriptor();
        }
    }
}

std::shared_future<void> mapping_module::async_reset() {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    reset_is_requested_ = true;
    if (!future_reset_.valid()) {
        future_reset_ = promise_reset_.get_future().share();
    }
    return future_reset_;
}

bool mapping_module::reset_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    return reset_is_requested_;
}

void mapping_module::reset() {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    CV_LOG_INFO(&g_log_tag, "reset mapping module");
    keyfrms_queue_.clear();
    {
        std::lock_guard<std::mutex> lock_keyfrm_queue(mtx_keyfrm_queue_);
        while (!promise_add_keyfrm_queue_.empty()) {
            promise_add_keyfrm_queue_.front().set_value();
            promise_add_keyfrm_queue_.pop_front();
        }
    }
    local_map_cleaner_->reset();
    reset_is_requested_ = false;
    promise_reset_.set_value();
    promise_reset_ = std::promise<void>();
    future_reset_ = std::shared_future<void>();
}

std::shared_future<void> mapping_module::async_pause() {
    std::lock_guard<std::mutex> lock_pause(mtx_pause_);
    pause_is_requested_ = true;
    abort_local_BA_ = true;
    if (!future_pause_.valid()) {
        future_pause_ = promise_pause_.get_future().share();
    }

    std::lock_guard<std::mutex> lock_terminate(mtx_terminate_);
    CV_LOG_INFO(&g_log_tag, "reset mapping module");
    std::shared_future<void> future_pause = future_pause_;
    if (is_terminated_ || is_paused_) {
        promise_pause_.set_value();
        // Clear request
        promise_pause_ = std::promise<void>();
        future_pause_ = std::shared_future<void>();
    }
    return future_pause;
}

bool mapping_module::is_paused() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return is_paused_;
}

bool mapping_module::pause_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return pause_is_requested_;
}

void mapping_module::pause() {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    CV_LOG_INFO(&g_log_tag, "pause mapping module");
    is_paused_ = true;
    promise_pause_.set_value();
    promise_pause_ = std::promise<void>();
    future_pause_ = std::shared_future<void>();
}

void mapping_module::resume() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    std::lock_guard<std::mutex> lock2(mtx_terminate_);

    // if it has been already terminated, cannot resume
    if (is_terminated_) {
        return;
    }

    assert(keyfrms_queue_.empty());

    is_paused_ = false;
    pause_is_requested_ = false;

    CV_LOG_INFO(&g_log_tag, "resume mapping module");
}

std::shared_future<void> mapping_module::async_terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    terminate_is_requested_ = true;
    if (!future_terminate_.valid()) {
        future_terminate_ = promise_terminate_.get_future().share();
    }
    return future_terminate_;
}

bool mapping_module::is_terminated() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return is_terminated_;
}

bool mapping_module::terminate_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return terminate_is_requested_;
}

void mapping_module::terminate() {
    {
        std::lock_guard<std::mutex> lock_pause(mtx_pause_);
        is_paused_ = true;
        promise_pause_.set_value();
        promise_pause_ = std::promise<void>();
        future_pause_ = std::shared_future<void>();
    }
    {
        std::lock_guard<std::mutex> lock_terminate(mtx_terminate_);
        is_terminated_ = true;
        promise_terminate_.set_value();
        promise_terminate_ = std::promise<void>();
        future_terminate_ = std::shared_future<void>();
    }
}

} // namespace cv::slam
