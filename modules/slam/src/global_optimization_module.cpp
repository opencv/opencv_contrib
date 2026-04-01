#include "global_optimization_module.hpp"
#include "mapping_module.hpp"
#include "tracking_module.hpp"
#include "data/keyframe.hpp"
#include "data/landmark.hpp"
#include "data/map_database.hpp"
#include "match/fuse.hpp"
#include "util/converter.hpp"
#include "util/yaml.hpp"

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam_module", cv::utils::logging::LOG_LEVEL_INFO);

global_optimization_module::global_optimization_module(data::map_database* map_db, data::bow_database* bow_db,
                                                       data::bow_vocabulary* bow_vocab, const YAML::Node& yaml_node,
                                                       const bool fix_scale)
    : loop_detector_(new module::loop_detector(bow_db, bow_vocab, util::yaml_optional_ref(yaml_node, "LoopDetector"), fix_scale)),
      loop_bundle_adjuster_(new module::loop_bundle_adjuster(
          map_db,
          util::yaml_optional_ref(yaml_node, "GlobalOptimizer")["num_iter"].as<unsigned int>(10),
          util::yaml_optional_ref(yaml_node, "GlobalOptimizer")["use_huber_kernel"].as<bool>(false),
          util::yaml_optional_ref(yaml_node, "GlobalOptimizer")["verbose"].as<bool>(false))),
      map_db_(map_db),
      graph_optimizer_(new optimize::graph_optimizer(util::yaml_optional_ref(yaml_node, "GraphOptimizer"), fix_scale)),
      thr_neighbor_keyframes_(util::yaml_optional_ref(yaml_node, "GlobalOptimizer")["thr_neighbor_keyframes"].as<unsigned int>(15)) {
    CV_LOG_DEBUG(&g_log_tag, "CONSTRUCT: global_optimization_module");
}

global_optimization_module::~global_optimization_module() {
    abort_loop_BA();
    if (thread_for_loop_BA_) {
        thread_for_loop_BA_->join();
    }
    CV_LOG_DEBUG(&g_log_tag, "DESTRUCT: global_optimization_module");
}

void global_optimization_module::set_tracking_module(tracking_module* tracker) {
    tracker_ = tracker;
}

void global_optimization_module::set_mapping_module(mapping_module* mapper) {
    mapper_ = mapper;
    loop_bundle_adjuster_->set_mapping_module(mapper);
}

void global_optimization_module::enable_loop_detector() {
    CV_LOG_INFO(&g_log_tag, "enable loop detector");
    loop_detector_->enable_loop_detector();
}

void global_optimization_module::disable_loop_detector() {
    CV_LOG_INFO(&g_log_tag, "disable loop detector");
    loop_detector_->disable_loop_detector();
}

bool global_optimization_module::loop_detector_is_enabled() const {
    return loop_detector_->is_enabled();
}

bool global_optimization_module::request_loop_closure(unsigned int keyfrm1_id, unsigned int keyfrm2_id) {
    std::lock_guard<std::mutex> lock(mtx_loop_closure_request_);
    if (loop_closure_is_requested_) {
        CV_LOG_WARNING(&g_log_tag, "Can not process new loop closure request while previous was not finished");
        return false;
    }
    loop_closure_is_requested_ = true;
    loop_closure_request_.keyfrm1_id_ = keyfrm1_id;
    loop_closure_request_.keyfrm2_id_ = keyfrm2_id;
    return true;
}

bool global_optimization_module::loop_closure_is_requested() {
    std::lock_guard<std::mutex> lock(mtx_loop_closure_request_);
    return loop_closure_is_requested_;
}

loop_closure_request& global_optimization_module::get_loop_closure_request() {
    std::lock_guard<std::mutex> lock(mtx_loop_closure_request_);
    return loop_closure_request_;
}

void global_optimization_module::finish_loop_closure_request() {
    std::lock_guard<std::mutex> lock(mtx_loop_closure_request_);
    loop_closure_is_requested_ = false;
}

bool global_optimization_module::loop_closure(const loop_closure_request& request) {
    {
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);
        unsigned int curr_keyfrm_id = std::max(request.keyfrm1_id_, request.keyfrm2_id_);
        unsigned int candidate_keyfrm_id = std::min(request.keyfrm1_id_, request.keyfrm2_id_);
        // not to be removed during loop detection and correction
        cur_keyfrm_ = map_db_->get_keyframe(curr_keyfrm_id);
        if (cur_keyfrm_ == nullptr) {
            CV_LOG_INFO(&g_log_tag, "keyframe " << curr_keyfrm_id << " not found");
            return false;
        }
        cur_keyfrm_->set_not_to_be_erased();
        loop_detector_->set_current_keyframe(cur_keyfrm_);
        auto candidate_keyfrm = map_db_->get_keyframe(candidate_keyfrm_id);
        if (candidate_keyfrm == nullptr) {
            CV_LOG_INFO(&g_log_tag, "candidate keyframe " << candidate_keyfrm_id << " not found");
            return false;
        }
        loop_detector_->add_loop_candidate(candidate_keyfrm);

        // validate candidates and select ONE candidate from them
        if (!loop_detector_->validate_candidates()) {
            // could not find
            // allow the removal of the current keyframe
            cur_keyfrm_->set_to_be_erased();
            return false;
        }
    }

    correct_loop();
    finish_loop_closure_request();
    return true;
}

void global_optimization_module::run() {
    CV_LOG_INFO(&g_log_tag, "start global optimization module");

    is_terminated_ = false;

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        // check if termination is requested
        if (terminate_is_requested()) {
            // terminate and break
            terminate();
            break;
        }

        // check if loop closure is requested
        if (loop_closure_is_requested()) {
            loop_closure(get_loop_closure_request());
        }

        // check if pause is requested
        if (pause_is_requested()) {
            // pause and wait
            pause();
            // check if termination or reset is requested during pause
            while (is_paused() && !terminate_is_requested() && !reset_is_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            }
        }

        // check if reset is requested
        if (reset_is_requested()) {
            // reset and continue
            reset();
            continue;
        }

        // if the queue is empty, the following process is not needed
        if (!keyframe_is_queued()) {
            continue;
        }

        // dequeue the keyframe from the queue -> cur_keyfrm_
        {
            std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
            cur_keyfrm_ = keyfrms_queue_.front();
            keyfrms_queue_.pop_front();
        }

        {
            std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);
            // not to be removed during loop detection and correction
            cur_keyfrm_->set_not_to_be_erased();

            // pass the current keyframe to the loop detector
            loop_detector_->set_current_keyframe(cur_keyfrm_);

            // detect some loop candidate with BoW
            if (!loop_detector_->detect_loop_candidates()) {
                // could not find
                // allow the removal of the current keyframe
                cur_keyfrm_->set_to_be_erased();
                continue;
            }

            // validate candidates and select ONE candidate from them
            if (!loop_detector_->validate_candidates()) {
                // could not find
                // allow the removal of the current keyframe
                cur_keyfrm_->set_to_be_erased();
                continue;
            }
        }

        correct_loop();
    }

    CV_LOG_INFO(&g_log_tag, "terminate global optimization module");
}

void global_optimization_module::queue_keyframe(const std::shared_ptr<data::keyframe>& keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    keyfrms_queue_.push_back(keyfrm);
}

bool global_optimization_module::keyframe_is_queued() const {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    return !keyfrms_queue_.empty();
}

void global_optimization_module::correct_loop() {
    auto final_candidate_keyfrm = loop_detector_->get_selected_candidate_keyframe();

    CV_LOG_INFO(&g_log_tag, "detect loop: keyframe " << final_candidate_keyfrm->id_ << " - keyframe " << cur_keyfrm_->id_);

    if (cur_keyfrm_->graph_node_->get_spanning_root() != final_candidate_keyfrm->graph_node_->get_spanning_root()) {
        CV_LOG_WARNING(&g_log_tag, "The feature to merge two spanning trees has not yet been implemented.");
        return;
    }

    // 0. pre-processing

    // 0-1. stop the mapping module and the previous loop bundle adjuster

    // pause the mapping module
    CV_LOG_DEBUG(&g_log_tag, "global_optimization_module: pause the mapping module");
    auto future_pause = mapper_->async_pause();
    // abort the previous loop bundle adjuster
    if (thread_for_loop_BA_ || loop_bundle_adjuster_->is_running()) {
        CV_LOG_DEBUG(&g_log_tag, "global_optimization_module: abort loop bundle adjustment");
        abort_loop_BA();
    }
    // wait till the mapping module pauses
    future_pause.get();

    // 1. compute the Sim3 of the covisibilities of the current keyframe whose Sim3 is already estimated by the loop detector
    //    then, the covisibilities are moved to the corrected positions
    //    finally, landmarks observed in them are also moved to the correct position using the camera poses before and after camera pose correction

    CV_LOG_DEBUG(&g_log_tag, "global_optimization_module: compute the Sim3 of the covisibilities of the current keyframe whose Sim3 is already estimated by the loop detector");
    // acquire the covisibilities of the current keyframe
    std::vector<std::shared_ptr<data::keyframe>> curr_neighbors = cur_keyfrm_->graph_node_->get_covisibilities_over_min_num_shared_lms(thr_neighbor_keyframes_);
    curr_neighbors.push_back(cur_keyfrm_);

    // Sim3 camera poses BEFORE loop correction
    module::keyframe_Sim3_pairs_t Sim3s_nw_before_correction;
    // Sim3 camera poses AFTER loop correction
    module::keyframe_Sim3_pairs_t Sim3s_nw_after_correction;

    std::unordered_map<unsigned int, unsigned int> found_lm_to_ref_keyfrm_id;
    const auto g2o_Sim3_cw_after_correction = loop_detector_->get_Sim3_world_to_current();
    {
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        // camera pose of the current keyframe BEFORE loop correction
        const Mat44_t cam_pose_wc_before_correction = cur_keyfrm_->get_pose_wc();

        // compute Sim3s BEFORE loop correction
        Sim3s_nw_before_correction = get_Sim3s_before_loop_correction(curr_neighbors);
        // compute Sim3s AFTER loop correction
        Sim3s_nw_after_correction = get_Sim3s_after_loop_correction(cam_pose_wc_before_correction, g2o_Sim3_cw_after_correction, curr_neighbors);

        // correct covibisibility landmark positions
        correct_covisibility_landmarks(Sim3s_nw_before_correction, Sim3s_nw_after_correction, found_lm_to_ref_keyfrm_id);
        // correct covisibility keyframe camera poses
        correct_covisibility_keyframes(Sim3s_nw_after_correction);
    }

    // 2. resolve duplications of landmarks caused by loop fusion

    CV_LOG_DEBUG(&g_log_tag, "global_optimization_module: resolve duplications of landmarks caused by loop fusion");
    const auto curr_match_lms_observed_in_cand = loop_detector_->current_matched_landmarks_observed_in_candidate();
    replace_duplicated_landmarks(curr_match_lms_observed_in_cand, Sim3s_nw_after_correction);

    // 3. extract the new connections created after loop fusion

    CV_LOG_DEBUG(&g_log_tag, "global_optimization_module: extract the new connections created after loop fusion");
    const auto new_connections = extract_new_connections(curr_neighbors);

    // 4. pose graph optimization

    CV_LOG_DEBUG(&g_log_tag, "global_optimization_module: pose graph optimization");
    graph_optimizer_->optimize(final_candidate_keyfrm, cur_keyfrm_, Sim3s_nw_before_correction, Sim3s_nw_after_correction, new_connections, found_lm_to_ref_keyfrm_id);

    // add a loop edge
    final_candidate_keyfrm->graph_node_->add_loop_edge(cur_keyfrm_);
    cur_keyfrm_->graph_node_->add_loop_edge(final_candidate_keyfrm);

    // 5. launch loop BA

    CV_LOG_DEBUG(&g_log_tag, "global_optimization_module: wait for loop BA");
    while (loop_bundle_adjuster_->is_running()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }
    if (thread_for_loop_BA_) {
        CV_LOG_DEBUG(&g_log_tag, "global_optimization_module: wait for last loop BA");
        thread_for_loop_BA_->join();
        thread_for_loop_BA_.reset(nullptr);
    }
    CV_LOG_DEBUG(&g_log_tag, "global_optimization_module: launch loop BA");
    thread_for_loop_BA_ = std::unique_ptr<std::thread>(new std::thread(&module::loop_bundle_adjuster::optimize, loop_bundle_adjuster_.get(), cur_keyfrm_));

    // 6. post-processing

    CV_LOG_DEBUG(&g_log_tag, "global_optimization_module: resume the mapping module");
    // resume the mapping module
    mapper_->resume();

    // set the loop fusion information to the loop detector
    loop_detector_->set_loop_correct_keyframe_id(cur_keyfrm_->id_);
}

module::keyframe_Sim3_pairs_t global_optimization_module::get_Sim3s_before_loop_correction(const std::vector<std::shared_ptr<data::keyframe>>& neighbors) const {
    module::keyframe_Sim3_pairs_t Sim3s_nw_before_loop_correction;

    for (const auto& neighbor : neighbors) {
        // camera pose of `neighbor` BEFORE loop correction
        const Mat44_t cam_pose_nw = neighbor->get_pose_cw();
        // create Sim3 from SE3
        const Mat33_t& rot_nw = cam_pose_nw.block<3, 3>(0, 0);
        const Vec3_t& trans_nw = cam_pose_nw.block<3, 1>(0, 3);
        const g2o::Sim3 Sim3_nw_before_correction(rot_nw, trans_nw, 1.0);
        Sim3s_nw_before_loop_correction[neighbor] = Sim3_nw_before_correction;
    }

    return Sim3s_nw_before_loop_correction;
}

module::keyframe_Sim3_pairs_t global_optimization_module::get_Sim3s_after_loop_correction(const Mat44_t& cam_pose_wc_before_correction,
                                                                                          const g2o::Sim3& g2o_Sim3_cw_after_correction,
                                                                                          const std::vector<std::shared_ptr<data::keyframe>>& neighbors) const {
    module::keyframe_Sim3_pairs_t Sim3s_nw_after_loop_correction;

    for (auto neighbor : neighbors) {
        // camera pose of `neighbor` BEFORE loop correction
        const Mat44_t cam_pose_nw_before_correction = neighbor->get_pose_cw();
        // create the relative Sim3 from the current to `neighbor`
        const Mat44_t cam_pose_nc = cam_pose_nw_before_correction * cam_pose_wc_before_correction;
        const Mat33_t& rot_nc = cam_pose_nc.block<3, 3>(0, 0);
        const Vec3_t& trans_nc = cam_pose_nc.block<3, 1>(0, 3);
        const g2o::Sim3 Sim3_nc(rot_nc, trans_nc, 1.0);
        // compute the camera poses AFTER loop correction of the neighbors
        const g2o::Sim3 Sim3_nw_after_correction = Sim3_nc * g2o_Sim3_cw_after_correction;
        Sim3s_nw_after_loop_correction[neighbor] = Sim3_nw_after_correction;
    }

    return Sim3s_nw_after_loop_correction;
}

void global_optimization_module::correct_covisibility_landmarks(const module::keyframe_Sim3_pairs_t& Sim3s_nw_before_correction,
                                                                const module::keyframe_Sim3_pairs_t& Sim3s_nw_after_correction,
                                                                std::unordered_map<unsigned int, unsigned int>& found_lm_to_ref_keyfrm_id) const {
    for (const auto& t : Sim3s_nw_after_correction) {
        auto neighbor = t.first;
        // neighbor->world AFTER loop correction
        const auto Sim3_wn_after_correction = t.second.inverse();
        // world->neighbor BEFORE loop correction
        const auto& Sim3_nw_before_correction = Sim3s_nw_before_correction.at(neighbor);

        const auto ngh_landmarks = neighbor->get_landmarks();
        for (const auto& lm : ngh_landmarks) {
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            // avoid duplication
            if (found_lm_to_ref_keyfrm_id.count(lm->id_)) {
                continue;
            }
            // record the reference keyframe used in loop fusion of landmarks
            found_lm_to_ref_keyfrm_id[lm->id_] = neighbor->id_;

            // correct position of `lm`
            const Vec3_t pos_w_before_correction = lm->get_pos_in_world();
            const Vec3_t pos_w_after_correction = Sim3_wn_after_correction.map(Sim3_nw_before_correction.map(pos_w_before_correction));
            lm->set_pos_in_world(pos_w_after_correction);
            // update geometry
            lm->update_mean_normal_and_obs_scale_variance();
        }
    }
}

void global_optimization_module::correct_covisibility_keyframes(const module::keyframe_Sim3_pairs_t& Sim3s_nw_after_correction) const {
    for (const auto& t : Sim3s_nw_after_correction) {
        auto neighbor = t.first;
        const auto Sim3_nw_after_correction = t.second;

        const auto s_nw = Sim3_nw_after_correction.scale();
        const Mat33_t rot_nw = Sim3_nw_after_correction.rotation().toRotationMatrix();
        const Vec3_t trans_nw = Sim3_nw_after_correction.translation() / s_nw;
        const Mat44_t cam_pose_nw = util::converter::to_eigen_pose(rot_nw, trans_nw);
        neighbor->set_pose_cw(cam_pose_nw);
    }
}

void global_optimization_module::replace_duplicated_landmarks(const std::vector<std::shared_ptr<data::landmark>>& curr_match_lms_observed_in_cand,
                                                              const module::keyframe_Sim3_pairs_t& Sim3s_nw_after_correction) const {
    nondeterministic::unordered_map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>> replaced_lms;
    // resolve duplications of landmarks between the current keyframe and the loop candidate
    {
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        for (unsigned int idx = 0; idx < cur_keyfrm_->frm_obs_.undist_keypts_.size(); ++idx) {
            auto curr_match_lm_in_cand = curr_match_lms_observed_in_cand.at(idx);
            if (!curr_match_lm_in_cand) {
                continue;
            }
            if (curr_match_lm_in_cand->will_be_erased()) {
                continue;
            }

            if (curr_match_lm_in_cand->is_observed_in_keyframe(cur_keyfrm_)) {
                cur_keyfrm_->erase_landmark(curr_match_lm_in_cand);
                curr_match_lm_in_cand->erase_observation(map_db_, cur_keyfrm_);
            }

            const auto& lm_in_curr = cur_keyfrm_->get_landmark(idx);
            if (lm_in_curr) {
                // if the landmark corresponding `idx` exists,
                // replace it with `curr_match_lm_in_cand` (observed in the candidate)
                if (lm_in_curr->id_ != curr_match_lm_in_cand->id_) {
                    replaced_lms[lm_in_curr] = curr_match_lm_in_cand;
                    lm_in_curr->replace(curr_match_lm_in_cand, map_db_);
                    if (!curr_match_lm_in_cand->has_representative_descriptor()) {
                        curr_match_lm_in_cand->compute_descriptor();
                    }
                    if (!curr_match_lm_in_cand->has_valid_prediction_parameters()) {
                        curr_match_lm_in_cand->update_mean_normal_and_obs_scale_variance();
                    }
                }
            }
            else {
                // if landmark corresponding `idx` does not exists,
                // add association between the current keyframe and `curr_match_lm_in_cand`
                curr_match_lm_in_cand->connect_to_keyframe(cur_keyfrm_, idx);
                curr_match_lm_in_cand->update_mean_normal_and_obs_scale_variance();
                curr_match_lm_in_cand->compute_descriptor();
            }
        }
    }

    // resolve duplications of landmarks between the current keyframe and the candidates of the loop candidate
    auto curr_match_lms_observed_in_cand_covis = loop_detector_->current_matched_landmarks_observed_in_candidate_covisibilities();
    match::fuse fuse_matcher(0.8);
    for (const auto& t : Sim3s_nw_after_correction) {
        auto neighbor = t.first;
        const Mat44_t Sim3_nw_after_correction = util::converter::to_eigen_mat(t.second);

        // reproject the landmarks observed in the current keyframe to the neighbor,
        // then search duplication of the landmarks
        std::unordered_map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>> duplicated_lms_in_keyfrm;
        std::unordered_map<unsigned int, std::shared_ptr<data::landmark>> new_connections;
        // Convert Sim3 into SE3
        const Mat33_t s_rot_cw = Sim3_nw_after_correction.block<3, 3>(0, 0);
        const auto s_cw = std::sqrt(s_rot_cw.block<1, 3>(0, 0).dot(s_rot_cw.block<1, 3>(0, 0)));
        const Mat33_t rot_cw = s_rot_cw / s_cw;
        const Vec3_t trans_cw = Sim3_nw_after_correction.block<3, 1>(0, 3) / s_cw;
        fuse_matcher.detect_duplication(neighbor, rot_cw, trans_cw, curr_match_lms_observed_in_cand_covis, 4.0, duplicated_lms_in_keyfrm, new_connections);

        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        for (const auto& best_idx_lm : new_connections) {
            const auto& best_idx = best_idx_lm.first;
            const auto& lm = best_idx_lm.second;
            lm->connect_to_keyframe(neighbor, best_idx);
            lm->update_mean_normal_and_obs_scale_variance();
            lm->compute_descriptor();
        }

        // if any landmark duplication is found, replace it
        for (const auto& lms_pair : duplicated_lms_in_keyfrm) {
            const auto& lm_to_replace = lms_pair.first;
            const auto& lm_in_neighbor = lms_pair.second;
            if (lm_to_replace->id_ != lm_in_neighbor->id_) {
                replaced_lms[lm_to_replace] = lm_in_neighbor;
                lm_to_replace->replace(lm_in_neighbor, map_db_);
                if (!lm_in_neighbor->has_representative_descriptor()) {
                    lm_in_neighbor->compute_descriptor();
                }
                if (!lm_in_neighbor->has_valid_prediction_parameters()) {
                    lm_in_neighbor->update_mean_normal_and_obs_scale_variance();
                }
            }
        }
    }
    tracker_->replace_landmarks_in_last_frm(replaced_lms);
}

auto global_optimization_module::extract_new_connections(const std::vector<std::shared_ptr<data::keyframe>>& covisibilities) const
    -> std::map<std::shared_ptr<data::keyframe>, std::set<std::shared_ptr<data::keyframe>>> {
    std::map<std::shared_ptr<data::keyframe>, std::set<std::shared_ptr<data::keyframe>>> new_connections;

    for (auto covisibility : covisibilities) {
        // acquire neighbors BEFORE loop fusion (because update_connections() is not called yet)
        const auto neighbors_before_update = covisibility->graph_node_->get_covisibilities();

        // call update_connections()
        covisibility->graph_node_->update_connections(map_db_->get_min_num_shared_lms());
        // acquire neighbors AFTER loop fusion
        new_connections[covisibility] = covisibility->graph_node_->get_connected_keyframes();

        // remove covisibilities
        for (const auto& keyfrm_to_erase : covisibilities) {
            new_connections.at(covisibility).erase(keyfrm_to_erase);
        }
        // remove nighbors before loop fusion
        for (const auto& keyfrm_to_erase : neighbors_before_update) {
            new_connections.at(covisibility).erase(keyfrm_to_erase);
        }
    }

    return new_connections;
}

std::shared_future<void> global_optimization_module::async_reset() {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    reset_is_requested_ = true;
    if (!future_reset_.valid()) {
        future_reset_ = promise_reset_.get_future().share();
    }
    return future_reset_;
}

bool global_optimization_module::reset_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    return reset_is_requested_;
}

void global_optimization_module::reset() {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    CV_LOG_INFO(&g_log_tag, "reset global optimization module");
    keyfrms_queue_.clear();
    loop_detector_->set_loop_correct_keyframe_id(0);
    reset_is_requested_ = false;
    promise_reset_.set_value();
    promise_reset_ = std::promise<void>();
    future_reset_ = std::shared_future<void>();
}

std::shared_future<void> global_optimization_module::async_pause() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    pause_is_requested_ = true;
    if (!future_pause_.valid()) {
        future_pause_ = promise_pause_.get_future().share();
    }
    return future_pause_;
}

bool global_optimization_module::pause_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return pause_is_requested_;
}

bool global_optimization_module::is_paused() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return is_paused_;
}

void global_optimization_module::pause() {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    CV_LOG_INFO(&g_log_tag, "pause global optimization module");
    is_paused_ = true;
    promise_pause_.set_value();
    promise_pause_ = std::promise<void>();
    future_pause_ = std::shared_future<void>();
}

void global_optimization_module::resume() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    std::lock_guard<std::mutex> lock2(mtx_terminate_);

    // if it has been already terminated, cannot resume
    if (is_terminated_) {
        return;
    }

    is_paused_ = false;
    pause_is_requested_ = false;

    CV_LOG_INFO(&g_log_tag, "resume global optimization module");
}

std::shared_future<void> global_optimization_module::async_terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    terminate_is_requested_ = true;
    if (!future_terminate_.valid()) {
        future_terminate_ = promise_terminate_.get_future().share();
    }
    return future_terminate_;
}

bool global_optimization_module::is_terminated() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return is_terminated_;
}

bool global_optimization_module::terminate_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return terminate_is_requested_;
}

void global_optimization_module::terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    is_terminated_ = true;
    promise_terminate_.set_value();
    promise_terminate_ = std::promise<void>();
    future_terminate_ = std::shared_future<void>();
}

bool global_optimization_module::loop_BA_is_running() const {
    return loop_bundle_adjuster_->is_running();
}

void global_optimization_module::abort_loop_BA() {
    loop_bundle_adjuster_->abort();
}

} // namespace cv::slam
