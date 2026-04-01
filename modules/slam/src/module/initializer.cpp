#include "config.hpp"
#include "data/keyframe.hpp"
#include "data/landmark.hpp"
#include "data/marker.hpp"
#include "data/map_database.hpp"
#include "initialize/bearing_vector.hpp"
#include "initialize/perspective.hpp"
#include "marker_model/base.hpp"
#include "match/area.hpp"
#include "module/initializer.hpp"
#include "module/marker_initializer.hpp"
#include "optimize/global_bundle_adjuster.hpp"

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace module {

initializer::initializer(data::map_database* map_db,
                         const YAML::Node& yaml_node)
    : map_db_(map_db),
      num_ransac_iters_(yaml_node["num_ransac_iterations"].as<unsigned int>(100)),
      min_num_valid_pts_(yaml_node["min_num_valid_pts"].as<unsigned int>(50)),
      min_num_triangulated_pts_(yaml_node["min_num_triangulated_pts"].as<unsigned int>(50)),
      parallax_deg_thr_(yaml_node["parallax_deg_threshold"].as<float>(1.0)),
      reproj_err_thr_(yaml_node["reprojection_error_threshold"].as<float>(4.0)),
      num_ba_iters_(yaml_node["num_ba_iterations"].as<unsigned int>(100)),
      scaling_factor_(yaml_node["scaling_factor"].as<float>(1.0)),
      use_fixed_seed_(yaml_node["use_fixed_seed"].as<bool>(false)),
      gain_threshold_(yaml_node["gain_threshold"].as<float>(1e-5)),
      verbose_(yaml_node["verbose"].as<bool>(false)) {
    CV_LOG_DEBUG(&g_log_tag, "CONSTRUCT: module::initializer");
}

initializer::~initializer() {
    CV_LOG_DEBUG(&g_log_tag, "DESTRUCT: module::initializer");
}

void initializer::reset() {
    initializer_.reset(nullptr);
    state_ = initializer_state_t::NotReady;
    init_frm_id_ = 0;
    init_frm_stamp_ = 0.0;
}

initializer_state_t initializer::get_state() const {
    return state_;
}

unsigned int initializer::get_initial_frame_id() const {
    return init_frm_id_;
}

double initializer::get_initial_frame_timestamp() const {
    return init_frm_stamp_;
}

bool initializer::get_use_fixed_seed() const {
    return use_fixed_seed_;
}

bool initializer::initialize(const camera::setup_type_t setup_type,
                             data::bow_vocabulary* bow_vocab, data::frame& curr_frm) {
    switch (setup_type) {
        case camera::setup_type_t::Monocular: {
            // construct an initializer if not constructed
            if (state_ == initializer_state_t::NotReady) {
                create_initializer(curr_frm);
                return false;
            }

            // try to initialize
            if (!try_initialize_for_monocular(curr_frm)) {
                // failed
                return false;
            }

            // create new map if succeeded
            create_map_for_monocular(bow_vocab, curr_frm);
            break;
        }
        case camera::setup_type_t::Stereo:
        case camera::setup_type_t::RGBD: {
            state_ = initializer_state_t::Initializing;

            // try to initialize
            if (!try_initialize_for_stereo(curr_frm)) {
                // failed
                return false;
            }

            // create new map if succeeded
            create_map_for_stereo(bow_vocab, curr_frm);
            break;
        }
        default: {
            throw std::runtime_error("Undefined camera setup");
        }
    }

    // check the state is succeeded or not
    if (state_ == initializer_state_t::Succeeded) {
        init_frm_id_ = curr_frm.id_;
        init_frm_stamp_ = curr_frm.timestamp_;
        return true;
    }
    else {
        return false;
    }
}

void initializer::create_initializer(data::frame& curr_frm) {
    // set the initial frame
    init_frm_ = data::frame(curr_frm);

    // initialize the previously matched coordinates
    prev_matched_coords_.resize(init_frm_.frm_obs_.undist_keypts_.size());
    for (unsigned int i = 0; i < init_frm_.frm_obs_.undist_keypts_.size(); ++i) {
        prev_matched_coords_.at(i) = init_frm_.frm_obs_.undist_keypts_.at(i).pt;
    }

    // initialize matchings (init_idx -> curr_idx)
    std::fill(init_matches_.begin(), init_matches_.end(), -1);

    // build a initializer
    initializer_.reset(nullptr);
    switch (init_frm_.camera_->model_type_) {
        case camera::model_type_t::Perspective:
        case camera::model_type_t::Fisheye:
        case camera::model_type_t::RadialDivision: {
            initializer_ = std::unique_ptr<initialize::perspective>(
                new initialize::perspective(
                    init_frm_, num_ransac_iters_, min_num_triangulated_pts_, min_num_valid_pts_,
                    parallax_deg_thr_, reproj_err_thr_, use_fixed_seed_));
            break;
        }
        case camera::model_type_t::Equirectangular: {
            initializer_ = std::unique_ptr<initialize::bearing_vector>(
                new initialize::bearing_vector(
                    init_frm_, num_ransac_iters_, min_num_triangulated_pts_, min_num_valid_pts_,
                    parallax_deg_thr_, reproj_err_thr_, use_fixed_seed_));
            break;
        }
    }

    state_ = initializer_state_t::Initializing;
}

bool initializer::try_initialize_for_monocular(data::frame& curr_frm) {
    assert(state_ == initializer_state_t::Initializing);

    match::area matcher(0.9, true);
    const auto num_matches = matcher.match_in_consistent_area(init_frm_, curr_frm, prev_matched_coords_, init_matches_, 100);

    if (num_matches < min_num_valid_pts_) {
        // rebuild the initializer with the next frame
        reset();
        return false;
    }

    // try to initialize with the initial frame and the current frame
    assert(initializer_);
    CV_LOG_DEBUG(&g_log_tag, "try to initialize with the initial frame and the current frame: frame " << init_frm_.id_ << " - frame " << curr_frm.id_);
    return initializer_->initialize(curr_frm, init_matches_);
}

bool initializer::create_map_for_monocular(data::bow_vocabulary* bow_vocab, data::frame& curr_frm) {
    assert(state_ == initializer_state_t::Initializing);

    eigen_alloc_vector<Vec3_t> init_triangulated_pts;
    {
        assert(initializer_);
        init_triangulated_pts = initializer_->get_triangulated_pts();
        const auto is_triangulated = initializer_->get_triangulated_flags();

        // make invalid the matchings which have not been triangulated
        for (unsigned int i = 0; i < init_matches_.size(); ++i) {
            if (init_matches_.at(i) < 0) {
                continue;
            }
            if (is_triangulated.at(i)) {
                continue;
            }
            init_matches_.at(i) = -1;
        }

        // set the camera poses
        init_frm_.set_pose_cw(Mat44_t::Identity());
        Mat44_t cam_pose_cw = Mat44_t::Identity();
        cam_pose_cw.block<3, 3>(0, 0) = initializer_->get_rotation_ref_to_cur();
        cam_pose_cw.block<3, 1>(0, 3) = initializer_->get_translation_ref_to_cur();
        curr_frm.set_pose_cw(cam_pose_cw);

        // destruct the initializer
        initializer_.reset(nullptr);
    }

    // create initial keyframes
    auto init_keyfrm = data::keyframe::make_keyframe(map_db_->next_keyframe_id_++, init_frm_);
    auto curr_keyfrm = data::keyframe::make_keyframe(map_db_->next_keyframe_id_++, curr_frm);
    curr_keyfrm->graph_node_->set_spanning_parent(init_keyfrm);
    init_keyfrm->graph_node_->add_spanning_child(curr_keyfrm);
    init_keyfrm->graph_node_->set_spanning_root(init_keyfrm);
    curr_keyfrm->graph_node_->set_spanning_root(init_keyfrm);
    map_db_->add_spanning_root(init_keyfrm);

    // compute BoW representations
    if (bow_vocab) {
        init_keyfrm->compute_bow(bow_vocab);
        curr_keyfrm->compute_bow(bow_vocab);
    }

    // add the keyframes to the map DB
    map_db_->add_keyframe(init_keyfrm);
    map_db_->add_keyframe(curr_keyfrm);

    // update the frame statistics
    init_frm_.ref_keyfrm_ = init_keyfrm;
    curr_frm.ref_keyfrm_ = curr_keyfrm;
    map_db_->update_frame_statistics(init_frm_, false);
    map_db_->update_frame_statistics(curr_frm, false);

    // assign 2D-3D associations
    std::vector<std::shared_ptr<data::landmark>> lms;
    for (unsigned int init_idx = 0; init_idx < init_matches_.size(); init_idx++) {
        const auto curr_idx = init_matches_.at(init_idx);
        if (curr_idx < 0) {
            continue;
        }

        // construct a landmark
        auto lm = std::make_shared<data::landmark>(map_db_->next_landmark_id_++, init_triangulated_pts.at(init_idx), curr_keyfrm);

        // set the assocications to the new keyframes
        lm->connect_to_keyframe(init_keyfrm, init_idx);
        lm->connect_to_keyframe(curr_keyfrm, curr_idx);

        // update the descriptor
        lm->compute_descriptor();
        // update the geometry
        lm->update_mean_normal_and_obs_scale_variance();

        // set the 2D-3D assocications to the current frame
        curr_frm.add_landmark(lm, curr_idx);

        // add the landmark to the map DB
        map_db_->add_landmark(lm);
        lms.push_back(lm);
    }

    bool indefinite_scale = true;
    for (const auto& id_mkr2d : init_keyfrm->markers_2d_) {
        if (curr_keyfrm->markers_2d_.count(id_mkr2d.first)) {
            indefinite_scale = false;
            break;
        }
    }

    // assign marker associations
    std::vector<std::shared_ptr<data::marker>> markers;
    const auto assign_marker_associations = [this, &markers](const std::shared_ptr<data::keyframe>& keyfrm) {
        for (const auto& id_mkr2d : keyfrm->markers_2d_) {
            auto marker = map_db_->get_marker(id_mkr2d.first);
            if (!marker) {
                auto mkr2d = id_mkr2d.second;
                eigen_alloc_vector<Vec3_t> corners_pos_w = mkr2d.compute_corners_pos_w(keyfrm->get_pose_wc(), mkr2d.marker_model_->corners_pos_);
                marker = std::make_shared<data::marker>(corners_pos_w, id_mkr2d.first, mkr2d.marker_model_);
                // add the marker to the map DB
                map_db_->add_marker(marker);
                markers.push_back(marker);
            }
            // Set the association to the new marker
            keyfrm->add_marker(marker);
            marker->observations_.emplace(keyfrm->id_, keyfrm);
        }
    };
    assign_marker_associations(init_keyfrm);
    assign_marker_associations(curr_keyfrm);

    // global bundle adjustment
    const auto global_bundle_adjuster = optimize::global_bundle_adjuster(num_ba_iters_, true, verbose_);
    std::vector<std::shared_ptr<data::keyframe>> keyfrms{init_keyfrm, curr_keyfrm};
    if (markers.size() > 0) {
        // Adjust map scale with reference to marker width.
        global_bundle_adjuster.optimize_for_initialization(keyfrms, lms, markers, gain_threshold_, true);
    }
    global_bundle_adjuster.optimize_for_initialization(keyfrms, lms, markers, gain_threshold_, false);

    if (indefinite_scale) {
        // scale the map so that the median of depths is 1.0
        float median_scale;
        if (init_keyfrm->camera_->model_type_ == camera::model_type_t::Equirectangular) {
            median_scale = init_keyfrm->compute_median_distance();
        }
        else {
            median_scale = init_keyfrm->compute_median_depth(true);
        }
        const auto inv_median_scale = 1.0 / median_scale;
        if (curr_keyfrm->get_num_tracked_landmarks(1) < min_num_triangulated_pts_ && median_scale < 0) {
            CV_LOG_INFO(&g_log_tag, "seems to be wrong initialization, resetting");
            state_ = initializer_state_t::Wrong;
            return false;
        }
        scale_map(init_keyfrm, curr_keyfrm, inv_median_scale * scaling_factor_);
    }

    // update the current frame pose
    curr_frm.set_pose_cw(curr_keyfrm->get_pose_cw());

    CV_LOG_INFO(&g_log_tag, "new map created with " << map_db_->get_num_landmarks() << " points: frame " << init_frm_.id_ << " - frame " << curr_frm.id_);
    state_ = initializer_state_t::Succeeded;
    return true;
}

void initializer::scale_map(const std::shared_ptr<data::keyframe>& init_keyfrm, const std::shared_ptr<data::keyframe>& curr_keyfrm, const double scale) {
    // scaling keyframes
    Mat44_t cam_pose_cw = curr_keyfrm->get_pose_cw();
    cam_pose_cw.block<3, 1>(0, 3) *= scale;
    curr_keyfrm->set_pose_cw(cam_pose_cw);

    // scaling landmarks
    const auto landmarks = init_keyfrm->get_landmarks();
    for (const auto& lm : landmarks) {
        if (!lm) {
            continue;
        }
        lm->set_pos_in_world(lm->get_pos_in_world() * scale);
        lm->update_mean_normal_and_obs_scale_variance();
    }
}

bool initializer::try_initialize_for_stereo(data::frame& curr_frm) {
    assert(state_ == initializer_state_t::Initializing);
    // count the number of valid depths
    unsigned int num_valid_depths = std::count_if(curr_frm.frm_obs_.depths_.begin(), curr_frm.frm_obs_.depths_.end(),
                                                  [](const float depth) {
                                                      return 0 < depth;
                                                  });
    return min_num_triangulated_pts_ <= num_valid_depths;
}

bool initializer::create_map_for_stereo(data::bow_vocabulary* bow_vocab, data::frame& curr_frm) {
    assert(state_ == initializer_state_t::Initializing);

    // create an initial keyframe
    curr_frm.set_pose_cw(Mat44_t::Identity());
    auto curr_keyfrm = data::keyframe::make_keyframe(map_db_->next_keyframe_id_++, curr_frm);
    curr_keyfrm->graph_node_->set_spanning_root(curr_keyfrm);
    map_db_->add_spanning_root(curr_keyfrm);

    // compute BoW representation
    if (bow_vocab) {
        curr_keyfrm->compute_bow(bow_vocab);
    }

    // add to the map DB
    map_db_->add_keyframe(curr_keyfrm);

    // update the frame statistics
    curr_frm.ref_keyfrm_ = curr_keyfrm;
    map_db_->update_frame_statistics(curr_frm, false);

    for (unsigned int idx = 0; idx < curr_frm.frm_obs_.undist_keypts_.size(); ++idx) {
        // add a new landmark if tht corresponding depth is valid
        const auto z = curr_frm.frm_obs_.depths_.at(idx);
        if (z <= 0) {
            continue;
        }

        // build a landmark
        const Vec3_t pos_w = curr_frm.triangulate_stereo(idx);
        auto lm = std::make_shared<data::landmark>(map_db_->next_landmark_id_++, pos_w, curr_keyfrm);

        // set the associations to the new keyframe
        lm->connect_to_keyframe(curr_keyfrm, idx);

        // update the descriptor
        lm->compute_descriptor();
        // update the geometry
        lm->update_mean_normal_and_obs_scale_variance();

        // set the 2D-3D associations to the current frame
        curr_frm.add_landmark(lm, idx);

        // add the landmark to the map DB
        map_db_->add_landmark(lm);
    }

    CV_LOG_INFO(&g_log_tag, "new map created with " << map_db_->get_num_landmarks() << " points: frame " << curr_frm.id_);
    state_ = initializer_state_t::Succeeded;
    return true;
}

} // namespace module
} // namespace cv::slam
