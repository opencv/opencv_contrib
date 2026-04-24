#include "system.hpp"
#include "config.hpp"
#include "tracking_module.hpp"
#include "mapping_module.hpp"
#include "global_optimization_module.hpp"
#include "camera/camera_factory.hpp"
#include "data/camera_database.hpp"
#include "data/common.hpp"
#include "data/frame_observation.hpp"
#include "data/orb_params_database.hpp"
#include "data/map_database.hpp"
#include "data/bow_database.hpp"
#include "data/bow_vocabulary.hpp"
#include "data/marker2d.hpp"
#include "data/marker.hpp"
#include "match/stereo.hpp"
#include "feature/orb_extractor.hpp"
#include "io/trajectory_io.hpp"
#include "io/map_database_io_factory.hpp"
#include "util/converter.hpp"
#include "util/image_converter.hpp"
#include "util/yaml.hpp"

#include <thread>

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);

system::system(const std::shared_ptr<config>& cfg, const std::string& vocab_file_path)
    : cfg_(cfg) {
    CV_LOG_DEBUG(&g_log_tag, "CONSTRUCT: system");
    print_info();

    // load ORB vocabulary
    if (!vocab_file_path.empty()) {
        CV_LOG_INFO(&g_log_tag, "loading ORB vocabulary: " << vocab_file_path);
        bow_vocab_ = data::bow_vocabulary_util::load(vocab_file_path);
    }
    else {
        CV_LOG_DEBUG(&g_log_tag, "Running without vocabulary");
    }

    const auto system_params = util::yaml_optional_ref(cfg->yaml_node_, "System");

    camera_ = camera::camera_factory::create(util::yaml_optional_ref(cfg->yaml_node_, "Camera"));
    orb_params_ = new feature::orb_params(util::yaml_optional_ref(cfg->yaml_node_, "Feature"));
    CV_LOG_INFO(&g_log_tag, "load orb_params \"" << orb_params_->name_ << "\"");

    // database
    cam_db_ = new data::camera_database();
    cam_db_->add_camera(camera_);
    map_db_ = new data::map_database(system_params["min_num_shared_lms"].as<unsigned int>(15));
    if (bow_vocab_) {
        bow_db_ = new data::bow_database(bow_vocab_);
    }
    orb_params_db_ = new data::orb_params_database();
    orb_params_db_->add_orb_params(orb_params_);

    // map I/O
    auto map_format = system_params["map_format"].as<std::string>("msgpack");
    map_database_io_ = io::map_database_io_factory::create(map_format);

    // tracking module
    tracker_ = new tracking_module(cfg_, camera_, map_db_, bow_vocab_, bow_db_);
    // mapping module
    mapper_ = new mapping_module(util::yaml_optional_ref(cfg->yaml_node_, "Mapping"), map_db_, bow_db_, bow_vocab_);
    // global optimization module
    if (bow_db_ && bow_vocab_) {
        global_optimizer_ = new global_optimization_module(map_db_, bow_db_, bow_vocab_, cfg_->yaml_node_, camera_->setup_type_ != camera::setup_type_t::Monocular);
    }

    // preprocessing modules
    const auto preprocessing_params = util::yaml_optional_ref(cfg->yaml_node_, "Preprocessing");
    if (camera_->setup_type_ == camera::setup_type_t::RGBD) {
        depthmap_factor_ = preprocessing_params["depthmap_factor"].as<double>(depthmap_factor_);
        if (depthmap_factor_ < 0.) {
            throw std::runtime_error("depthmap_factor must be greater than 0");
        }
    }
    auto mask_rectangles = util::get_rectangles(preprocessing_params["mask_rectangles"]);

    const auto min_size = preprocessing_params["min_size"].as<unsigned int>(800);
    const auto desc_type_str = preprocessing_params["descriptor_type"].as<std::string>("ORB");
    const auto desc_type = feature::descriptor_type_from_string(desc_type_str);
    extractor_left_ = new feature::orb_extractor(orb_params_, min_size, desc_type, mask_rectangles);
    if (camera_->setup_type_ == camera::setup_type_t::Stereo) {
        extractor_right_ = new feature::orb_extractor(orb_params_, min_size, desc_type, mask_rectangles);
    }

    num_grid_cols_ = preprocessing_params["num_grid_cols"].as<unsigned int>(64);
    num_grid_rows_ = preprocessing_params["num_grid_rows"].as<unsigned int>(48);

    // connect modules each other
    tracker_->set_mapping_module(mapper_);
    mapper_->set_tracking_module(tracker_);
    if (global_optimizer_) {
        tracker_->set_global_optimization_module(global_optimizer_);
        mapper_->set_global_optimization_module(global_optimizer_);
        global_optimizer_->set_tracking_module(tracker_);
        global_optimizer_->set_mapping_module(mapper_);
    }
}

system::~system() {
    global_optimization_thread_.reset(nullptr);
    if (global_optimizer_) {
        delete global_optimizer_;
        global_optimizer_ = nullptr;
    }

    mapping_thread_.reset(nullptr);
    delete mapper_;
    mapper_ = nullptr;

    delete tracker_;
    tracker_ = nullptr;

    delete bow_db_;
    bow_db_ = nullptr;
    delete map_db_;
    map_db_ = nullptr;
    delete cam_db_;
    cam_db_ = nullptr;
    if (bow_vocab_) {
        delete bow_vocab_;
        bow_vocab_ = nullptr;
    }

    delete extractor_left_;
    extractor_left_ = nullptr;
    delete extractor_right_;
    extractor_right_ = nullptr;

    delete orb_params_db_;
    orb_params_db_ = nullptr;

    CV_LOG_DEBUG(&g_log_tag, "DESTRUCT: system");
}

void system::print_info() {
    std::ostringstream message_stream;

    message_stream << std::endl;
    message_stream << "OpenCV SLAM module (based on OpenVSLAM and stella_vslam)," << std::endl;
    message_stream << "Copyright (C) 2019, National Institute of Advanced Industrial Science and Technology (AIST)" << std::endl;
    message_stream << "Copyright (C) 2022, stella-cv contributors" << std::endl;
    message_stream << "Copyright (C) 2026, OpenCV Contrib SLAM module contributors" << std::endl;
    message_stream << "All rights reserved." << std::endl;
    message_stream << std::endl;

    // show configuration
    message_stream << *cfg_ << std::endl;

    CV_LOG_INFO(&g_log_tag, message_stream.str());
}

void system::startup(const bool need_initialize) {
    CV_LOG_INFO(&g_log_tag, "startup SLAM system");
    system_is_running_ = true;

    if (!need_initialize) {
        tracker_->tracking_state_ = tracker_state_t::Lost;
    }

    mapping_thread_ = std::unique_ptr<std::thread>(new std::thread(&cv::slam::mapping_module::run, mapper_));
    if (global_optimizer_) {
        global_optimization_thread_ = std::unique_ptr<std::thread>(new std::thread(&cv::slam::global_optimization_module::run, global_optimizer_));
    }
}

void system::shutdown() {
    // terminate the other threads
    if (global_optimizer_) {
        auto future_mapper_terminate = mapper_->async_terminate();
        auto future_global_optimizer_terminate = global_optimizer_->async_terminate();
        future_mapper_terminate.get();
        future_global_optimizer_terminate.get();
    }
    else {
        auto future_mapper_terminate = mapper_->async_terminate();
        future_mapper_terminate.get();
    }

    // wait until the threads stop
    mapping_thread_->join();
    if (global_optimization_thread_) {
        global_optimization_thread_->join();
    }

    CV_LOG_INFO(&g_log_tag, "shutdown SLAM system");
    system_is_running_ = false;
}

void system::save_frame_trajectory(const std::string& path, const std::string& format) const {
    pause_other_threads();
    io::trajectory_io trajectory_io(map_db_);
    trajectory_io.save_frame_trajectory(path, format);
    resume_other_threads();
}

void system::save_keyframe_trajectory(const std::string& path, const std::string& format) const {
    pause_other_threads();
    io::trajectory_io trajectory_io(map_db_);
    trajectory_io.save_keyframe_trajectory(path, format);
    resume_other_threads();
}

bool system::load_map_database(const std::string& path) const {
    pause_other_threads();
    CV_LOG_DEBUG(&g_log_tag, "load_map_database: " << path);
    bool ok = map_database_io_->load(path, cam_db_, orb_params_db_, map_db_, bow_db_, bow_vocab_);
    auto keyfrms = map_db_->get_all_keyframes();

    for (const auto& keyfrm : keyfrms) {
        keyfrm->frm_obs_.num_grid_cols_ = num_grid_cols_;
        keyfrm->frm_obs_.num_grid_rows_ = num_grid_rows_;
        data::assign_keypoints_to_grid(keyfrm->camera_, keyfrm->frm_obs_.undist_keypts_, keyfrm->frm_obs_.keypt_indices_in_cells_,
                                       keyfrm->frm_obs_.num_grid_cols_, keyfrm->frm_obs_.num_grid_rows_);
    }

    resume_other_threads();
    return ok;
}

bool system::save_map_database(const std::string& path) const {
    pause_other_threads();
    CV_LOG_DEBUG(&g_log_tag, "save_map_database: " << path);
    bool ok = map_database_io_->save(path, cam_db_, orb_params_db_, map_db_);
    resume_other_threads();
    return ok;
}

void system::enable_mapping_module() {
    std::lock_guard<std::mutex> lock(mtx_mapping_);
    if (!system_is_running_) {
        CV_LOG_FATAL(&g_log_tag, "please call system::enable_mapping_module() after system::startup()");
    }
    // resume the mapping module
    mapper_->resume();
}

void system::disable_mapping_module() {
    std::lock_guard<std::mutex> lock(mtx_mapping_);
    if (!system_is_running_) {
        CV_LOG_FATAL(&g_log_tag, "please call system::disable_mapping_module() after system::startup()");
    }
    // pause the mapping module
    auto future_pause = mapper_->async_pause();
    // wait until it stops
    future_pause.get();
}

bool system::mapping_module_is_enabled() const {
    return !mapper_->is_paused();
}

void system::enable_loop_detector() {
    std::lock_guard<std::mutex> lock(mtx_loop_detector_);
    if (global_optimizer_) {
        global_optimizer_->enable_loop_detector();
    }
}

void system::disable_loop_detector() {
    std::lock_guard<std::mutex> lock(mtx_loop_detector_);
    if (global_optimizer_) {
        global_optimizer_->disable_loop_detector();
    }
}

bool system::loop_detector_is_enabled() const {
    return global_optimizer_ && global_optimizer_->loop_detector_is_enabled();
}

bool system::request_loop_closure(int keyfrm1_id, int keyfrm2_id) {
    return global_optimizer_ && global_optimizer_->request_loop_closure(keyfrm1_id, keyfrm2_id);
}

bool system::loop_BA_is_running() const {
    return global_optimizer_ && global_optimizer_->loop_BA_is_running();
}

void system::abort_loop_BA() {
    if (global_optimizer_) {
        global_optimizer_->abort_loop_BA();
    }
}

void system::enable_temporal_mapping() {
    map_db_->set_fixed_keyframe_id_threshold();
}

data::frame system::create_monocular_frame(const cv::Mat& img, const double timestamp, const cv::Mat& mask) {
    // color conversion
    if (!camera_->is_valid_shape(img)) {
        CV_LOG_WARNING(&g_log_tag, "preprocess: Input image size is invalid");
    }
    cv::Mat img_gray = img;
    util::convert_to_grayscale(img_gray, camera_->color_order_);

    data::frame_observation frm_obs;

    // Extract features (Phase 2: support external detector)
    keypts_.clear();
    if (external_feature_detector_) {
        // Use external feature detector (e.g., SIFT, AKAZE, BRISK)
        external_feature_detector_->detectAndCompute(img_gray, mask, keypts_, frm_obs.descriptors_);
    } else {
        // Use internal ORB extractor (default behavior)
        extractor_left_->extract(img_gray, mask, keypts_, frm_obs.descriptors_);
    }
    if (keypts_.empty()) {
        CV_LOG_WARNING(&g_log_tag, "preprocess: cannot extract any keypoints");
    }

    // Undistort keypoints
    camera_->undistort_keypoints(keypts_, frm_obs.undist_keypts_);

    // Convert to bearing vector
    camera_->convert_keypoints_to_bearings(frm_obs.undist_keypts_, frm_obs.bearings_);

    // Assign all the keypoints into grid
    frm_obs.num_grid_cols_ = num_grid_cols_;
    frm_obs.num_grid_rows_ = num_grid_rows_;
    data::assign_keypoints_to_grid(camera_, frm_obs.undist_keypts_, frm_obs.keypt_indices_in_cells_,
                                   frm_obs.num_grid_cols_, frm_obs.num_grid_rows_);

    return data::frame(next_frame_id_++, timestamp, camera_, orb_params_, frm_obs, std::unordered_map<unsigned int, data::marker2d>{});
}

data::frame system::create_stereo_frame(const cv::Mat& left_img, const cv::Mat& right_img, const double timestamp, const cv::Mat& mask) {
    // color conversion
    if (!camera_->is_valid_shape(left_img)) {
        CV_LOG_WARNING(&g_log_tag, "preprocess: Input image size is invalid");
    }
    if (!camera_->is_valid_shape(right_img)) {
        CV_LOG_WARNING(&g_log_tag, "preprocess: Input image size is invalid");
    }
    cv::Mat img_gray = left_img;
    cv::Mat right_img_gray = right_img;
    util::convert_to_grayscale(img_gray, camera_->color_order_);
    util::convert_to_grayscale(right_img_gray, camera_->color_order_);

    data::frame_observation frm_obs;
    //! keypoints of stereo right image
    std::vector<cv::KeyPoint> keypts_right;
    //! ORB descriptors of stereo right image
    cv::Mat descriptors_right;

    // Extract ORB feature
    keypts_.clear();
    std::thread thread_left([this, &frm_obs, &img_gray, &mask]() {
        extractor_left_->extract(img_gray, mask, keypts_, frm_obs.descriptors_);
    });
    std::thread thread_right([this, &frm_obs, &right_img_gray, &mask, &keypts_right, &descriptors_right]() {
        extractor_right_->extract(right_img_gray, mask, keypts_right, descriptors_right);
    });
    thread_left.join();
    thread_right.join();
    if (keypts_.empty()) {
        CV_LOG_WARNING(&g_log_tag, "preprocess: cannot extract any keypoints");
    }

    // Undistort keypoints
    camera_->undistort_keypoints(keypts_, frm_obs.undist_keypts_);

    // Estimate depth with stereo match
    match::stereo stereo_matcher(extractor_left_->image_pyramid_, extractor_right_->image_pyramid_,
                                 keypts_, keypts_right, frm_obs.descriptors_, descriptors_right,
                                 orb_params_->scale_factors_, orb_params_->inv_scale_factors_,
                                 camera_->focal_x_baseline_, camera_->true_baseline_);
    stereo_matcher.compute(frm_obs.stereo_x_right_, frm_obs.depths_);

    // Convert to bearing vector
    camera_->convert_keypoints_to_bearings(frm_obs.undist_keypts_, frm_obs.bearings_);

    // Assign all the keypoints into grid
    frm_obs.num_grid_cols_ = num_grid_cols_;
    frm_obs.num_grid_rows_ = num_grid_rows_;
    data::assign_keypoints_to_grid(camera_, frm_obs.undist_keypts_, frm_obs.keypt_indices_in_cells_,
                                   frm_obs.num_grid_cols_, frm_obs.num_grid_rows_);

    // Detect marker
    std::unordered_map<unsigned int, data::marker2d> markers_2d;

    return data::frame(next_frame_id_++, timestamp, camera_, orb_params_, frm_obs, std::move(markers_2d));
}

data::frame system::create_RGBD_frame(const cv::Mat& rgb_img, const cv::Mat& depthmap, const double timestamp, const cv::Mat& mask) {
    // color and depth scale conversion
    if (!camera_->is_valid_shape(rgb_img)) {
        CV_LOG_WARNING(&g_log_tag, "preprocess: Input image size is invalid");
    }
    if (!camera_->is_valid_shape(depthmap)) {
        CV_LOG_WARNING(&g_log_tag, "preprocess: Input image size is invalid");
    }
    cv::Mat img_gray = rgb_img;
    cv::Mat img_depth = depthmap;
    util::convert_to_grayscale(img_gray, camera_->color_order_);
    util::convert_to_true_depth(img_depth, depthmap_factor_);

    data::frame_observation frm_obs;

    // Extract ORB feature
    keypts_.clear();
    extractor_left_->extract(img_gray, mask, keypts_, frm_obs.descriptors_);
    if (keypts_.empty()) {
        CV_LOG_WARNING(&g_log_tag, "preprocess: cannot extract any keypoints");
    }

    // Undistort keypoints
    camera_->undistort_keypoints(keypts_, frm_obs.undist_keypts_);

    // Calculate disparity from depth
    // Initialize with invalid value
    frm_obs.stereo_x_right_ = std::vector<float>(frm_obs.undist_keypts_.size(), -1);
    frm_obs.depths_ = std::vector<float>(frm_obs.undist_keypts_.size(), -1);

    for (unsigned int idx = 0; idx < frm_obs.undist_keypts_.size(); idx++) {
        const auto& keypt = keypts_.at(idx);
        const auto& undist_keypt = frm_obs.undist_keypts_.at(idx);

        const float x = keypt.pt.x;
        const float y = keypt.pt.y;

        const float depth = img_depth.at<float>(y, x);

        if (depth <= 0) {
            continue;
        }

        frm_obs.depths_.at(idx) = depth;
        frm_obs.stereo_x_right_.at(idx) = undist_keypt.pt.x - camera_->focal_x_baseline_ / depth;
    }

    // Convert to bearing vector
    camera_->convert_keypoints_to_bearings(frm_obs.undist_keypts_, frm_obs.bearings_);

    // Assign all the keypoints into grid
    frm_obs.num_grid_cols_ = num_grid_cols_;
    frm_obs.num_grid_rows_ = num_grid_rows_;
    data::assign_keypoints_to_grid(camera_, frm_obs.undist_keypts_, frm_obs.keypt_indices_in_cells_,
                                   frm_obs.num_grid_cols_, frm_obs.num_grid_rows_);

    // Detect marker
    std::unordered_map<unsigned int, data::marker2d> markers_2d;

    return data::frame(next_frame_id_++, timestamp, camera_, orb_params_, frm_obs, std::move(markers_2d));
}

std::shared_ptr<Mat44_t> system::feed_monocular_frame(const cv::Mat& img, const double timestamp, const cv::Mat& mask) {
    check_reset_request();

    assert(camera_->setup_type_ == camera::setup_type_t::Monocular);
    if (img.empty()) {
        CV_LOG_WARNING(&g_log_tag, "preprocess: empty image");
        return nullptr;
    }
    const auto start = std::chrono::system_clock::now();
    auto frm = create_monocular_frame(img, timestamp, mask);
    const auto end = std::chrono::system_clock::now();
    double extraction_time_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return feed_frame(frm, img, extraction_time_elapsed_ms);
}

std::shared_ptr<Mat44_t> system::feed_stereo_frame(const cv::Mat& left_img, const cv::Mat& right_img, const double timestamp, const cv::Mat& mask) {
    check_reset_request();

    assert(camera_->setup_type_ == camera::setup_type_t::Stereo);
    if (left_img.empty() || right_img.empty()) {
        CV_LOG_WARNING(&g_log_tag, "preprocess: empty image");
        return nullptr;
    }
    const auto start = std::chrono::system_clock::now();
    auto frm = create_stereo_frame(left_img, right_img, timestamp, mask);
    const auto end = std::chrono::system_clock::now();
    double extraction_time_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return feed_frame(frm, left_img, extraction_time_elapsed_ms);
}

std::shared_ptr<Mat44_t> system::feed_RGBD_frame(const cv::Mat& rgb_img, const cv::Mat& depthmap, const double timestamp, const cv::Mat& mask) {
    check_reset_request();

    assert(camera_->setup_type_ == camera::setup_type_t::RGBD);
    if (rgb_img.empty() || depthmap.empty()) {
        CV_LOG_WARNING(&g_log_tag, "preprocess: empty image");
        return nullptr;
    }
    const auto start = std::chrono::system_clock::now();
    auto frm = create_RGBD_frame(rgb_img, depthmap, timestamp, mask);
    const auto end = std::chrono::system_clock::now();
    double extraction_time_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return feed_frame(frm, rgb_img, extraction_time_elapsed_ms);
}

std::shared_ptr<Mat44_t> system::feed_frame(const data::frame& frm, const cv::Mat& img, const double extraction_time_elapsed_ms) {
    const auto start = std::chrono::system_clock::now();

    const auto cam_pose_wc = tracker_->feed_frame(frm);

    const auto end = std::chrono::system_clock::now();
    double tracking_time_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return cam_pose_wc;
}

bool system::relocalize_by_pose(const Mat44_t& cam_pose_wc) {
    const Mat44_t cam_pose_cw = util::converter::inverse_pose(cam_pose_wc);
    return tracker_->request_relocalize_by_pose(cam_pose_cw);
}

bool system::relocalize_by_pose_2d(const Mat44_t& cam_pose_wc, const Vec3_t& normal_vector) {
    const Mat44_t cam_pose_cw = util::converter::inverse_pose(cam_pose_wc);
    return tracker_->request_relocalize_by_pose_2d(cam_pose_cw, normal_vector);
}

void system::pause_tracker() {
    auto future_pause = tracker_->async_pause();
    future_pause.get();
}

bool system::tracker_is_paused() const {
    return tracker_->is_paused();
}

void system::resume_tracker() {
    tracker_->resume();
}

void system::set_allow_initialization(bool allow) {
    if (tracker_) {
        tracker_->set_allow_initialization(allow);
    }
}

void system::request_reset() {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    reset_is_requested_ = true;
}

bool system::reset_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    return reset_is_requested_;
}

void system::request_terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    terminate_is_requested_ = true;
}

bool system::terminate_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return terminate_is_requested_;
}

camera::base* system::get_camera() const {
    return camera_;
}

void system::check_reset_request() {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    if (reset_is_requested_) {
        tracker_->reset();
        reset_is_requested_ = false;
    }
}

void system::pause_other_threads() const {
    // pause the mapping module
    if (mapper_ && !mapper_->is_terminated()) {
        auto future_pause = mapper_->async_pause();
        while (future_pause.wait_for(std::chrono::milliseconds(5)) == std::future_status::timeout) {
            if (mapper_->is_terminated()) {
                break;
            }
        }
    }
    // pause the global optimization module
    if (global_optimizer_ && !global_optimizer_->is_terminated()) {
        auto future_pause = global_optimizer_->async_pause();
        while (future_pause.wait_for(std::chrono::milliseconds(5)) == std::future_status::timeout) {
            if (global_optimizer_->is_terminated()) {
                break;
            }
        }
    }
}

void system::resume_other_threads() const {
    // resume the global optimization module
    if (global_optimizer_) {
        global_optimizer_->resume();
    }
    // resume the mapping module
    if (mapper_) {
        mapper_->resume();
    }
}

void system::set_feature_detector(const cv::Ptr<cv::Feature2D>& detector) {
    external_feature_detector_ = detector;
    if (detector) {
        CV_LOG_INFO(&g_log_tag, "External feature detector set: " << detector->getDefaultName());
        // Disable loop detector for non-ORB features (FBoW vocabulary is ORB-specific)
        if (global_optimizer_) {
            CV_LOG_INFO(&g_log_tag, "Disabling loop detector for non-ORB feature: " << detector->getDefaultName());
            disable_loop_detector();
        }
    } else {
        CV_LOG_INFO(&g_log_tag, "External feature detector cleared, using internal ORB");
        // Re-enable loop detector for ORB
        if (global_optimizer_) {
            CV_LOG_INFO(&g_log_tag, "Re-enabling loop detector for ORB");
            enable_loop_detector();
        }
    }
}

cv::Ptr<cv::Feature2D> system::get_feature_detector() const {
    return external_feature_detector_;
}

}
