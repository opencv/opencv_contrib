#include "data/common.hpp"
#include "data/frame.hpp"
#include "data/keyframe.hpp"
#include "data/landmark.hpp"
#include "data/marker.hpp"
#include "data/marker2d.hpp"
#include "data/map_database.hpp"
#include "data/bow_database.hpp"
#include "data/camera_database.hpp"
#include "data/orb_params_database.hpp"
#include "data/bow_vocabulary.hpp"
#include "feature/orb_params.hpp"
#include "util/converter.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/core/utils/logger.hpp>

namespace {
using namespace cv::slam;
using namespace cv::slam::data;
inline std::vector<double> markers2d_to_blob(const std::unordered_map<unsigned int, marker2d>& markers_2d) {
    std::vector<double> blob;
    for (auto& m_it : markers_2d) {
        const auto& m2d = m_it.second;

        // undist_corners_
        assert(m2d.undist_corners_.size() == 4);
        for (auto& c : m2d.undist_corners_) {
            blob.push_back(c.x);
            blob.push_back(c.y);
        }

        // bearings_
        assert(m2d.bearings_.size() == 4);
        for (auto& b : m2d.bearings_) {
            blob.push_back(b.x());
            blob.push_back(b.y());
            blob.push_back(b.z());
        }

        // rot_cm_
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                blob.push_back(m2d.rot_cm_(i, j));
            }
        }

        // trans_cm_
        blob.push_back(m2d.trans_cm_.x());
        blob.push_back(m2d.trans_cm_.y());
        blob.push_back(m2d.trans_cm_.z());

        // id_
        blob.push_back(m2d.id_);
    }
    return blob;
}

std::unordered_map<unsigned int, marker2d>
markers2d_from_blob(size_t amount,
                    const std::vector<double>& blob) {
    std::unordered_map<unsigned int, marker2d> markers_2d;
    size_t offset = 0;
    for (size_t i = 0; i < amount; i++) {
        std::vector<cv::Point2f> undist_corners;
        for (size_t i = 0; i < 4; i++) {
            float x = (float)blob[offset++];
            float y = (float)blob[offset++];
            undist_corners.push_back(cv::Point2f(x, y));
        }

        eigen_alloc_vector<Vec3_t> bearings;
        for (size_t i = 0; i < 4; i++) {
            double x = blob[offset++];
            double y = blob[offset++];
            double z = blob[offset++];
            bearings.push_back(Vec3_t(x, y, z));
        }

        Mat33_t rot_cm;
        for (size_t i = 0; i < 3; i++)
            for (size_t j = 0; j < 3; j++)
                rot_cm(i, j) = blob[offset++];

        Vec3_t trans_cm;

        trans_cm(0) = blob[offset++];
        trans_cm(1) = blob[offset++];
        trans_cm(2) = blob[offset++];

        unsigned int id = (unsigned int)blob[offset++];

        marker2d m2d(undist_corners, bearings, rot_cm, trans_cm, id, nullptr, {}); // WARNING: marker_model is set to nullptr! Don't use yet!
        assert(markers_2d.find(m2d.id_) == markers_2d.end());
        markers_2d.emplace(m2d.id_, m2d);
    }
    return markers_2d;
}
} // namespace

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace data {

keyframe::keyframe(unsigned int id, const frame& frm)
    : id_(id), timestamp_(frm.timestamp_),
      camera_(frm.camera_), orb_params_(frm.orb_params_),
      frm_obs_(frm.frm_obs_),
      bow_vec_(frm.bow_vec_), bow_feat_vec_(frm.bow_feat_vec_),
      markers_2d_(frm.markers_2d_),
      landmarks_(frm.get_landmarks()) {
    // set pose parameters (pose_wc_, trans_wc_) using frm.pose_cw_
    set_pose_cw(frm.get_pose_cw());
}

keyframe::keyframe(const unsigned int id, const double timestamp,
                   const Mat44_t& pose_cw, camera::base* camera,
                   const feature::orb_params* orb_params, const frame_observation& frm_obs,
                   const bow_vector& bow_vec, const bow_feature_vector& bow_feat_vec,
                   std::unordered_map<unsigned int, marker2d> markers_2d)
    : id_(id),
      timestamp_(timestamp), camera_(camera),
      orb_params_(orb_params), frm_obs_(frm_obs),
      bow_vec_(bow_vec), bow_feat_vec_(bow_feat_vec),
      markers_2d_(markers_2d),
      landmarks_(std::vector<std::shared_ptr<landmark>>(frm_obs_.undist_keypts_.size(), nullptr)) {
    // set pose parameters (pose_wc_, trans_wc_) using pose_cw_
    set_pose_cw(pose_cw);

    // The following process needs to take place:
    //   should set the pointers of landmarks_ using add_landmark()
    //   should set connections using graph_node->update_connections()
    //   should set spanning_parent_ using graph_node->set_spanning_parent()
    //   should set spanning_children_ using graph_node->add_spanning_child()
    //   should set loop_edges_ using graph_node->add_loop_edge()
}

keyframe::~keyframe() {
    CV_LOG_DEBUG(&g_log_tag, "keyframe::~keyframe: " << id_);
}

std::shared_ptr<keyframe> keyframe::make_keyframe(unsigned int id, const frame& frm) {
    auto ptr = std::allocate_shared<keyframe>(Eigen::aligned_allocator<keyframe>(), id, frm);
    // covisibility graph node (connections is not assigned yet)
    ptr->graph_node_ = cv::slam::make_unique<graph_node>(ptr);
    return ptr;
}

std::shared_ptr<keyframe> keyframe::make_keyframe(
    const unsigned int id, const double timestamp,
    const Mat44_t& pose_cw, camera::base* camera,
    const feature::orb_params* orb_params, const frame_observation& frm_obs,
    const bow_vector& bow_vec, const bow_feature_vector& bow_feat_vec,
    std::unordered_map<unsigned int, marker2d> markers_2d) {
    auto ptr = std::allocate_shared<keyframe>(
        Eigen::aligned_allocator<keyframe>(),
        id, timestamp,
        pose_cw, camera, orb_params,
        frm_obs, bow_vec, bow_feat_vec, markers_2d);
    // covisibility graph node (connections is not assigned yet)
    ptr->graph_node_ = cv::slam::make_unique<graph_node>(ptr);
    return ptr;
}

std::shared_ptr<keyframe> keyframe::from_stmt(sqlite3_stmt* stmt,
                                              camera_database* cam_db,
                                              orb_params_database* orb_params_db,
                                              bow_vocabulary* bow_vocab,
                                              unsigned int next_keyframe_id) {
    const char* p;
    int column_id = 0;
    auto id = sqlite3_column_int64(stmt, column_id);
    column_id++;
    // NOTE: src_frm_id is removed
    column_id++;
    auto timestamp = sqlite3_column_double(stmt, column_id);
    column_id++;
    p = reinterpret_cast<const char*>(sqlite3_column_blob(stmt, column_id));
    std::string camera_name(p, p + sqlite3_column_bytes(stmt, column_id));
    const auto camera = cam_db->get_camera(camera_name);
    assert(camera != nullptr);
    column_id++;
    p = reinterpret_cast<const char*>(sqlite3_column_blob(stmt, column_id));
    std::string orb_params_name(p, p + sqlite3_column_bytes(stmt, column_id));
    const auto orb_params = orb_params_db->get_orb_params(orb_params_name);
    assert(orb_params != nullptr);
    column_id++;
    Mat44_t pose_cw;
    p = reinterpret_cast<const char*>(sqlite3_column_blob(stmt, column_id));
    std::memcpy(pose_cw.data(), p, sqlite3_column_bytes(stmt, column_id));
    column_id++;
    unsigned int num_keypts = sqlite3_column_int64(stmt, column_id);
    column_id++;
    std::vector<cv::KeyPoint> undist_keypts(num_keypts);
    p = reinterpret_cast<const char*>(sqlite3_column_blob(stmt, column_id));
    std::memcpy(undist_keypts.data(), p, sqlite3_column_bytes(stmt, column_id));
    column_id++;
    std::vector<float> stereo_x_right;
    std::vector<float> depths;
    if (camera->setup_type_ != cv::slam::camera::setup_type_t::Monocular) {
        stereo_x_right.resize(num_keypts);
        p = reinterpret_cast<const char*>(sqlite3_column_blob(stmt, column_id));
        std::memcpy(stereo_x_right.data(), p, sqlite3_column_bytes(stmt, column_id));
        column_id++;
        depths.resize(num_keypts);
        p = reinterpret_cast<const char*>(sqlite3_column_blob(stmt, column_id));
        std::memcpy(depths.data(), p, sqlite3_column_bytes(stmt, column_id));
        column_id++;
    }
    else {
        column_id += 2;
    }
    cv::Mat descriptors(num_keypts, 32, CV_8U);
    p = reinterpret_cast<const char*>(sqlite3_column_blob(stmt, column_id));
    std::memcpy(descriptors.data, p, sqlite3_column_bytes(stmt, column_id));
    column_id++;

    std::unordered_map<unsigned int, marker2d> markers_2d;
    if (sqlite3_column_count(stmt) > column_id) {
        auto num_markers = sqlite3_column_int64(stmt, column_id);
        column_id++;

        std::vector<double> markers_blob(num_markers * MARKERS2D_BLOB_NUM_DOUBLES);

        p = reinterpret_cast<const char*>(sqlite3_column_blob(stmt, column_id));
        assert(static_cast<size_t>(sqlite3_column_bytes(stmt, column_id)) == markers_blob.size() * sizeof(double));
        std::memcpy(markers_blob.data(), p, sqlite3_column_bytes(stmt, column_id));
        column_id++;

        assert(markers_blob.size() == MARKERS2D_BLOB_NUM_DOUBLES * num_markers);
        markers_2d = markers2d_from_blob(num_markers, markers_blob);
    }

    auto bearings = eigen_alloc_vector<Vec3_t>();
    camera->convert_keypoints_to_bearings(undist_keypts, bearings);
    assert(bearings.size() == num_keypts);

    // Construct a new object
    data::bow_vector bow_vec;
    data::bow_feature_vector bow_feat_vec;
    // Construct frame_observation
    frame_observation frm_obs{descriptors, undist_keypts, bearings, stereo_x_right, depths};
    // Compute BoW
    if (bow_vocab) {
        data::bow_vocabulary_util::compute_bow(bow_vocab, descriptors, bow_vec, bow_feat_vec);
    }
    // NOTE: 3D marker info will be filled in later based on loaded markers
    auto keyfrm = data::keyframe::make_keyframe(
        id + next_keyframe_id, timestamp, pose_cw, camera, orb_params,
        frm_obs, bow_vec, bow_feat_vec, markers_2d);

    return keyfrm;
}

nlohmann::json keyframe::to_json() const {
    // extract landmark IDs
    std::vector<int> landmark_ids(landmarks_.size(), -1);
    for (unsigned int i = 0; i < landmark_ids.size(); ++i) {
        if (landmarks_.at(i) && !landmarks_.at(i)->will_be_erased()) {
            landmark_ids.at(i) = landmarks_.at(i)->id_;
        }
    }

    // extract spanning tree parent
    const auto& spanning_parent = graph_node_->get_spanning_parent();

    // extract spanning tree children
    const auto spanning_children = graph_node_->get_spanning_children();
    std::vector<int> spanning_child_ids;
    spanning_child_ids.reserve(spanning_children.size());
    for (const auto& spanning_child : spanning_children) {
        spanning_child_ids.push_back(spanning_child->id_);
    }

    // extract loop edges
    const auto loop_edges = graph_node_->get_loop_edges();
    std::vector<int> loop_edge_ids;
    for (const auto& loop_edge : loop_edges) {
        loop_edge_ids.push_back(loop_edge->id_);
    }

    // TODO: msgpack format does not yet support markers save/load

    return {{"ts", timestamp_},
            {"cam", camera_->name_},
            {"orb_params", orb_params_->name_},
            // camera pose
            {"rot_cw", convert_rotation_to_json(pose_cw_.block<3, 3>(0, 0))},
            {"trans_cw", convert_translation_to_json(pose_cw_.block<3, 1>(0, 3))},
            // features and observations
            {"n_keypts", frm_obs_.undist_keypts_.size()},
            {"undist_keypts", convert_keypoints_to_json(frm_obs_.undist_keypts_)},
            {"x_rights", frm_obs_.stereo_x_right_},
            {"depths", frm_obs_.depths_},
            {"descs", convert_descriptors_to_json(frm_obs_.descriptors_)},
            {"lm_ids", landmark_ids},
            // graph information
            {"span_parent", spanning_parent ? spanning_parent->id_ : -1},
            {"span_children", spanning_child_ids},
            {"loop_edges", loop_edge_ids}};
}

bool keyframe::bind_to_stmt(sqlite3* db, sqlite3_stmt* stmt) const {
    int ret = SQLITE_ERROR;
    int column_id = 1;
    ret = sqlite3_bind_int64(stmt, column_id++, id_);
    // NOTE: src_frm_id is removed
    column_id++;
    ret = sqlite3_bind_double(stmt, column_id++, timestamp_);
    if (ret == SQLITE_OK) {
        const auto& camera_name = camera_->name_;
        ret = sqlite3_bind_blob(stmt, column_id++, camera_name.c_str(), camera_name.size(), SQLITE_TRANSIENT);
    }
    if (ret == SQLITE_OK) {
        const auto& orb_params_name = orb_params_->name_;
        ret = sqlite3_bind_blob(stmt, column_id++, orb_params_name.c_str(), orb_params_name.size(), SQLITE_TRANSIENT);
    }
    if (ret == SQLITE_OK) {
        const Mat44_t pose_cw = get_pose_cw();
        ret = sqlite3_bind_blob(stmt, column_id++, pose_cw.data(), pose_cw.rows() * pose_cw.cols() * sizeof(decltype(pose_cw)::Scalar), SQLITE_TRANSIENT);
    }
    size_t num_keypts = 0;
    if (ret == SQLITE_OK) {
        num_keypts = frm_obs_.undist_keypts_.size();
        ret = sqlite3_bind_int64(stmt, column_id++, num_keypts);
    }
    if (ret == SQLITE_OK) {
        const auto& undist_keypts = frm_obs_.undist_keypts_;
        assert(undist_keypts.size() == num_keypts);
        ret = sqlite3_bind_blob(stmt, column_id++, undist_keypts.data(), undist_keypts.size() * sizeof(std::remove_reference<decltype(undist_keypts)>::type::value_type), SQLITE_TRANSIENT);
    }
    if (ret == SQLITE_OK) {
        const auto& stereo_x_right = frm_obs_.stereo_x_right_;
        ret = sqlite3_bind_blob(stmt, column_id++, stereo_x_right.data(), stereo_x_right.size() * sizeof(std::remove_reference<decltype(stereo_x_right)>::type::value_type), SQLITE_TRANSIENT);
    }
    if (ret == SQLITE_OK) {
        const auto& depths = frm_obs_.depths_;
        ret = sqlite3_bind_blob(stmt, column_id++, depths.data(), depths.size() * sizeof(std::remove_reference<decltype(depths)>::type::value_type), SQLITE_TRANSIENT);
    }
    if (ret == SQLITE_OK) {
        const auto& descriptors = frm_obs_.descriptors_;
        assert(descriptors.dims == 2);
        assert(descriptors.channels() == 1);
        assert(descriptors.cols == 32);
        assert(descriptors.rows > 0 && static_cast<size_t>(descriptors.rows) == num_keypts);
        assert(descriptors.elemSize() == 1);
        ret = sqlite3_bind_blob(stmt, column_id++, descriptors.data, descriptors.total() * descriptors.elemSize(), SQLITE_TRANSIENT);
    }
    size_t num_markers = 0;
    if (ret == SQLITE_OK) {
        num_markers = markers_2d_.size();
        ret = sqlite3_bind_int64(stmt, column_id++, num_markers);
    }

    if (ret == SQLITE_OK) {
        std::vector<double> marker2d_blob = markers2d_to_blob(markers_2d_);
        assert(marker2d_blob.size() == MARKERS2D_BLOB_NUM_DOUBLES * markers_2d_.size());
        ret = sqlite3_bind_blob(stmt, column_id++, marker2d_blob.data(), marker2d_blob.size() * sizeof(double), SQLITE_TRANSIENT);
    }
    if (ret != SQLITE_OK) {
        CV_LOG_ERROR(&g_log_tag, "SQLite error (bind): " << sqlite3_errmsg(db));
    }
    return ret == SQLITE_OK;
}

void keyframe::set_pose_cw(const Mat44_t& pose_cw) {
    std::lock_guard<std::mutex> lock(mtx_pose_);
    pose_cw_ = pose_cw;

    const Mat33_t rot_cw = pose_cw_.block<3, 3>(0, 0);
    const Vec3_t trans_cw = pose_cw_.block<3, 1>(0, 3);
    const Mat33_t rot_wc = rot_cw.transpose();
    trans_wc_ = -rot_wc * trans_cw;

    pose_wc_ = Mat44_t::Identity();
    pose_wc_.block<3, 3>(0, 0) = rot_wc;
    pose_wc_.block<3, 1>(0, 3) = trans_wc_;
}

Mat44_t keyframe::get_pose_cw() const {
    std::lock_guard<std::mutex> lock(mtx_pose_);
    return pose_cw_;
}

Mat44_t keyframe::get_pose_wc() const {
    std::lock_guard<std::mutex> lock(mtx_pose_);
    return pose_wc_;
}

Vec3_t keyframe::get_trans_wc() const {
    std::lock_guard<std::mutex> lock(mtx_pose_);
    return trans_wc_;
}

Mat33_t keyframe::get_rot_cw() const {
    std::lock_guard<std::mutex> lock(mtx_pose_);
    return pose_cw_.block<3, 3>(0, 0);
}

Vec3_t keyframe::get_trans_cw() const {
    std::lock_guard<std::mutex> lock(mtx_pose_);
    return pose_cw_.block<3, 1>(0, 3);
}

bool keyframe::bow_is_available() const {
    return !bow_vec_.empty() && !bow_feat_vec_.empty();
}

void keyframe::compute_bow(bow_vocabulary* bow_vocab) {
    bow_vocabulary_util::compute_bow(bow_vocab, frm_obs_.descriptors_, bow_vec_, bow_feat_vec_);
}

void keyframe::add_landmark(std::shared_ptr<landmark> lm, const unsigned int idx) {
    std::lock_guard<std::mutex> lock(mtx_observations_);
    landmarks_.at(idx) = lm;
}

void keyframe::erase_landmark_with_index(const unsigned int idx) {
    std::lock_guard<std::mutex> lock(mtx_observations_);
    landmarks_.at(idx) = nullptr;
}

void keyframe::erase_landmark(const std::shared_ptr<landmark>& lm) {
    std::lock_guard<std::mutex> lock(mtx_observations_);
    int idx = lm->get_index_in_keyframe(shared_from_this());
    if (0 <= idx) {
        landmarks_.at(static_cast<unsigned int>(idx)) = nullptr;
    }
}

void keyframe::update_landmarks() {
    std::lock_guard<std::mutex> lock(mtx_observations_);
    for (unsigned int idx = 0; idx < landmarks_.size(); ++idx) {
        auto lm = landmarks_.at(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        // update connection
        lm->add_observation(shared_from_this(), idx);
        // update geometry
        lm->update_mean_normal_and_obs_scale_variance();
        lm->compute_descriptor();
    }
}

std::vector<std::shared_ptr<landmark>> keyframe::get_landmarks() const {
    std::lock_guard<std::mutex> lock(mtx_observations_);
    return landmarks_;
}

std::set<std::shared_ptr<landmark>> keyframe::get_valid_landmarks() const {
    std::lock_guard<std::mutex> lock(mtx_observations_);
    std::set<std::shared_ptr<landmark>> valid_landmarks;

    for (const auto& lm : landmarks_) {
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        valid_landmarks.insert(lm);
    }

    return valid_landmarks;
}

unsigned int keyframe::get_num_tracked_landmarks(const unsigned int min_num_obs_thr) const {
    std::lock_guard<std::mutex> lock(mtx_observations_);
    unsigned int num_tracked_lms = 0;

    if (0 < min_num_obs_thr) {
        for (const auto& lm : landmarks_) {
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            if (min_num_obs_thr <= lm->num_observations()) {
                ++num_tracked_lms;
            }
        }
    }
    else {
        for (const auto& lm : landmarks_) {
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            ++num_tracked_lms;
        }
    }

    return num_tracked_lms;
}

std::shared_ptr<landmark>& keyframe::get_landmark(const unsigned int idx) {
    std::lock_guard<std::mutex> lock(mtx_observations_);
    return landmarks_.at(idx);
}

std::vector<unsigned int> keyframe::get_keypoints_in_cell(const float ref_x, const float ref_y, const float margin,
                                                          const int min_level, const int max_level) const {
    return data::get_keypoints_in_cell(camera_, frm_obs_, ref_x, ref_y, margin, min_level, max_level);
}

Vec3_t keyframe::triangulate_stereo(const unsigned int idx) const {
    Mat44_t pose_wc;
    {
        std::lock_guard<std::mutex> lock(mtx_pose_);
        pose_wc = pose_wc_;
    }
    return data::triangulate_stereo(camera_, pose_wc.block<3, 3>(0, 0), pose_wc.block<3, 1>(0, 3), frm_obs_, idx);
}

float keyframe::compute_median_depth(const bool abs) const {
    std::vector<std::shared_ptr<landmark>> landmarks;
    Mat44_t pose_cw;
    {
        std::lock_guard<std::mutex> lock1(mtx_observations_);
        std::lock_guard<std::mutex> lock2(mtx_pose_);
        landmarks = landmarks_;
        pose_cw = pose_cw_;
    }

    std::vector<float> depths;
    depths.reserve(frm_obs_.undist_keypts_.size());
    const Vec3_t rot_cw_z_row = pose_cw.block<1, 3>(2, 0);
    const float trans_cw_z = pose_cw(2, 3);

    for (const auto& lm : landmarks) {
        if (!lm) {
            continue;
        }
        const Vec3_t pos_w = lm->get_pos_in_world();
        const auto pos_c_z = rot_cw_z_row.dot(pos_w) + trans_cw_z;
        depths.push_back(abs ? std::abs(pos_c_z) : pos_c_z);
    }

    std::sort(depths.begin(), depths.end());

    return depths.at((depths.size() - 1) / 2);
}

float keyframe::compute_median_distance() const {
    std::vector<std::shared_ptr<landmark>> landmarks;
    Mat44_t pose_cw;
    {
        std::lock_guard<std::mutex> lock1(mtx_observations_);
        std::lock_guard<std::mutex> lock2(mtx_pose_);
        landmarks = landmarks_;
        pose_cw = pose_cw_;
    }

    std::vector<float> distances;
    distances.reserve(frm_obs_.undist_keypts_.size());
    const Mat33_t rot_cw = pose_cw.block<3, 3>(0, 0);
    const Vec3_t trans_cw = pose_cw.block<3, 1>(0, 3);

    for (const auto& lm : landmarks) {
        if (!lm) {
            continue;
        }
        const Vec3_t pos_w = lm->get_pos_in_world();
        const Vec3_t pos_c = rot_cw * pos_w + trans_cw;
        float distance = pos_c.norm();
        distances.push_back(distance);
    }

    std::sort(distances.begin(), distances.end());

    return distances.at((distances.size() - 1) / 2);
}

bool keyframe::depth_is_available() const {
    return camera_->setup_type_ != camera::setup_type_t::Monocular;
}

void keyframe::add_marker(const std::shared_ptr<marker>& mkr) {
    std::lock_guard<std::mutex> lock(mtx_observations_);
    markers_[mkr->id_] = mkr;
}

std::vector<std::shared_ptr<marker>> keyframe::get_markers() const {
    std::lock_guard<std::mutex> lock(mtx_observations_);
    std::vector<std::shared_ptr<marker>> markers;
    markers.reserve(markers_.size());
    for (const auto& id_marker : markers_) {
        markers.push_back(id_marker.second);
    }
    return markers;
}

void keyframe::set_not_to_be_erased() {
    cannot_be_erased_ = true;
}

void keyframe::set_to_be_erased() {
    if (!graph_node_->has_loop_edge()) {
        cannot_be_erased_ = false;
    }
}

void keyframe::prepare_for_erasing(map_database* map_db, bow_database* bow_db) {
    if (graph_node_->is_spanning_root()) {
        CV_LOG_WARNING(&g_log_tag, "cannot erase the root node: " << id_);
        return;
    }

    // cannot erase if the frag is raised
    if (cannot_be_erased_) {
        return;
    }

    // 1. raise the flag which indicates it has been erased

    CV_LOG_DEBUG(&g_log_tag, "keyframe::prepare_for_erasing " << id_);
    will_be_erased_ = true;

    // 2. remove associations between keypoints and landmarks

    {
        std::lock_guard<std::mutex> lock(mtx_observations_);
        for (const auto& lm : landmarks_) {
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }
            lm->erase_observation(map_db, shared_from_this());
            if (!lm->will_be_erased()) {
                lm->compute_descriptor();
                lm->update_mean_normal_and_obs_scale_variance();
            }
        }
        for (const auto& mkr : markers_) {
            mkr.second->observations_.erase(id_);
        }
    }

    // 3. recover covisibility graph and spanning tree

    // remove covisibility information
    graph_node_->erase_all_connections();
    // recover spanning tree
    graph_node_->recover_spanning_connections();

    // 3. update frame statistics

    map_db->replace_reference_keyframe(shared_from_this(), graph_node_->get_spanning_parent());

    // 4. remove myself from the databased

    map_db->erase_keyframe(shared_from_this());
    if (bow_db) {
        bow_db->erase_keyframe(shared_from_this());
    }
}

bool keyframe::will_be_erased() {
    return will_be_erased_;
}

} // namespace data
} // namespace cv::slam
