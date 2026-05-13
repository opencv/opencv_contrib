#include "camera/base.hpp"
#include "data/common.hpp"
#include "data/frame.hpp"
#include "data/keyframe.hpp"
#include "data/landmark.hpp"
#include "data/marker.hpp"
#include "data/camera_database.hpp"
#include "data/orb_params_database.hpp"
#include "data/map_database.hpp"
#include "data/bow_vocabulary.hpp"
#include "util/converter.hpp"
#ifdef USE_SQLITE3
#include "util/sqlite3.hpp"
#endif

#include <opencv2/core/utils/logger.hpp>
#include <nlohmann/json.hpp>
#ifdef USE_SQLITE3
#include <sqlite3.h>
#endif

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace data {

std::mutex map_database::mtx_database_;

map_database::map_database(unsigned int min_num_shared_lms)
    : fixed_keyframe_id_threshold_(0), min_num_shared_lms_(min_num_shared_lms) {
    CV_LOG_DEBUG(&g_log_tag, "CONSTRUCT: data::map_database");
}

map_database::~map_database() {
    clear();
    CV_LOG_DEBUG(&g_log_tag, "DESTRUCT: data::map_database");
}

void map_database::set_fixed_keyframe_id_threshold() {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    fixed_keyframe_id_threshold_ = next_keyframe_id_;
}

unsigned int map_database::get_fixed_keyframe_id_threshold() {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    return fixed_keyframe_id_threshold_;
}

void map_database::add_keyframe(const std::shared_ptr<keyframe>& keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    keyframes_[keyfrm->id_] = keyfrm;
    last_inserted_keyfrm_ = keyfrm;
}

void map_database::erase_keyframe(const std::shared_ptr<keyframe>& keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    keyframes_.erase(keyfrm->id_);
}

std::shared_ptr<keyframe> map_database::get_keyframe(unsigned int id) const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    if (!keyframes_.count(id)) {
        return nullptr;
    }
    return keyframes_.at(id);
}

void map_database::add_landmark(std::shared_ptr<landmark>& lm) {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    landmarks_[lm->id_] = lm;
}

void map_database::erase_landmark(unsigned int id) {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    landmarks_.erase(id);
}

std::shared_ptr<landmark> map_database::get_landmark(unsigned int id) const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    if (!landmarks_.count(id)) {
        return nullptr;
    }
    return landmarks_.at(id);
}

void map_database::add_marker(const std::shared_ptr<marker>& mkr) {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    markers_[mkr->id_] = mkr;
}

void map_database::erase_marker(const std::shared_ptr<marker>& mkr) {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    markers_.erase(mkr->id_);
}

std::shared_ptr<marker> map_database::get_marker(unsigned int id) const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    std::shared_ptr<marker> mkr;
    if (markers_.count(id) == 0) {
        mkr = nullptr;
    }
    else {
        mkr = markers_.at(id);
    }
    return mkr;
}

void map_database::add_spanning_root(std::shared_ptr<keyframe>& keyframe) {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    spanning_roots_.push_back(keyframe);
}

std::vector<std::shared_ptr<keyframe>> map_database::get_spanning_roots() {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    return spanning_roots_;
}

void map_database::set_local_landmarks(const std::vector<std::shared_ptr<landmark>>& local_lms) {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    local_landmarks_ = local_lms;
}

std::vector<std::shared_ptr<landmark>> map_database::get_local_landmarks() const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    return local_landmarks_;
}

std::vector<std::shared_ptr<keyframe>> map_database::get_all_keyframes() const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    std::vector<std::shared_ptr<keyframe>> keyframes;
    keyframes.reserve(keyframes_.size());
    for (const auto& id_keyframe : keyframes_) {
        keyframes.push_back(id_keyframe.second);
    }
    return keyframes;
}

std::vector<std::shared_ptr<keyframe>> map_database::get_close_keyframes_2d(const Mat44_t& pose_cw,
                                                                            const Vec3_t& normal_vector,
                                                                            const double distance_threshold,
                                                                            const double angle_threshold) const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);

    // Close (within given thresholds) keyframes
    std::vector<std::shared_ptr<keyframe>> filtered_keyframes;

    const double cos_angle_threshold = std::cos(angle_threshold);
    Mat44_t pose_wc = util::converter::inverse_pose(pose_cw);

    // Calculate angles and distances between given pose and all keyframes
    Mat33_t M = pose_wc.block<3, 3>(0, 0);
    Vec3_t Mt = pose_wc.block<3, 1>(0, 3);
    for (const auto& id_keyframe : keyframes_) {
        Mat33_t N = id_keyframe.second->get_pose_wc().block<3, 3>(0, 0);
        Vec3_t Nt = id_keyframe.second->get_pose_wc().block<3, 1>(0, 3);
        // Angle between two cameras related to given pose and selected keyframe
        const double cos_angle = ((M * N.transpose()).trace() - 1) / 2;
        // Distance between given pose and selected keyframe
        const double dist = ((Nt - Nt.dot(normal_vector) * normal_vector)
                             - (Mt - Mt.dot(normal_vector) * normal_vector))
                                .norm();
        if (dist < distance_threshold && cos_angle > cos_angle_threshold) {
            filtered_keyframes.push_back(id_keyframe.second);
        }
    }

    return filtered_keyframes;
}

std::vector<std::shared_ptr<keyframe>> map_database::get_close_keyframes(const Mat44_t& pose_cw,
                                                                         const double distance_threshold,
                                                                         const double angle_threshold) const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);

    // Close (within given thresholds) keyframes
    std::vector<std::shared_ptr<keyframe>> filtered_keyframes;

    const double cos_angle_threshold = std::cos(angle_threshold);
    Mat44_t pose_wc = util::converter::inverse_pose(pose_cw);

    // Calculate angles and distances between given pose and all keyframes
    Mat33_t M = pose_wc.block<3, 3>(0, 0);
    Vec3_t Mt = pose_wc.block<3, 1>(0, 3);
    for (const auto& id_keyframe : keyframes_) {
        Mat33_t N = id_keyframe.second->get_pose_wc().block<3, 3>(0, 0);
        Vec3_t Nt = id_keyframe.second->get_pose_wc().block<3, 1>(0, 3);
        // Angle between two cameras related to given pose and selected keyframe
        const double cos_angle = ((M * N.transpose()).trace() - 1) / 2;
        // Distance between given pose and selected keyframe
        const double dist = (Nt - Mt).norm();
        if (dist < distance_threshold && cos_angle > cos_angle_threshold) {
            filtered_keyframes.push_back(id_keyframe.second);
        }
    }

    return filtered_keyframes;
}

unsigned int map_database::get_num_keyframes() const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    return keyframes_.size();
}

std::vector<std::shared_ptr<landmark>> map_database::get_all_landmarks() const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    std::vector<std::shared_ptr<landmark>> landmarks;
    landmarks.reserve(landmarks_.size());
    for (const auto& id_landmark : landmarks_) {
        landmarks.push_back(id_landmark.second);
    }
    return landmarks;
}

std::shared_ptr<keyframe> map_database::get_last_inserted_keyframe() const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    return last_inserted_keyfrm_;
}

std::vector<std::shared_ptr<marker>> map_database::get_all_markers() const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    std::vector<std::shared_ptr<marker>> markers;
    markers.reserve(markers_.size());
    for (const auto& id_marker : markers_) {
        markers.push_back(id_marker.second);
    }
    return markers;
}

unsigned int map_database::get_num_markers() const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    return markers_.size();
}

unsigned int map_database::get_num_landmarks() const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    return landmarks_.size();
}

unsigned int map_database::get_min_num_shared_lms() const {
    return min_num_shared_lms_;
}

void map_database::clear() {
    std::lock_guard<std::mutex> lock(mtx_map_access_);

    landmarks_.clear();
    keyframes_.clear();
    markers_.clear();
    last_inserted_keyfrm_ = nullptr;
    local_landmarks_.clear();
    spanning_roots_.clear();

    frm_stats_.clear();

    next_keyframe_id_ = 0;
    next_landmark_id_ = 0;
    fixed_keyframe_id_threshold_ = 0;

    CV_LOG_INFO(&g_log_tag, "clear map database");
}

void map_database::from_json(camera_database* cam_db, orb_params_database* orb_params_db, bow_vocabulary* bow_vocab,
                             const nlohmann::json& json_keyfrms, const nlohmann::json& json_landmarks) {
    std::lock_guard<std::mutex> lock(mtx_map_access_);

    // When loading the map, leave last_inserted_keyfrm_ as nullptr.
    last_inserted_keyfrm_ = nullptr;
    local_landmarks_.clear();

    // Step 2. Register keyframes
    // If the object does not exist at this step, the corresponding pointer is set as nullptr.
    CV_LOG_INFO(&g_log_tag, "decoding " << json_keyfrms.size() << " keyframes to load");
    for (const auto& json_id_keyfrm : json_keyfrms.items()) {
        const auto keyfrm_id_in_storage = std::stoi(json_id_keyfrm.key());
        assert(0 <= keyfrm_id_in_storage);
        const auto keyfrm_id = keyfrm_id_in_storage + next_keyframe_id_;
        const auto json_keyfrm = json_id_keyfrm.value();

        register_keyframe(cam_db, orb_params_db, bow_vocab, keyfrm_id, json_keyfrm);
    }

    // Step 3. Register 3D landmark point
    // If the object does not exist at this step, the corresponding pointer is set as nullptr.
    CV_LOG_INFO(&g_log_tag, "decoding " << json_landmarks.size() << " landmarks to load");
    for (const auto& json_id_landmark : json_landmarks.items()) {
        const auto landmark_id_in_storage = std::stoi(json_id_landmark.key());
        assert(0 <= landmark_id_in_storage);
        const auto landmark_id = landmark_id_in_storage + next_landmark_id_;
        const auto json_landmark = json_id_landmark.value();

        register_landmark(landmark_id, json_landmark);
    }

    // Step 4. Register graph information
    CV_LOG_INFO(&g_log_tag, "registering essential graph");
    for (const auto& json_id_keyfrm : json_keyfrms.items()) {
        const auto keyfrm_id_in_storage = std::stoi(json_id_keyfrm.key());
        assert(0 <= keyfrm_id_in_storage);
        const auto keyfrm_id = keyfrm_id_in_storage + next_keyframe_id_;
        const auto json_keyfrm = json_id_keyfrm.value();

        register_graph(keyfrm_id, json_keyfrm);
    }

    // Step 5. Register association between keyframs and 3D points
    CV_LOG_INFO(&g_log_tag, "registering keyframe-landmark association");
    for (const auto& json_id_keyfrm : json_keyfrms.items()) {
        const auto keyfrm_id_in_storage = std::stoi(json_id_keyfrm.key());
        assert(0 <= keyfrm_id_in_storage);
        const auto keyfrm_id = keyfrm_id_in_storage + next_keyframe_id_;
        const auto json_keyfrm = json_id_keyfrm.value();

        register_association(keyfrm_id, json_keyfrm);
    }

    // find root node
    std::unordered_set<unsigned int> already_found_root_ids;
    for (const auto& json_id_keyfrm : json_keyfrms.items()) {
        const auto id = std::stoi(json_id_keyfrm.key()) + next_keyframe_id_;
        auto keyfrm = keyframes_.at(id);
        auto root = keyfrm->graph_node_->get_spanning_root();
        if (already_found_root_ids.count(root->id_)) {
            continue;
        }
        already_found_root_ids.insert(root->id_);
        CV_LOG_DEBUG(&g_log_tag, "found root node " << root->id_);
        spanning_roots_.push_back(root);
    }

    // Step 6. Update graph
    CV_LOG_INFO(&g_log_tag, "updating covisibility graph");
    for (const auto& json_id_keyfrm : json_keyfrms.items()) {
        const auto keyfrm_id_in_storage = std::stoi(json_id_keyfrm.key());
        assert(0 <= keyfrm_id_in_storage);
        const auto keyfrm_id = keyfrm_id_in_storage + next_keyframe_id_;

        assert(keyframes_.count(keyfrm_id));
        auto keyfrm = keyframes_.at(keyfrm_id);

        keyfrm->graph_node_->update_connections(min_num_shared_lms_);
        keyfrm->graph_node_->update_covisibility_orders();
    }

    // Step 7. Update geometry
    CV_LOG_INFO(&g_log_tag, "updating landmark geometry");
    for (const auto& json_id_landmark : json_landmarks.items()) {
        const auto landmark_id_in_storage = std::stoi(json_id_landmark.key());
        assert(0 <= landmark_id_in_storage);
        const auto landmark_id = landmark_id_in_storage + next_landmark_id_;

        assert(landmarks_.count(landmark_id));
        const auto& lm = landmarks_.at(landmark_id);

        if (!lm->has_valid_prediction_parameters()) {
            lm->update_mean_normal_and_obs_scale_variance();
        }
        if (!lm->has_representative_descriptor()) {
            lm->compute_descriptor();
        }
    }
}

void map_database::register_keyframe(camera_database* cam_db, orb_params_database* orb_params_db, bow_vocabulary* bow_vocab,
                                     const unsigned int id, const nlohmann::json& json_keyfrm) {
    // Metadata
    const auto timestamp = json_keyfrm.at("ts").get<double>();
    const auto camera_name = json_keyfrm.at("cam").get<std::string>();
    const auto camera = cam_db->get_camera(camera_name);
    assert(camera != nullptr);
    const auto orb_params_name = json_keyfrm.at("orb_params").get<std::string>();
    const auto orb_params = orb_params_db->get_orb_params(orb_params_name);
    assert(orb_params != nullptr);

    // Pose information
    const Mat33_t rot_cw = convert_json_to_rotation(json_keyfrm.at("rot_cw"));
    const Vec3_t trans_cw = convert_json_to_translation(json_keyfrm.at("trans_cw"));
    const auto pose_cw = util::converter::to_eigen_pose(rot_cw, trans_cw);

    // Keypoints information
    const auto num_keypts = json_keyfrm.at("n_keypts").get<unsigned int>();
    // undist_keypts
    const auto json_undist_keypts = json_keyfrm.at("undist_keypts");
    const auto undist_keypts = convert_json_to_keypoints(json_undist_keypts);
    assert(undist_keypts.size() == num_keypts);
    // bearings
    auto bearings = eigen_alloc_vector<Vec3_t>();
    camera->convert_keypoints_to_bearings(undist_keypts, bearings);
    assert(bearings.size() == num_keypts);
    // stereo_x_right
    const auto stereo_x_right = json_keyfrm.at("x_rights").get<std::vector<float>>();
    // depths
    const auto depths = json_keyfrm.at("depths").get<std::vector<float>>();
    // descriptors
    const auto json_descriptors = json_keyfrm.at("descs");
    const auto descriptors = convert_json_to_descriptors(json_descriptors);
    assert(descriptors.rows == static_cast<int>(num_keypts));

    // Construct a new object
    data::bow_vector bow_vec;
    data::bow_feature_vector bow_feat_vec;
    // Construct frame_observation
    frame_observation frm_obs{descriptors, undist_keypts, bearings, stereo_x_right, depths};
    // Compute BoW
    if (bow_vocab) {
        data::bow_vocabulary_util::compute_bow(bow_vocab, descriptors, bow_vec, bow_feat_vec);
    }
    auto keyfrm = data::keyframe::make_keyframe(
        id, timestamp, pose_cw, camera, orb_params,
        frm_obs, bow_vec, bow_feat_vec);

    // Append to map database
    assert(!keyframes_.count(id));
    keyframes_[keyfrm->id_] = keyfrm;
}

void map_database::register_landmark(const unsigned int id, const nlohmann::json& json_landmark) {
    const auto first_keyfrm_id = json_landmark.at("1st_keyfrm").get<int>() + next_keyframe_id_;
    const auto pos_w = Vec3_t(json_landmark.at("pos_w").get<std::vector<Vec3_t::value_type>>().data());
    const auto ref_keyfrm_id = json_landmark.at("ref_keyfrm").get<int>() + next_keyframe_id_;
    const auto ref_keyfrm = keyframes_.at(ref_keyfrm_id);
    const auto num_visible = json_landmark.at("n_vis").get<unsigned int>();
    const auto num_found = json_landmark.at("n_fnd").get<unsigned int>();

    auto lm = std::make_shared<data::landmark>(
        id, first_keyfrm_id, pos_w, ref_keyfrm,
        num_visible, num_found);
    assert(!landmarks_.count(id));
    landmarks_[lm->id_] = lm;
}

void map_database::register_graph(const unsigned int id, const nlohmann::json& json_keyfrm) {
    // Graph information
    const auto spanning_parent_id = json_keyfrm.at("span_parent").get<int>();
    const auto spanning_children_ids = json_keyfrm.at("span_children").get<std::vector<int>>();
    const auto loop_edge_ids = json_keyfrm.at("loop_edges").get<std::vector<int>>();

    assert(keyframes_.count(id));
    assert(spanning_parent_id == -1 || keyframes_.count(spanning_parent_id + next_keyframe_id_));
    keyframes_.at(id)->graph_node_->set_spanning_parent((spanning_parent_id == -1) ? nullptr : keyframes_.at(spanning_parent_id + next_keyframe_id_));
    for (const auto spanning_child_id : spanning_children_ids) {
        assert(keyframes_.count(spanning_child_id));
        keyframes_.at(id)->graph_node_->add_spanning_child(keyframes_.at(spanning_child_id + next_keyframe_id_));
    }
    for (const auto loop_edge_id : loop_edge_ids) {
        assert(keyframes_.count(loop_edge_id));
        keyframes_.at(id)->graph_node_->add_loop_edge(keyframes_.at(loop_edge_id + next_keyframe_id_));
    }
}

void map_database::register_association(const unsigned int keyfrm_id, const nlohmann::json& json_keyfrm) {
    // Key points information
    const auto num_keypts = json_keyfrm.at("n_keypts").get<unsigned int>();
    const auto landmark_ids = json_keyfrm.at("lm_ids").get<std::vector<int>>();
    assert(landmark_ids.size() == num_keypts);

    assert(keyframes_.count(keyfrm_id));
    auto keyfrm = keyframes_.at(keyfrm_id);
    for (unsigned int idx = 0; idx < num_keypts; ++idx) {
        auto lm_id = landmark_ids.at(idx);
        if (lm_id < 0) {
            continue;
        }
        lm_id += next_landmark_id_;
        if (!landmarks_.count(lm_id)) {
            CV_LOG_WARNING(&g_log_tag, "landmark " << lm_id << ": not found in the database");
            continue;
        }

        landmarks_.at(lm_id)->connect_to_keyframe(keyfrm, idx);
    }
}

void map_database::to_json(nlohmann::json& json_keyfrms, nlohmann::json& json_landmarks) const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);

    // Save each keyframe as json
    CV_LOG_INFO(&g_log_tag, "encoding " << keyframes_.size() << " keyframes to store");
    std::map<std::string, nlohmann::json> keyfrms;
    for (const auto& id_keyfrm : keyframes_) {
        const auto id = id_keyfrm.first;
        const auto keyfrm = id_keyfrm.second;
        assert(keyfrm);
        assert(id == keyfrm->id_);
        assert(!keyfrm->will_be_erased());
        keyfrm->graph_node_->update_connections(min_num_shared_lms_);
        assert(!keyfrms.count(std::to_string(id)));
        keyfrms[std::to_string(id)] = keyfrm->to_json();
    }
    json_keyfrms = keyfrms;

    // Save each 3D point as json
    CV_LOG_INFO(&g_log_tag, "encoding " << landmarks_.size() << " landmarks to store");
    std::map<std::string, nlohmann::json> landmarks;
    for (const auto& id_lm : landmarks_) {
        const auto id = id_lm.first;
        const auto& lm = id_lm.second;
        assert(lm);
        assert(id == lm->id_);
        assert(!lm->will_be_erased());
        assert(!landmarks.count(std::to_string(id)));
        landmarks[std::to_string(id)] = lm->to_json();
    }
    json_landmarks = landmarks;
}

#ifdef USE_SQLITE3
bool map_database::from_db(sqlite3* db,
                           camera_database* cam_db,
                           orb_params_database* orb_params_db,
                           bow_vocabulary* bow_vocab) {
    std::lock_guard<std::mutex> lock(mtx_map_access_);

    // When loading the map, leave last_inserted_keyfrm_ as nullptr.
    last_inserted_keyfrm_ = nullptr;
    local_landmarks_.clear();

    // Step 2. load data from database
    bool ok = load_keyframes_from_db(db, "keyframes", cam_db, orb_params_db, bow_vocab);
    if (!ok) {
        return false;
    }
    ok = load_landmarks_from_db(db, "landmarks");
    if (!ok) {
        return false;
    }
    ok = load_associations_from_db(db, "associations");
    if (!ok) {
        return false;
    }
    bool have_markers = load_markers_from_db(db, "markers");
    if (!have_markers) {
        CV_LOG_WARNING(&g_log_tag, "no such table: markers");
    }

    // find root node
    std::unordered_set<unsigned int> already_found_root_ids;
    for (const auto& root : spanning_roots_) {
        already_found_root_ids.insert(root->id_);
    }
    for (const auto& id_keyfrm : keyframes_) {
        const auto keyfrm = id_keyfrm.second;
        auto root = keyfrm->graph_node_->get_spanning_root();
        if (already_found_root_ids.count(root->id_)) {
            continue;
        }
        already_found_root_ids.insert(root->id_);
        CV_LOG_DEBUG(&g_log_tag, "found root node " << root->id_);
        spanning_roots_.push_back(root);
    }

    CV_LOG_INFO(&g_log_tag, "updating covisibility graph");
    for (const auto& id_keyfrm : keyframes_) {
        const auto keyfrm = id_keyfrm.second;

        keyfrm->graph_node_->update_connections(min_num_shared_lms_);
        keyfrm->graph_node_->update_covisibility_orders();
    }

    CV_LOG_INFO(&g_log_tag, "updating landmark geometry");
    for (const auto& id_landmark : landmarks_) {
        const auto lm = id_landmark.second;

        if (!lm->has_valid_prediction_parameters()) {
            lm->update_mean_normal_and_obs_scale_variance();
        }
        if (!lm->has_representative_descriptor()) {
            lm->compute_descriptor();
        }
    }
    return ok;
}

bool map_database::load_keyframes_from_db(sqlite3* db,
                                          const std::string& table_name,
                                          camera_database* cam_db,
                                          orb_params_database* orb_params_db,
                                          bow_vocabulary* bow_vocab) {
    sqlite3_stmt* stmt = util::sqlite3_util::create_select_stmt(db, table_name);
    if (!stmt) {
        return false;
    }

    int ret = SQLITE_ERROR;
    while ((ret = sqlite3_step(stmt)) == SQLITE_ROW) {
        auto keyfrm = data::keyframe::from_stmt(stmt, cam_db, orb_params_db, bow_vocab, next_keyframe_id_);
        // Append to map database
        assert(!keyframes_.count(keyfrm->id_));
        keyframes_[keyfrm->id_] = keyfrm;
    }

    sqlite3_finalize(stmt);
    return ret == SQLITE_DONE;
}

bool map_database::load_landmarks_from_db(sqlite3* db, const std::string& table_name) {
    sqlite3_stmt* stmt = util::sqlite3_util::create_select_stmt(db, table_name);
    if (!stmt) {
        return false;
    }

    int ret = SQLITE_ERROR;
    while ((ret = sqlite3_step(stmt)) == SQLITE_ROW) {
        auto lm = data::landmark::from_stmt(stmt, keyframes_, next_landmark_id_, next_keyframe_id_);
        assert(!landmarks_.count(lm->id_));
        landmarks_[lm->id_] = lm;
    }
    sqlite3_finalize(stmt);
    return ret == SQLITE_DONE;
}

void map_database::load_association_from_stmt(sqlite3_stmt* stmt) {
    const char* p;
    int column_id = 0;
    auto keyfrm_id = sqlite3_column_int64(stmt, column_id);
    assert(keyframes_.count(keyfrm_id));
    column_id++;
    std::vector<int> lm_ids(keyframes_.at(keyfrm_id)->frm_obs_.undist_keypts_.size(), -1);
    p = reinterpret_cast<const char*>(sqlite3_column_blob(stmt, column_id));
    std::memcpy(lm_ids.data(), p, sqlite3_column_bytes(stmt, column_id));
    column_id++;
    auto spanning_parent_id = sqlite3_column_int64(stmt, column_id);
    column_id++;
    auto n_spanning_children = sqlite3_column_int64(stmt, column_id);
    column_id++;
    std::vector<int> spanning_children_ids(n_spanning_children);
    p = reinterpret_cast<const char*>(sqlite3_column_blob(stmt, column_id));
    std::memcpy(spanning_children_ids.data(), p, sqlite3_column_bytes(stmt, column_id));
    column_id++;
    auto n_loop_edges = sqlite3_column_int64(stmt, column_id);
    column_id++;
    std::vector<int> loop_edge_ids(n_loop_edges);
    p = reinterpret_cast<const char*>(sqlite3_column_blob(stmt, column_id));
    std::memcpy(loop_edge_ids.data(), p, sqlite3_column_bytes(stmt, column_id));
    column_id++;

    assert(spanning_parent_id == -1 || keyframes_.count(spanning_parent_id + next_keyframe_id_));
    keyframes_.at(keyfrm_id)->graph_node_->set_spanning_parent((spanning_parent_id == -1LL) ? nullptr : keyframes_.at(spanning_parent_id + next_keyframe_id_));
    for (const auto spanning_child_id : spanning_children_ids) {
        assert(keyframes_.count(spanning_child_id + next_keyframe_id_));
        keyframes_.at(keyfrm_id)->graph_node_->add_spanning_child(keyframes_.at(spanning_child_id + next_keyframe_id_));
    }
    for (const auto loop_edge_id : loop_edge_ids) {
        assert(keyframes_.count(loop_edge_id + next_keyframe_id_));
        keyframes_.at(keyfrm_id)->graph_node_->add_loop_edge(keyframes_.at(loop_edge_id + next_keyframe_id_));
    }

    assert(keyframes_.count(keyfrm_id));
    auto keyfrm = keyframes_.at(keyfrm_id);
    for (unsigned int idx = 0; idx < lm_ids.size(); ++idx) {
        auto lm_id = lm_ids.at(idx);
        if (lm_id < 0) {
            continue;
        }
        lm_id += next_landmark_id_;
        if (!landmarks_.count(lm_id)) {
            CV_LOG_WARNING(&g_log_tag, "landmark " << lm_id << ": not found in the database");
            continue;
        }

        landmarks_.at(lm_id)->connect_to_keyframe(keyfrm, idx);
    }
}

bool map_database::load_associations_from_db(sqlite3* db, const std::string& table_name) {
    sqlite3_stmt* stmt = util::sqlite3_util::create_select_stmt(db, table_name);
    if (!stmt) {
        return false;
    }

    int ret = SQLITE_ERROR;
    while ((ret = sqlite3_step(stmt)) == SQLITE_ROW) {
        load_association_from_stmt(stmt);
    }
    sqlite3_finalize(stmt);
    return ret == SQLITE_DONE;
}

bool map_database::to_db(sqlite3* db) const {
    std::lock_guard<std::mutex> lock(mtx_map_access_);
    for (const auto& id_keyfrm : keyframes_) {
        const auto keyfrm = id_keyfrm.second;
        assert(keyfrm);
        assert(!keyfrm->will_be_erased());
        keyfrm->graph_node_->update_connections(min_num_shared_lms_);
    }

    bool ok = util::sqlite3_util::drop_table(db, "keyframes");
    ok = ok && util::sqlite3_util::drop_table(db, "landmarks");
    ok = ok && util::sqlite3_util::drop_table(db, "associations");
    ok = ok && util::sqlite3_util::drop_table(db, "markers");
    ok = ok && save_keyframes_to_db(db, "keyframes");
    ok = ok && save_landmarks_to_db(db, "landmarks");
    ok = ok && save_associations_to_db(db, "associations");
    ok = ok && save_markers_to_db(db, "markers");
    return ok;
}

bool map_database::save_keyframes_to_db(sqlite3* db, const std::string& table_name) const {
    const auto columns = data::keyframe::columns();
    bool ok = util::sqlite3_util::create_table(db, table_name, columns);
    ok = ok && util::sqlite3_util::begin(db);
    if (!ok) {
        return false;
    }
    sqlite3_stmt* stmt = util::sqlite3_util::create_insert_stmt(db, table_name, columns);
    if (!stmt) {
        return false;
    }
    for (const auto& id_keyfrm : keyframes_) {
        const auto keyfrm = id_keyfrm.second;
        assert(keyfrm);
        assert(!keyfrm->will_be_erased());
        ok = keyfrm->bind_to_stmt(db, stmt);
        ok = ok && util::sqlite3_util::next(db, stmt);
        if (!ok) {
            return false;
        }
    }
    sqlite3_finalize(stmt);
    return util::sqlite3_util::commit(db);
}

bool map_database::save_landmarks_to_db(sqlite3* db, const std::string& table_name) const {
    const auto columns = data::landmark::columns();
    bool ok = util::sqlite3_util::create_table(db, table_name, columns);
    ok = ok && util::sqlite3_util::begin(db);
    if (!ok) {
        return false;
    }
    sqlite3_stmt* stmt = util::sqlite3_util::create_insert_stmt(db, table_name, columns);
    if (!stmt) {
        return false;
    }
    for (const auto& id_landmark : landmarks_) {
        const auto lm = id_landmark.second;
        assert(lm);
        assert(!lm->will_be_erased());
        bool ok = lm->bind_to_stmt(db, stmt);
        ok = ok && util::sqlite3_util::next(db, stmt);
        if (!ok) {
            return false;
        }
    }
    sqlite3_finalize(stmt);
    return util::sqlite3_util::commit(db);
}

bool map_database::bind_association_to_stmt(sqlite3_stmt* stmt,
                                            const std::shared_ptr<keyframe>& keyfrm) const {
    int ret = SQLITE_ERROR;
    int column_id = 1;
    ret = sqlite3_bind_int64(stmt, column_id++, keyfrm->id_);
    if (ret == SQLITE_OK) {
        // extract landmark IDs
        auto lms = keyfrm->get_landmarks();
        std::vector<int> lm_ids(lms.size(), -1);
        for (unsigned int i = 0; i < lms.size(); ++i) {
            if (lms.at(i) && !lms.at(i)->will_be_erased()) {
                lm_ids.at(i) = lms.at(i)->id_;
            }
        }
        ret = sqlite3_bind_blob(stmt, column_id++, lm_ids.data(), lm_ids.size() * sizeof(decltype(lm_ids)::value_type), SQLITE_TRANSIENT);
    }
    if (ret == SQLITE_OK) {
        const auto& spanning_parent = keyfrm->graph_node_->get_spanning_parent();
        ret = sqlite3_bind_int64(stmt, column_id++,
                                 spanning_parent ? static_cast<int64_t>(spanning_parent->id_) : -1LL);
    }
    if (ret == SQLITE_OK) {
        const auto spanning_children = keyfrm->graph_node_->get_spanning_children();
        ret = sqlite3_bind_int64(stmt, column_id++, spanning_children.size());
    }
    if (ret == SQLITE_OK) {
        // extract spanning tree children
        const auto spanning_children = keyfrm->graph_node_->get_spanning_children();
        std::vector<int> spanning_child_ids;
        spanning_child_ids.reserve(spanning_children.size());
        for (const auto& spanning_child : spanning_children) {
            spanning_child_ids.push_back(spanning_child->id_);
        }

        ret = sqlite3_bind_blob(stmt, column_id++, spanning_child_ids.data(), spanning_child_ids.size() * sizeof(decltype(spanning_child_ids)::value_type), SQLITE_TRANSIENT);
    }
    if (ret == SQLITE_OK) {
        const auto loop_edges = keyfrm->graph_node_->get_loop_edges();
        ret = sqlite3_bind_int64(stmt, column_id++, loop_edges.size());
    }
    if (ret == SQLITE_OK) {
        // extract loop edges
        const auto loop_edges = keyfrm->graph_node_->get_loop_edges();
        std::vector<int> loop_edge_ids;
        for (const auto& loop_edge : loop_edges) {
            loop_edge_ids.push_back(loop_edge->id_);
        }

        ret = sqlite3_bind_blob(stmt, column_id++, loop_edge_ids.data(), loop_edge_ids.size() * sizeof(decltype(loop_edge_ids)::value_type), SQLITE_TRANSIENT);
    }
    return ret == SQLITE_OK;
}

bool map_database::save_associations_to_db(sqlite3* db, const std::string& table_name) const {
    const auto columns = association_columns();
    bool ok = util::sqlite3_util::create_table(db, table_name, columns);
    ok = ok && util::sqlite3_util::begin(db);
    if (!ok) {
        return false;
    }
    sqlite3_stmt* stmt = util::sqlite3_util::create_insert_stmt(db, table_name, columns);
    if (!stmt) {
        return false;
    }
    for (const auto& id_keyfrm : keyframes_) {
        const auto keyfrm = id_keyfrm.second;
        assert(keyfrm);
        assert(!keyfrm->will_be_erased());
        ok = bind_association_to_stmt(stmt, keyfrm);
        ok = ok && util::sqlite3_util::next(db, stmt);
        if (!ok) {
            return false;
        }
    }
    sqlite3_finalize(stmt);
    return util::sqlite3_util::commit(db);
}

bool map_database::save_markers_to_db(sqlite3* db, const std::string& table_name) const {
    const auto columns = data::marker::columns();
    bool ok = util::sqlite3_util::create_table(db, table_name, columns);
    ok = ok && util::sqlite3_util::begin(db);
    if (!ok) {
        return false;
    }
    sqlite3_stmt* stmt = util::sqlite3_util::create_insert_stmt(db, table_name, columns);
    if (!stmt) {
        return false;
    }

    for (const auto& id_marker : markers_) {
        const auto mkr = id_marker.second;
        assert(mkr);
        bool ok = mkr->bind_to_stmt(db, stmt);
        ok = ok && util::sqlite3_util::next(db, stmt);
        if (!ok) {
            return false;
        }
    }

    sqlite3_finalize(stmt);
    return util::sqlite3_util::commit(db);
}

bool map_database::load_markers_from_db(sqlite3* db, const std::string& table_name) {
    sqlite3_stmt* stmt = util::sqlite3_util::create_select_stmt(db, table_name);
    if (!stmt) {
        return false;
    }

    int ret = SQLITE_ERROR;
    while ((ret = sqlite3_step(stmt)) == SQLITE_ROW) {
        auto mkr = data::marker::from_stmt(stmt, keyframes_);
        assert(!markers_.count(mkr->id_));
        markers_[mkr->id_] = mkr;
    }
    sqlite3_finalize(stmt);
    return ret == SQLITE_DONE;
}

} // namespace data
} // namespace cv::slam

#endif