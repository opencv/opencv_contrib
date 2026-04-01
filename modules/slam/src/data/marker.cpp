#include "data/marker.hpp"
#include "data/keyframe.hpp"
#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace data {

marker::marker(const eigen_alloc_vector<Vec3_t>& corners_pos_w, unsigned int id, const std::shared_ptr<marker_model::base>& marker_model)
    : corners_pos_w_(corners_pos_w), id_(id), marker_model_(marker_model) {}

void marker::set_corner_pos(const eigen_alloc_vector<Vec3_t>& corner_pos_w) {
    std::lock_guard<std::mutex> lock(mtx_position_);
    corners_pos_w_ = corner_pos_w;
}

std::shared_ptr<marker> marker::from_stmt(sqlite3_stmt* stmt,
                                          std::unordered_map<unsigned int, std::shared_ptr<cv::slam::data::keyframe>>& keyframes) {
    int column_id = 0;
    auto id = sqlite3_column_int64(stmt, column_id);
    column_id++;

    const char* p = reinterpret_cast<const char*>(sqlite3_column_blob(stmt, column_id));
    size_t bs = sqlite3_column_bytes(stmt, column_id);
    column_id++;

    std::vector<double> double_buffer(4 * 3); // 4 corners of 3 coords
    assert(double_buffer.size() * sizeof(double) == bs);
    std::memcpy(double_buffer.data(), p, bs);

    eigen_alloc_vector<Vec3_t> corners_pos_w(4);
    for (size_t i = 0; i < double_buffer.size(); i++)
        corners_pos_w[i / 3](i % 3) = double_buffer[i];

    auto fxd = sqlite3_column_int64(stmt, column_id);
    column_id++;
    bool keep_fixed = (fxd != 0);

    auto num_obs = sqlite3_column_int64(stmt, column_id);
    column_id++;

    p = reinterpret_cast<const char*>(sqlite3_column_blob(stmt, column_id));
    bs = sqlite3_column_bytes(stmt, column_id);
    column_id++;

    std::vector<uint64_t> frame_ids(num_obs);
    assert(frame_ids.size() * sizeof(uint64_t) == bs);
    std::memcpy(frame_ids.data(), p, bs);

    auto init_before_int = sqlite3_column_int64(stmt, column_id);
    column_id++;
    bool initialized_before = (init_before_int != 0);

    std::shared_ptr<marker> mkr = std::make_shared<marker>(corners_pos_w, id, nullptr); // WARNING: don't use marker model yet, will be filled in later
    mkr->keep_fixed_ = keep_fixed;
    mkr->initialized_before_ = initialized_before;

    for (auto frame_id : frame_ids) {
        auto it = keyframes.find(frame_id);
        if (it != keyframes.end()) {
            auto& keyfrm = it->second;
            mkr->observations_.emplace(keyfrm->id_, keyfrm);
            keyfrm->add_marker(mkr);
        }
        else
            CV_LOG_WARNING(&g_log_tag, "Marker " << id << " refers to keyframe " << frame_id << " which cannot be found");
    }

    return mkr;
}

bool marker::bind_to_stmt(sqlite3* db, sqlite3_stmt* stmt) const {
    std::vector<double> corners_pos_w;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            corners_pos_w.push_back(corners_pos_w_[i](j));
        }
    }

    std::vector<uint64_t> observations;
    for (const auto& id_keyfrm : observations_) {
        const auto& keyfrm = id_keyfrm.second;
        observations.push_back(keyfrm->id_);
    }

    int ret = SQLITE_ERROR;
    int column_id = 1;
    ret = sqlite3_bind_int64(stmt, column_id++, id_);
    if (ret == SQLITE_OK) {
        ret = sqlite3_bind_blob(stmt, column_id++, corners_pos_w.data(), corners_pos_w.size() * sizeof(double), SQLITE_TRANSIENT);
    }
    if (ret == SQLITE_OK) {
        ret = sqlite3_bind_int64(stmt, column_id++, keep_fixed_);
    }
    if (ret == SQLITE_OK) {
        ret = sqlite3_bind_int64(stmt, column_id++, observations.size());
    }
    if (ret == SQLITE_OK) {
        ret = sqlite3_bind_blob(stmt, column_id++, observations.data(), observations.size() * sizeof(uint64_t), SQLITE_TRANSIENT);
    }
    if (ret == SQLITE_OK) {
        ret = sqlite3_bind_int64(stmt, column_id++, initialized_before_);
    }
    if (ret != SQLITE_OK) {
        CV_LOG_ERROR(&g_log_tag, "SQLite error (bind): " << sqlite3_errmsg(db));
    }

    return ret == SQLITE_OK;
}

} // namespace data
} // namespace cv::slam
