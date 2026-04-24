#ifndef SLAM_DATA_MARKER_H
#define SLAM_DATA_MARKER_H

#include "type.hpp"

#include <mutex>
#include <Eigen/Core>
#include <sqlite3.h>

namespace cv::slam {
namespace marker_model {
class base;
}

namespace data {

class keyframe;

class marker {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    //! constructor
    marker(const eigen_alloc_vector<Vec3_t>& corners_pos_w, unsigned int id, const std::shared_ptr<marker_model::base>& marker_model);

    void set_corner_pos(const eigen_alloc_vector<Vec3_t>& corner_pos_w);

    // Factory method to load marker from db
    static std::shared_ptr<marker> from_stmt(sqlite3_stmt* stmt,
                                             std::unordered_map<unsigned int, std::shared_ptr<cv::slam::data::keyframe>>& keyframes);

    // Save marker info to db
    static std::vector<std::pair<std::string, std::string>> columns() {
        return std::vector<std::pair<std::string, std::string>>{
            {"corners_pos_w", "BLOB"},
            {"keep_fixed", "INTEGER"},
            {"n_observations", "INTEGER"},
            {"observations", "BLOB"},
            {"initialized_before", "INTEGER"}};
    };
    bool bind_to_stmt(sqlite3* db, sqlite3_stmt* stmt) const;

    //! corner positions
    eigen_alloc_vector<Vec3_t> corners_pos_w_;

    //! marker ID
    unsigned int id_;

    bool keep_fixed_ = false;

    bool initialized_before_ = false;

    //! marker model
    std::shared_ptr<marker_model::base> marker_model_;

    //! observed keyframes
    std::unordered_map<unsigned int, std::shared_ptr<keyframe>> observations_;

    mutable std::mutex mtx_position_;
};

} // namespace data
} // namespace cv::slam

#endif // SLAM_DATA_MARKER_H
