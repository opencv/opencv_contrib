#ifndef SLAM_DATA_KEYFRAME_H
#define SLAM_DATA_KEYFRAME_H

#include "type.hpp"
#include "camera/base.hpp"
#include "feature/orb_params.hpp"
#include "data/graph_node.hpp"
#include "data/bow_vocabulary.hpp"
#include "data/frame_observation.hpp"
#include "data/marker2d.hpp"
#include "data/bow_vocabulary_fwd.hpp"

#include <set>
#include <mutex>
#include <atomic>
#include <memory>

#include <nlohmann/json_fwd.hpp>
#ifdef USE_SQLITE3
#include <sqlite3.h>
#endif

namespace cv::slam {

namespace camera {
class base;
} // namespace camera

namespace data {

class frame;
class landmark;
class marker;
class marker2d;
class map_database;
class bow_database;
class camera_database;
class orb_params_database;

class keyframe : public std::enable_shared_from_this<keyframe> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * Constructor for building from a frame
     */
    explicit keyframe(unsigned int id, const frame& frm);

    /**
     * Constructor for map loading
     * (NOTE: some variables must be recomputed after the construction. See the definition.)
     */
    keyframe(const unsigned int id,
             const double timestamp, const Mat44_t& pose_cw, camera::base* camera,
             const feature::orb_params* orb_params, const frame_observation& frm_obs,
             const bow_vector& bow_vec, const bow_feature_vector& bow_feat_vec,
             std::unordered_map<unsigned int, marker2d> markers_2d = {});
    virtual ~keyframe();

    // Factory method for create keyframe
    static std::shared_ptr<keyframe> make_keyframe(unsigned int id, const frame& frm);
    static std::shared_ptr<keyframe> make_keyframe(
        const unsigned int id,
        const double timestamp, const Mat44_t& pose_cw, camera::base* camera,
        const feature::orb_params* orb_params, const frame_observation& frm_obs,
        const bow_vector& bow_vec, const bow_feature_vector& bow_feat_vec,
        std::unordered_map<unsigned int, marker2d> markers_2d = {});
    static std::shared_ptr<keyframe> from_stmt(sqlite3_stmt* stmt,
                                               camera_database* cam_db,
                                               orb_params_database* orb_params_db,
                                               bow_vocabulary* bow_vocab,
                                               unsigned int next_keyframe_id);

    // operator overrides
    bool operator==(const keyframe& keyfrm) const { return id_ == keyfrm.id_; }
    bool operator!=(const keyframe& keyfrm) const { return !(*this == keyfrm); }
    bool operator<(const keyframe& keyfrm) const { return id_ < keyfrm.id_; }
    bool operator<=(const keyframe& keyfrm) const { return id_ <= keyfrm.id_; }
    bool operator>(const keyframe& keyfrm) const { return id_ > keyfrm.id_; }
    bool operator>=(const keyframe& keyfrm) const { return id_ >= keyfrm.id_; }

    /**
     * Encode this keyframe information as JSON
     */
    nlohmann::json to_json() const;

    /**
     * Save this keyframe information to db
     */
    static std::vector<std::pair<std::string, std::string>> columns() {
        return std::vector<std::pair<std::string, std::string>>{
            {"src_frm_id", "INTEGER"}, // removed
            {"ts", "REAL"},
            {"cam", "BLOB"},
            {"orb_params", "BLOB"},
            {"pose_cw", "BLOB"},
            {"n_keypts", "INTEGER"},
            {"undist_keypts", "BLOB"},
            {"x_rights", "BLOB"},
            {"depths", "BLOB"},
            {"descs", "BLOB"},
            {"n_markers", "INTEGER"},
            {"markers", "BLOB"}};
    };
#ifdef USE_SQLITE3
    bool bind_to_stmt(sqlite3* db, sqlite3_stmt* stmt) const;
#endif

    //-----------------------------------------
    // camera pose

    /**
     * Set camera pose
     */
    void set_pose_cw(const Mat44_t& pose_cw);

    /**
     * Get the camera pose
     */
    Mat44_t get_pose_cw() const;

    /**
     * Get the inverse of the camera pose
     */
    Mat44_t get_pose_wc() const;

    /**
     * Get the camera center
     */
    Vec3_t get_trans_wc() const;

    /**
     * Get the rotation of the camera pose
     */
    Mat33_t get_rot_cw() const;

    /**
     * Get the translation of the camera pose
     */
    Vec3_t get_trans_cw() const;

    //-----------------------------------------
    // features and observations

    /**
     * Returns true if BoW has been computed.
     */
    bool bow_is_available() const;

    /**
     * Compute BoW representation
     */
    void compute_bow(bow_vocabulary* bow_vocab);

    /**
     * Add a landmark observed by myself at keypoint idx
     */
    void add_landmark(std::shared_ptr<landmark> lm, const unsigned int idx);

    /**
     * Erase a landmark observed by myself at keypoint idx
     */
    void erase_landmark_with_index(const unsigned int idx);

    /**
     * Erase a landmark
     */
    void erase_landmark(const std::shared_ptr<landmark>& lm);

    /**
     * Update all of the landmarks
     */
    void update_landmarks();

    /**
     * Get all of the landmarks
     * (NOTE: including nullptr)
     */
    std::vector<std::shared_ptr<landmark>> get_landmarks() const;

    /**
     * Get the valid landmarks
     */
    std::set<std::shared_ptr<landmark>> get_valid_landmarks() const;

    /**
     * Get the number of tracked landmarks which have observers equal to or greater than the threshold
     */
    unsigned int get_num_tracked_landmarks(const unsigned int min_num_obs_thr) const;

    /**
     * Get the landmark associated keypoint idx
     */
    std::shared_ptr<landmark>& get_landmark(const unsigned int idx);

    /**
     * Get the keypoint indices in the cell which reference point is located
     */
    std::vector<unsigned int> get_keypoints_in_cell(const float ref_x, const float ref_y, const float margin,
                                                    const int min_level = -1, const int max_level = -1) const;

    /**
     * Triangulate the keypoint using the disparity
     */
    Vec3_t triangulate_stereo(const unsigned int idx) const;

    /**
     * Compute median of depths
     */
    float compute_median_depth(const bool abs = false) const;

    /**
     * Compute median of distances
     */
    float compute_median_distance() const;

    /**
     * Whether or not the camera setting is capable of obtaining depth information
     */
    bool depth_is_available() const;

    /**
     * Add a marker
     */
    void add_marker(const std::shared_ptr<marker>& mkr);

    /**
     * Get all of the markers
     * (NOTE: including nullptr)
     */
    std::vector<std::shared_ptr<marker>> get_markers() const;

    //-----------------------------------------
    // flags

    /**
     * Set this keyframe as non-erasable
     */
    void set_not_to_be_erased();

    /**
     * Set this keyframe as erasable
     */
    void set_to_be_erased();

    /**
     * Erase this keyframe
     */
    void prepare_for_erasing(map_database* map_db, bow_database* bow_db);

    /**
     * Whether this keyframe will be erased shortly or not
     */
    bool will_be_erased();

    //-----------------------------------------
    // meta information

    //! keyframe ID
    unsigned int id_;

    //! timestamp in seconds
    const double timestamp_;

    //-----------------------------------------
    // camera parameters

    //! camera model
    camera::base* camera_;

    //-----------------------------------------
    // feature extraction parameters

    //! ORB feature extraction model
    const feature::orb_params* orb_params_;

    //-----------------------------------------
    // constant observations

    frame_observation frm_obs_;

    //! BoW features (DBoW2 or FBoW)
    bow_vector bow_vec_;
    bow_feature_vector bow_feat_vec_;

    //! observed markers 2D (ID to marker2d map)
    std::unordered_map<unsigned int, marker2d> markers_2d_;

    //-----------------------------------------
    // covisibility graph

    //! graph node
    std::unique_ptr<graph_node> graph_node_ = nullptr;

private:
    //-----------------------------------------
    // camera pose

    //! need mutex for access to poses
    mutable std::mutex mtx_pose_;
    //! camera pose from the world to the current
    Mat44_t pose_cw_;
    //! camera pose from the current to the world
    Mat44_t pose_wc_;
    //! camera center
    Vec3_t trans_wc_;

    //-----------------------------------------
    // observations

    //! need mutex for access to landmark observations
    mutable std::mutex mtx_observations_;
    //! observed landmarks
    std::vector<std::shared_ptr<landmark>> landmarks_;

    //-----------------------------------------
    // marker observations

    //! observed markers
    std::unordered_map<unsigned int, std::shared_ptr<marker>> markers_;

    //-----------------------------------------
    // flags

    //! flag which indicates this keyframe is erasable or not
    std::atomic<bool> cannot_be_erased_{false};

    //! flag which indicates this keyframe will be erased
    std::atomic<bool> will_be_erased_{false};

    //-----------------------------------------
    // misc

    //!
    static constexpr int MARKERS2D_BLOB_NUM_DOUBLES = 33;
};

} // namespace data
} // namespace cv::slam

#endif // SLAM_DATA_KEYFRAME_H
