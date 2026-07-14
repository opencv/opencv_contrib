#ifndef SLAM_DATA_MAP_DATABASE_H
#define SLAM_DATA_MAP_DATABASE_H

#include "data/bow_vocabulary_fwd.hpp"
#include "data/frame_statistics.hpp"

#include <mutex>
#include <vector>
#include <unordered_map>
#include <memory>

#include <nlohmann/json_fwd.hpp>

typedef struct sqlite3 sqlite3;

namespace cv::slam {

namespace camera {
class base;
} // namespace camera

namespace data {

class frame;
class keyframe;
class landmark;
class marker;
class camera_database;
class orb_params_database;
class bow_database;

class map_database {
public:
    /**
     * Constructor
     */
    map_database(unsigned int min_num_shared_lms);

    /**
     * Destructor
     */
    ~map_database();

    /**
     * Set fixed_keyframe_id_threshold
     */
    void set_fixed_keyframe_id_threshold();

    /**
     * Get fixed_keyframe_id_threshold
     */
    unsigned int get_fixed_keyframe_id_threshold();

    /**
     * Add keyframe to the database
     * @param keyfrm
     */
    void add_keyframe(const std::shared_ptr<keyframe>& keyfrm);

    /**
     * Erase keyframe from the database
     * @param keyfrm
     */
    void erase_keyframe(const std::shared_ptr<keyframe>& keyfrm);

    /**
     * Get keyframe from the database
     * @param id
     */
    std::shared_ptr<keyframe> get_keyframe(unsigned int id) const;

    /**
     * Add landmark to the database
     * @param lm
     */
    void add_landmark(std::shared_ptr<landmark>& lm);

    /**
     * Erase landmark from the database
     * @param id
     */
    void erase_landmark(unsigned int id);

    /**
     * Get landmark from the database
     * @param id
     */
    std::shared_ptr<landmark> get_landmark(unsigned int id) const;

    /**
     * Add marker to the database
     * @param mkr
     */
    void add_marker(const std::shared_ptr<marker>& mkr);

    /**
     * Erase marker from the database
     * @param mkr
     */
    void erase_marker(const std::shared_ptr<marker>& mkr);

    /**
     * Set local landmarks
     * @param local_lms
     */
    void set_local_landmarks(const std::vector<std::shared_ptr<landmark>>& local_lms);

    /**
     * Get local landmarks
     * @return
     */
    std::vector<std::shared_ptr<landmark>> get_local_landmarks() const;

    /**
     * Get all of the keyframes in the database
     * NOTE: Access multiple spanning trees. Used only to read and write databases.
     * @return
     */
    std::vector<std::shared_ptr<keyframe>> get_all_keyframes() const;

    /**
     * Get closest keyframes to a given 2d pose
     * @param pose Given 2d pose
     * @param normal_vector normal vector of plane
     * @param distance_threshold Maximum distance where close keyframes could be found
     * @param angle_threshold Maximum angle between given pose and close keyframes
     * @return Vector closest keyframes
     */
    std::vector<std::shared_ptr<keyframe>> get_close_keyframes_2d(const Mat44_t& pose_cw,
                                                                  const Vec3_t& normal_vector,
                                                                  const double distance_threshold,
                                                                  const double angle_threshold) const;

    /**
     * Get closest keyframes to a given pose
     * @param pose Given pose
     * @param distance_threshold Maximum distance where close keyframes could be found
     * @param angle_threshold Maximum angle between given pose and close keyframes
     * @return Vector closest keyframes
     */
    std::vector<std::shared_ptr<keyframe>> get_close_keyframes(const Mat44_t& pose_cw,
                                                               const double distance_threshold,
                                                               const double angle_threshold) const;

    /**
     * Get the number of keyframes
     * @return
     */
    unsigned get_num_keyframes() const;

    /**
     * Get all of the landmarks in the database
     * @return
     */
    std::vector<std::shared_ptr<landmark>> get_all_landmarks() const;

    /**
     * Get the last keyframe added to the database
     * @return shared pointer to the last keyframe added to the database
     */
    std::shared_ptr<keyframe> get_last_inserted_keyframe() const;

    /**
     * Get all of the markers in the database
     * @return
     */
    std::vector<std::shared_ptr<marker>> get_all_markers() const;

    /**
     * Get the number of markers
     * @return
     */
    unsigned int get_num_markers() const;

    /**
     * Get marker
     * @return marker
     */
    std::shared_ptr<marker> get_marker(unsigned int id) const;

    /**
     * Add spanning root
     */
    void add_spanning_root(std::shared_ptr<keyframe>& keyframe);

    /**
     * Get spanning roots
     */
    std::vector<std::shared_ptr<keyframe>> get_spanning_roots();

    /**
     * Get the number of landmarks
     * @return
     */
    unsigned int get_num_landmarks() const;

    /**
     * Get minimum threshold for covisibility graph connection
     * @return minimum threshold for covisibility graph connection
     */
    unsigned int get_min_num_shared_lms() const;

    /**
     * Update frame statistics
     * @param frm
     * @param is_lost
     */
    void update_frame_statistics(const data::frame& frm, const bool is_lost) {
        std::lock_guard<std::mutex> lock(mtx_map_access_);
        frm_stats_.update_frame_statistics(frm, is_lost);
    }

    /**
     * Replace a keyframe which will be erased in frame statistics
     * @param old_keyfrm
     * @param new_keyfrm
     */
    void replace_reference_keyframe(const std::shared_ptr<data::keyframe>& old_keyfrm, const std::shared_ptr<data::keyframe>& new_keyfrm) {
        std::lock_guard<std::mutex> lock(mtx_map_access_);
        frm_stats_.replace_reference_keyframe(old_keyfrm, new_keyfrm);
    }

    /**
     * Get frame statistics
     * @return
     */
    frame_statistics get_frame_statistics() const {
        std::lock_guard<std::mutex> lock(mtx_map_access_);
        return frm_stats_;
    }

    /**
     * Clear the database
     */
    void clear();

    /**
     * Load keyframes and landmarks from JSON
     * @param cam_db
     * @param orb_params_db
     * @param bow_vocab
     * @param json_keyfrms
     * @param json_landmarks
     */
    void from_json(camera_database* cam_db, orb_params_database* orb_params_db, bow_vocabulary* bow_vocab,
                   const nlohmann::json& json_keyfrms, const nlohmann::json& json_landmarks);

    /**
     * Dump keyframes and landmarks as JSON
     * @param json_keyfrms
     * @param json_landmarks
     */
    void to_json(nlohmann::json& json_keyfrms, nlohmann::json& json_landmarks) const;

    /**
     * Load keyframes and landmarks from database
     */
#ifdef USE_SQLITE3
    bool from_db(sqlite3* db,
#endif
                 camera_database* cam_db,
                 orb_params_database* orb_params_db,
                 bow_vocabulary* bow_vocab);

    /**
     * Dump keyframes and landmarks to database
     */
#ifdef USE_SQLITE3
    bool to_db(sqlite3* db) const;

    //! mutex for locking ALL access to the database
    //! (NOTE: cannot used in map_database class)
    static std::mutex mtx_database_;

    //! next ID
#endif
    std::atomic<unsigned int> next_keyframe_id_{0};
    std::atomic<unsigned int> next_landmark_id_{0};

private:
    /**
     * Decode JSON and register keyframe information to the map database
     * (NOTE: objects which are not constructed yet will be set as nullptr)
     * @param cam_db
     * @param orb_params_db
     * @param bow_vocab
     * @param id
     * @param json_keyfrm
     */
    void register_keyframe(camera_database* cam_db, orb_params_database* orb_params_db, bow_vocabulary* bow_vocab,
                           const unsigned int id, const nlohmann::json& json_keyfrm);

    /**
     * Decode JSON and register landmark information to the map database
     * (NOTE: objects which are not constructed yet will be set as nullptr)
     * @param id
     * @param json_landmark
     */
    void register_landmark(const unsigned int id, const nlohmann::json& json_landmark);

    /**
     * Decode JSON and register essential graph information
     * (NOTE: keyframe database must be completely constructed before calling this function)
     * @param id
     * @param json_keyfrm
     */
    void register_graph(const unsigned int id, const nlohmann::json& json_keyfrm);

    /**
     * Decode JSON and register keyframe-landmark associations
     * (NOTE: keyframe and landmark database must be completely constructed before calling this function)
     * @param keyfrm_id
     * @param json_keyfrm
     */
    void register_association(const unsigned int keyfrm_id, const nlohmann::json& json_keyfrm);

#ifdef USE_SQLITE3
    bool load_keyframes_from_db(sqlite3* db,
#endif
                                const std::string& table_name,
                                camera_database* cam_db,
                                orb_params_database* orb_params_db,
                                bow_vocabulary* bow_vocab);
#ifdef USE_SQLITE3
    bool load_landmarks_from_db(sqlite3* db, const std::string& table_name);
    void load_association_from_stmt(sqlite3_stmt* stmt);
    bool load_associations_from_db(sqlite3* db, const std::string& table_name);
    bool save_keyframes_to_db(sqlite3* db, const std::string& table_name) const;
    bool save_landmarks_to_db(sqlite3* db, const std::string& table_name) const;
    static std::vector<std::pair<std::string, std::string>> association_columns() {
#endif
        return std::vector<std::pair<std::string, std::string>>{
            {"lm_ids", "BLOB"},
            {"span_parent", "INTEGER"},
            {"n_spanning_children", "INTEGER"},
            {"spanning_children", "BLOB"},
            {"n_loop_edges", "INTEGER"},
            {"loop_edges", "BLOB"}};
    };
#ifdef USE_SQLITE3
    bool bind_association_to_stmt(sqlite3_stmt* stmt,
#endif
                                  const std::shared_ptr<keyframe>& keyfrm) const;
#ifdef USE_SQLITE3
    bool save_associations_to_db(sqlite3* db, const std::string& table_name) const;

    bool load_markers_from_db(sqlite3* db, const std::string& table_name);
    bool save_markers_to_db(sqlite3* db, const std::string& table_name) const;

    //! mutex for mutual exclusion controll between class methods
#endif
    mutable std::mutex mtx_map_access_;

    //-----------------------------------------
    // keyframe and landmark database

    //! IDs and keyframes
    std::unordered_map<unsigned int, std::shared_ptr<keyframe>> keyframes_;
    //! IDs and landmarks
    std::unordered_map<unsigned int, std::shared_ptr<landmark>> landmarks_;
    //! IDs and markers
    std::unordered_map<unsigned int, std::shared_ptr<marker>> markers_;

    //! spanning roots
    std::vector<std::shared_ptr<keyframe>> spanning_roots_;

    //! The last keyframe added to the database
    std::shared_ptr<keyframe> last_inserted_keyfrm_ = nullptr;

    //! local landmarks
    std::vector<std::shared_ptr<landmark>> local_landmarks_;

    //! keyframes with id less than or equal to fixed_keyframe_id_threshold are not optimized
    unsigned int fixed_keyframe_id_threshold_ = 0;

    //-----------------------------------------
    // parameters for global/local mapping (optimization)

    //! minimum threshold for covisibility graph connection
    const unsigned int min_num_shared_lms_ = 15;

    //-----------------------------------------
    // frame statistics for odometry evaluation

    //! frame statistics
    frame_statistics frm_stats_;
};

} // namespace data
} // namespace cv::slam

#endif // SLAM_DATA_MAP_DATABASE_H
