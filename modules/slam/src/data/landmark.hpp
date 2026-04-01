#ifndef SLAM_DATA_LANDMARK_H
#define SLAM_DATA_LANDMARK_H

#include "type.hpp"

#include <map>
#include <mutex>
#include <atomic>
#include <memory>

#include <opencv2/core/mat.hpp>
#include <nlohmann/json_fwd.hpp>
#include <sqlite3.h>

namespace cv::slam {
namespace data {

class frame;

class keyframe;

class map_database;

class landmark : public std::enable_shared_from_this<landmark> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    //! Data structure for sorting keyframes by ID for consistent results in local map cleaning/BA
    using observations_t = std::map<std::weak_ptr<keyframe>, unsigned int, id_less<std::weak_ptr<keyframe>>>;

    //! constructor
    landmark(unsigned int id, const Vec3_t& pos_w, const std::shared_ptr<keyframe>& ref_keyfrm);

    //! constructor for map loading with computing parameters which can be recomputed
    landmark(const unsigned int id, const unsigned int first_keyfrm_id,
             const Vec3_t& pos_w, const std::shared_ptr<keyframe>& ref_keyfrm,
             const unsigned int num_visible, const unsigned int num_found);

    virtual ~landmark();

    // Factory method for create landmark
    static std::shared_ptr<landmark> from_stmt(sqlite3_stmt* stmt,
                                               std::unordered_map<unsigned int, std::shared_ptr<cv::slam::data::keyframe>>& keyframes,
                                               unsigned int next_landmark_id,
                                               unsigned int next_keyframe_id);

    /**
     * Save this landmark information to db
     */
    static std::vector<std::pair<std::string, std::string>> columns() {
        return std::vector<std::pair<std::string, std::string>>{
            {"first_keyfrm", "INTEGER"},
            {"pos_w", "BLOB"},
            {"ref_keyfrm", "INTEGER"},
            {"n_vis", "INTEGER"},
            {"n_fnd", "INTEGER"}};
    };
    bool bind_to_stmt(sqlite3* db, sqlite3_stmt* stmt) const;

    //! set world coordinates of this landmark
    void set_pos_in_world(const Vec3_t& pos_w);
    //! get world coordinates of this landmark
    Vec3_t get_pos_in_world() const;

    //! get mean normalized vector of keyframe->lm vectors, for keyframes such that observe the 3D point.
    Vec3_t get_obs_mean_normal() const;
    //! get reference keyframe, a keyframe at the creation of a given 3D point
    std::shared_ptr<keyframe> get_ref_keyframe() const;

    //! add observation
    void add_observation(const std::shared_ptr<keyframe>& keyfrm, unsigned int idx);
    //! erase observation
    void erase_observation(map_database* map_db, const std::shared_ptr<keyframe>& keyfrm);

    //! get observations (keyframe and keypoint idx)
    observations_t get_observations() const;
    //! get number of observations
    unsigned int num_observations() const;
    //! whether this landmark is observed from more than zero keyframes
    bool has_observation() const;

    //! get index of associated keypoint in the specified keyframe
    int get_index_in_keyframe(const std::shared_ptr<keyframe>& keyfrm) const;
    //! whether this landmark is observed in the specified keyframe
    bool is_observed_in_keyframe(const std::shared_ptr<keyframe>& keyfrm) const;

    //! check the distance between landmark and camera is in ORB scale variance
    inline bool is_inside_in_orb_scale(const float cam_to_lm_dist, const float margin_far, const float margin_near) const {
        const float max_dist = margin_far * get_max_valid_distance();
        const float min_dist = margin_near * get_min_valid_distance();
        return (min_dist <= cam_to_lm_dist && cam_to_lm_dist <= max_dist);
    }

    //! true if the landmark has representative descriptor
    bool has_representative_descriptor() const;

    //! get representative descriptor
    cv::Mat get_descriptor() const;

    //! compute representative descriptor
    void compute_descriptor();

    //! update observation mean normal and ORB scale variance
    void update_mean_normal_and_obs_scale_variance();

    //! true if the landmark has valid prediction parameters
    bool has_valid_prediction_parameters() const;
    //! get max valid distance between landmark and camera
    float get_min_valid_distance() const;
    //! get min valid distance between landmark and camera
    float get_max_valid_distance() const;

    //! predict scale level assuming this landmark is observed in the specified frame/keyframe
    unsigned int predict_scale_level(const float cam_to_lm_dist, float num_scale_levels, float log_scale_factor) const;

    //! erase this landmark from database
    void prepare_for_erasing(map_database* map_db);
    //! whether this landmark will be erased shortly or not
    bool will_be_erased();

    //! Make an interconnection by landmark::add_observation and keyframe::add_landmark
    void connect_to_keyframe(const std::shared_ptr<keyframe>& keyfrm, unsigned int idx);

    //! replace this with specified landmark
    void replace(std::shared_ptr<landmark> lm, data::map_database* map_db);

    void increase_num_observable(unsigned int num_observable = 1);
    void increase_num_observed(unsigned int num_observed = 1);
    unsigned int get_num_observed() const;
    unsigned int get_num_observable() const;
    float get_observed_ratio() const;

    //! encode landmark information as JSON
    nlohmann::json to_json() const;

public:
    unsigned int id_;
    unsigned int first_keyfrm_id_ = 0;
    unsigned int num_observations_ = 0;

protected:
    void compute_mean_normal(const observations_t& observations,
                             const Vec3_t& pos_w,
                             Vec3_t& mean_normal) const;
    void compute_orb_scale_variance(const observations_t& observations,
                                    const std::shared_ptr<keyframe>& ref_keyfrm,
                                    const Vec3_t& pos_w,
                                    float& max_valid_dist,
                                    float& min_valid_dist) const;

private:
    //! world coordinates of this landmark
    Vec3_t pos_w_;

    //! observations (keyframe and keypoint index)
    observations_t observations_;

    //! true if the landmark has representative descriptor
    std::atomic<bool> has_representative_descriptor_{false};
    //! representative descriptor
    cv::Mat descriptor_;

    //! reference keyframe
    std::weak_ptr<keyframe> ref_keyfrm_;

    // track counter
    unsigned int num_observable_ = 1;
    unsigned int num_observed_ = 1;

    //! this landmark will be erased shortly or not
    std::atomic<bool> will_be_erased_{false};

    // parameters for prediction
    //! true if the landmark has valid prediction parameters
    std::atomic<bool> has_valid_prediction_parameters_{false};
    //! Normalized average vector (unit vector) of keyframe->lm, for keyframes such that observe the 3D point.
    Vec3_t mean_normal_ = Vec3_t::Zero();
    //! max valid distance between landmark and camera
    float min_valid_dist_ = 0;
    //! min valid distance between landmark and camera
    float max_valid_dist_ = 0;

    mutable std::mutex mtx_position_;
    mutable std::mutex mtx_observations_;
};

} // namespace data
} // namespace cv::slam

#endif // SLAM_DATA_LANDMARK_H
