#ifndef SLAM_DATA_FRAME_H
#define SLAM_DATA_FRAME_H

#include "type.hpp"
#include "camera/base.hpp"
#include "feature/orb_params.hpp"
#include "util/converter.hpp"
#include "data/frame_observation.hpp"
#include "data/bow_vocabulary.hpp"
#include "data/marker2d.hpp"
#include "data/bow_vocabulary_fwd.hpp"

#include <vector>
#include <atomic>
#include <memory>
#include <unordered_set>

#include <Eigen/Core>

namespace cv::slam {

namespace camera {
class base;
} // namespace camera

namespace feature {
class orb_extractor;
struct orb_params;
} // namespace feature

namespace data {

class keyframe;
class landmark;

class frame {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    frame() = default;

    bool operator==(const frame& frm) { return this->id_ == frm.id_; }
    bool operator!=(const frame& frm) { return !(*this == frm); }

    /**
     * Constructor for monocular frame
     * @param frame_id
     * @param timestamp
     * @param camera
     * @param orb_params
     * @param frm_obs
     * @param markers_2d
     */
    frame(const unsigned int frame_id, const double timestamp, camera::base* camera, feature::orb_params* orb_params,
          const frame_observation frm_obs, const std::unordered_map<unsigned int, marker2d>& markers_2d);

    /**
     * Set camera pose and refresh rotation and translation
     * @param pose_cw
     */
    void set_pose_cw(const Mat44_t& pose_cw);

    /**
     * Get camera pose
     */
    Mat44_t get_pose_cw() const;

    /**
     * Get the inverse of the camera pose
     */
    Mat44_t get_pose_wc() const;

    /**
     * Get camera center
     * @return
     */
    Vec3_t get_trans_wc() const;

    /**
     * Get inverse of rotation
     * @return
     */
    Mat33_t get_rot_wc() const;

    /**
     * Get the translation of the camera pose
     */
    Vec3_t get_trans_cw() const {
        return trans_cw_;
    }

    /**
     * Get the rotation of the camera pose
     */
    Mat33_t get_rot_cw() const {
        return rot_cw_;
    }

    /**
     * Invalidate pose
     */
    void invalidate_pose() {
        pose_is_valid_ = false;
    }

    /**
     * Return true if pose is valid
     */
    bool pose_is_valid() const {
        return pose_is_valid_;
    }

    /**
     * Returns true if BoW has been computed.
     */
    bool bow_is_available() const;

    /**
     * Compute BoW representation
     */
    void compute_bow(bow_vocabulary* bow_vocab);

    /**
     * Check observability of the landmark
     */
    bool can_observe(const std::shared_ptr<landmark>& lm, const float ray_cos_thr,
                     Vec2_t& reproj, float& x_right, unsigned int& pred_scale_level) const;

    bool has_landmark(const std::shared_ptr<landmark>& lm) const;

    void add_landmark(const std::shared_ptr<landmark>&, const unsigned int idx);

    std::shared_ptr<landmark> get_landmark(const unsigned int idx) const;

    void erase_landmark_with_index(const unsigned int idx);

    void erase_landmark(const std::shared_ptr<landmark>& lm);

    std::vector<std::shared_ptr<landmark>> get_landmarks() const;

    void erase_landmarks();

    void set_landmarks(const std::vector<std::shared_ptr<landmark>>& landmarks);

    /**
     * Get keypoint indices in the cell which reference point is located
     * @param ref_x
     * @param ref_y
     * @param margin
     * @param min_level
     * @param max_level
     * @return
     */
    std::vector<unsigned int> get_keypoints_in_cell(const float ref_x, const float ref_y, const float margin, const int min_level = -1, const int max_level = -1) const;

    /**
     * Perform stereo triangulation of the keypoint
     * @param idx
     * @return
     */
    Vec3_t triangulate_stereo(const unsigned int idx) const;

    //! current frame ID
    unsigned int id_;

    //! timestamp
    double timestamp_;

    //! camera model
    camera::base* camera_ = nullptr;

    //! ORB scale pyramid information
    const feature::orb_params* orb_params_ = nullptr;

    //! constant observations
    frame_observation frm_obs_;

    //! markers 2D (ID to marker2d map)
    std::unordered_map<unsigned int, marker2d> markers_2d_;

    //! BoW features (DBoW2 or FBoW)
    bow_vector bow_vec_;
    bow_feature_vector bow_feat_vec_;

    //! reference keyframe for tracking
    std::shared_ptr<keyframe> ref_keyfrm_ = nullptr;

private:
    //! landmarks, whose nullptr indicates no-association
    std::vector<std::shared_ptr<landmark>> landmarks_;
    std::unordered_map<std::shared_ptr<landmark>, unsigned int> landmarks_idx_map_;

    //! camera pose: world -> camera
    bool pose_is_valid_ = false;
    Mat44_t pose_cw_;

    //! Camera pose
    //! rotation: world -> camera
    Mat33_t rot_cw_;
    //! translation: world -> camera
    Vec3_t trans_cw_;
    //! rotation: camera -> world
    Mat33_t rot_wc_;
    //! translation: camera -> world
    Vec3_t trans_wc_;
};

} // namespace data
} // namespace cv::slam

#endif // SLAM_DATA_FRAME_H
