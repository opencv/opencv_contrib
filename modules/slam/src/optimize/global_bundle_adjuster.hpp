#ifndef SLAM_OPTIMIZE_GLOBAL_BUNDLE_ADJUSTER_H
#define SLAM_OPTIMIZE_GLOBAL_BUNDLE_ADJUSTER_H

namespace cv::slam {

namespace data {
class map_database;
} // namespace data

namespace optimize {

class global_bundle_adjuster {
public:
    /**
     * Constructor
     * @param num_iter
     * @param use_huber_kernel
     * @param verbose
     */
    explicit global_bundle_adjuster(
        unsigned int num_iter = 10,
        bool use_huber_kernel = true,
        bool verbose = false);

    /**
     * Destructor
     */
    virtual ~global_bundle_adjuster() = default;

    void optimize_for_initialization(const std::vector<std::shared_ptr<data::keyframe>>& keyfrms,
                                     const std::vector<std::shared_ptr<data::landmark>>& lms,
                                     const std::vector<std::shared_ptr<data::marker>>& markers,
                                     float gain_threshold,
                                     bool fix_markers,
                                     bool* const force_stop_flag = nullptr) const;

    /**
     * Perform optimization
     * @param keyfrms
     * @param optimized_keyfrm_ids
     * @param optimized_landmark_ids
     * @param lm_to_pos_w_after_global_BA
     * @param keyfrm_to_pose_cw_after_global_BA
     * @param force_stop_flag
     * @return false if aborted
     */
    bool optimize(const std::vector<std::shared_ptr<data::keyframe>>& keyfrms,
                  std::unordered_set<unsigned int>& optimized_keyfrm_ids,
                  std::unordered_set<unsigned int>& optimized_landmark_ids,
                  std::unordered_set<unsigned int>& optimized_marker_ids,
                  eigen_alloc_unord_map<unsigned int, Vec3_t>& lm_to_pos_w_after_global_BA,
                  eigen_alloc_unord_map<unsigned int, Mat44_t>& keyfrm_to_pose_cw_after_global_BA,
                  eigen_alloc_unord_map<unsigned int, std::array<Vec3_t, 4>>& marker_to_pos_w_after_global_BA,
                  bool* const force_stop_flag = nullptr) const;

private:
    //! number of iterations of optimization
    unsigned int num_iter_;
    //! use Huber loss or not
    const bool use_huber_kernel_;
    //! Verbosity (for g2o)
    const bool verbose_ = false;
};

} // namespace optimize
} // namespace cv::slam

#endif // SLAM_OPTIMIZE_GLOBAL_BUNDLE_ADJUSTER_H
