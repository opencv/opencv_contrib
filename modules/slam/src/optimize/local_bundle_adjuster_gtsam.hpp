#ifndef SLAM_OPTIMIZE_LOCAL_BUNDLE_ADJUSTER_GTSAM_H
#define SLAM_OPTIMIZE_LOCAL_BUNDLE_ADJUSTER_GTSAM_H

#include "optimize/local_bundle_adjuster.hpp"

#include <memory>

namespace cv::slam {

namespace data {
class keyframe;
class map_database;
} // namespace data

namespace optimize {

class local_bundle_adjuster_gtsam : public local_bundle_adjuster {
public:
    /**
     * Constructor
     * @param yaml_node
     * @param num_first_iter
     * @param num_second_iter
     */
    explicit local_bundle_adjuster_gtsam(const cv::FileNode& yaml_node,
                                         const unsigned int num_first_iter = 5,
                                         const unsigned int num_second_iter = 10);

    /**
     * Destructor
     */
    virtual ~local_bundle_adjuster_gtsam() = default;

    /**
     * Perform optimization
     * @param map_db
     * @param curr_keyfrm
     * @param force_stop_flag
     */
    void optimize(data::map_database* map_db, const std::shared_ptr<data::keyframe>& curr_keyfrm, bool* const force_stop_flag) const override;

private:
    //! number of iterations of first optimization
    const unsigned int num_first_iter_;
    //! number of iterations of second optimization
    const unsigned int num_second_iter_;
    //!
    const unsigned int use_additional_keyframes_for_monocular_ = false;
};

} // namespace optimize
} // namespace cv::slam

#endif // SLAM_OPTIMIZE_LOCAL_BUNDLE_ADJUSTER_GTSAM_H
