#ifndef SLAM_INITIALIZE_BEARING_VECTOR_H
#define SLAM_INITIALIZE_BEARING_VECTOR_H

#include "type.hpp"
#include "initialize/base.hpp"

namespace cv::slam {

namespace data {
class frame;
} // namespace data

namespace initialize {

class bearing_vector final : public base {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bearing_vector() = delete;

    //! Constructor
    bearing_vector(const data::frame& ref_frm,
                   const unsigned int num_ransac_iters,
                   const unsigned int min_num_triangulated,
                   const unsigned int min_num_valid_pts,
                   const float parallax_deg_thr,
                   const float reproj_err_thr,
                   bool use_fixed_seed = false);

    //! Destructor
    ~bearing_vector() override;

    //! Initialize with the current frame
    bool initialize(const data::frame& cur_frm, const std::vector<int>& ref_matches_with_cur) override;

private:
    //! Reconstruct the initial map with the E matrix
    //! (NOTE: the output variables will be set if succeeded)
    bool reconstruct_with_E(const Mat33_t& E_ref_to_cur, const std::vector<bool>& is_inlier_match);

    //! Use fixed random seed for RANSAC if true
    const bool use_fixed_seed_;
};

} // namespace initialize
} // namespace cv::slam

#endif // SLAM_INITIALIZE_BEARING_VECTOR_H
