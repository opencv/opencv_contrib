#include "data/frame.hpp"
#include "initialize/base.hpp"
#include "solve/triangulator.hpp"

namespace cv::slam {
namespace initialize {

base::base(const data::frame& ref_frm,
           const unsigned int num_ransac_iters,
           const unsigned int min_num_triangulated,
           const unsigned int min_num_valid_pts,
           const float parallax_deg_thr,
           const float reproj_err_thr)
    : ref_camera_(ref_frm.camera_), ref_undist_keypts_(ref_frm.frm_obs_.undist_keypts_), ref_bearings_(ref_frm.frm_obs_.bearings_),
      num_ransac_iters_(num_ransac_iters), min_num_triangulated_(min_num_triangulated),
      min_num_valid_pts_(min_num_valid_pts),
      parallax_deg_thr_(parallax_deg_thr), reproj_err_thr_(reproj_err_thr) {}

Mat33_t base::get_rotation_ref_to_cur() const {
    return rot_ref_to_cur_;
}

Vec3_t base::get_translation_ref_to_cur() const {
    return trans_ref_to_cur_;
}

eigen_alloc_vector<Vec3_t> base::get_triangulated_pts() const {
    return triangulated_pts_;
}

std::vector<bool> base::get_triangulated_flags() const {
    return is_triangulated_;
}

bool base::find_most_plausible_pose(const eigen_alloc_vector<Mat33_t>& init_rots, const eigen_alloc_vector<Vec3_t>& init_transes,
                                    const std::vector<bool>& is_inlier_match, const bool depth_is_positive) {
    assert(init_rots.size() == init_transes.size());
    const auto num_hypothesis = init_rots.size();

    // triangulated 3D points
    std::vector<eigen_alloc_vector<Vec3_t>> init_triangulated_pts(num_hypothesis);
    // valid/invalid flag for each 3D point
    std::vector<std::vector<bool>> init_is_triangulated(num_hypothesis);
    // parallax between the two observations of each 3D point
    std::vector<float> init_parallax(num_hypothesis);
    // number of valid 3D points
    std::vector<unsigned int> nums_valid_pts(num_hypothesis);
    // number of triangulated 3D points
    std::vector<unsigned int> num_triangulated_pts(num_hypothesis);

    for (unsigned int i = 0; i < num_hypothesis; ++i) {
        nums_valid_pts.at(i) = triangulate(init_rots.at(i), init_transes.at(i), is_inlier_match, depth_is_positive,
                                           init_triangulated_pts.at(i), init_is_triangulated.at(i), num_triangulated_pts.at(i), init_parallax.at(i));
    }

    rot_ref_to_cur_ = Mat33_t::Zero();
    trans_ref_to_cur_ = Vec3_t::Zero();

    // find the maximum number of the valid points among all of the hypothesis
    const auto max_num_valid_pts_iter = std::max_element(nums_valid_pts.begin(), nums_valid_pts.end());
    // get the index of the hypothesis
    const unsigned int max_num_valid_index = std::distance(nums_valid_pts.begin(), max_num_valid_pts_iter);

    // reject if the number of valid points does not fulfill the threshold
    if (*max_num_valid_pts_iter < min_num_valid_pts_) {
        return false;
    }

    // reject if most plausible hypothesis is unclear
    const auto num_similars = std::count_if(nums_valid_pts.begin(), nums_valid_pts.end(),
                                            [max_num_valid_pts_iter](unsigned int num_valid_pts) {
                                                return 0.8 * (*max_num_valid_pts_iter) < num_valid_pts;
                                            });
    if (1 < num_similars) {
        return false;
    }

    // reject if the parallax is too small
    if (init_parallax.at(max_num_valid_index) > std::cos(parallax_deg_thr_ / 180.0 * M_PI)) {
        return false;
    }

    // reject if the number of 3D points does not fulfill the threshold
    if (num_triangulated_pts.at(max_num_valid_index) < min_num_triangulated_) {
        return false;
    }

    // store the reconstructed map
    rot_ref_to_cur_ = init_rots.at(max_num_valid_index);
    trans_ref_to_cur_ = init_transes.at(max_num_valid_index);
    triangulated_pts_ = init_triangulated_pts.at(max_num_valid_index);
    is_triangulated_ = init_is_triangulated.at(max_num_valid_index);

    return true;
}

unsigned int base::triangulate(const Mat33_t& rot_ref_to_cur, const Vec3_t& trans_ref_to_cur,
                               const std::vector<bool>& is_inlier_match, const bool depth_is_positive,
                               eigen_alloc_vector<Vec3_t>& triangulated_pts,
                               std::vector<bool>& is_triangulated,
                               unsigned int& num_triangulated_pts,
                               float& parallax_cos) {
    // = cos(0.5deg)
    constexpr float cos_parallax_thr = 0.99996192306;
    const float reproj_err_thr_sq = reproj_err_thr_ * reproj_err_thr_;

    // resize buffers according to the number of observed keypoints in the reference
    is_triangulated.resize(ref_undist_keypts_.size(), false);
    triangulated_pts.resize(ref_undist_keypts_.size());

    std::vector<float> cos_parallaxes;
    cos_parallaxes.reserve(ref_undist_keypts_.size());

    // camera centers
    const Vec3_t ref_cam_center = Vec3_t::Zero();
    const Vec3_t cur_cam_center = -rot_ref_to_cur.transpose() * trans_ref_to_cur;

    unsigned int num_valid_pts = 0;
    num_triangulated_pts = 0;

    // for each matching, triangulate a 3D point and compute a parallax and a reprojection error
    for (unsigned int i = 0; i < ref_cur_matches_.size(); ++i) {
        if (!is_inlier_match.at(i)) {
            continue;
        }

        const Vec3_t& ref_bearing = ref_bearings_.at(ref_cur_matches_.at(i).first);
        const Vec3_t& cur_bearing = cur_bearings_.at(ref_cur_matches_.at(i).second);

        const Vec3_t pos_c_in_ref = solve::triangulator::triangulate(ref_bearing, cur_bearing, rot_ref_to_cur, trans_ref_to_cur);

        if (!std::isfinite(pos_c_in_ref(0))
            || !std::isfinite(pos_c_in_ref(1))
            || !std::isfinite(pos_c_in_ref(2))) {
            continue;
        }

        // compute a parallax
        const Vec3_t ref_normal = pos_c_in_ref - ref_cam_center;
        const float ref_norm = ref_normal.norm();
        const Vec3_t cur_normal = pos_c_in_ref - cur_cam_center;
        const float cur_norm = cur_normal.norm();
        const float cos_parallax = ref_normal.dot(cur_normal) / (ref_norm * cur_norm);

        const bool parallax_is_small = cos_parallax_thr < cos_parallax;

        // reject if the 3D point is in front of the cameras
        if (depth_is_positive) {
            if (!parallax_is_small && pos_c_in_ref(2) <= 0) {
                continue;
            }
            const Vec3_t pos_c_in_cur = rot_ref_to_cur * pos_c_in_ref + trans_ref_to_cur;
            if (!parallax_is_small && pos_c_in_cur(2) <= 0) {
                continue;
            }
        }

        const auto& ref_undist_keypt = ref_undist_keypts_.at(ref_cur_matches_.at(i).first);
        const auto& cur_undist_keypt = cur_undist_keypts_.at(ref_cur_matches_.at(i).second);

        // compute a reprojection error in the reference
        Vec2_t reproj_in_ref;
        float x_right_in_ref;
        const auto is_valid_ref = ref_camera_->reproject_to_image(Mat33_t::Identity(), Vec3_t::Zero(), pos_c_in_ref,
                                                                  reproj_in_ref, x_right_in_ref);
        if (!parallax_is_small && !is_valid_ref) {
            continue;
        }

        const float ref_reproj_err_sq = (reproj_in_ref - ref_undist_keypt.pt).squaredNorm();
        if (reproj_err_thr_sq < ref_reproj_err_sq) {
            continue;
        }

        // compute a reprojection error in the current
        Vec2_t reproj_in_cur;
        float x_right_in_cur;
        const auto is_valid_cur = cur_camera_->reproject_to_image(rot_ref_to_cur, trans_ref_to_cur, pos_c_in_ref,
                                                                  reproj_in_cur, x_right_in_cur);
        if (!parallax_is_small && !is_valid_cur) {
            continue;
        }
        const float cur_reproj_err_sq = (reproj_in_cur - cur_undist_keypt.pt).squaredNorm();
        if (reproj_err_thr_sq < cur_reproj_err_sq) {
            continue;
        }

        // triangulation is valid
        ++num_valid_pts;
        cos_parallaxes.push_back(cos_parallax);

        if (!parallax_is_small) {
            // triangulated
            triangulated_pts.at(ref_cur_matches_.at(i).first) = pos_c_in_ref;
            is_triangulated.at(ref_cur_matches_.at(i).first) = true;
            num_triangulated_pts++;
        }
    }

    if (0 < num_valid_pts) {
        // return the 50th smallest parallax
        std::sort(cos_parallaxes.begin(), cos_parallaxes.end());
        const auto idx = std::min(50, static_cast<int>(cos_parallaxes.size() - 1));
        parallax_cos = cos_parallaxes.at(idx);
    }
    else {
        parallax_cos = 1.0;
    }

    return num_valid_pts;
}

} // namespace initialize
} // namespace cv::slam
