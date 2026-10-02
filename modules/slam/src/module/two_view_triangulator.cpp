#include "data/keyframe.hpp"
#include "module/two_view_triangulator.hpp"
#include "solve/triangulator.hpp"

namespace cv::slam {
namespace module {

two_view_triangulator::two_view_triangulator(const std::shared_ptr<data::keyframe>& keyfrm_1, const std::shared_ptr<data::keyframe>& keyfrm_2,
                                             const float rays_parallax_deg_thr)
    : keyfrm_1_(keyfrm_1), keyfrm_2_(keyfrm_2),
      rot_1w_(keyfrm_1->get_rot_cw()), rot_w1_(rot_1w_.transpose()), trans_1w_(keyfrm_1->get_trans_cw()),
      cam_pose_1w_(keyfrm_1->get_pose_cw()), cam_center_1_(keyfrm_1->get_trans_wc()), camera_1_(keyfrm_1->camera_),
      rot_2w_(keyfrm_2->get_rot_cw()), rot_w2_(rot_2w_.transpose()), trans_2w_(keyfrm_2->get_trans_cw()),
      cam_pose_2w_(keyfrm_2->get_pose_cw()), cam_center_2_(keyfrm_2->get_trans_wc()), camera_2_(keyfrm_2->camera_),
      ratio_factor_(2.0f * std::max(keyfrm_1->orb_params_->scale_factor_, keyfrm_2->orb_params_->scale_factor_)),
      cos_rays_parallax_thr_(std::cos(rays_parallax_deg_thr * M_PI / 180.0)) {}

bool two_view_triangulator::triangulate(const unsigned idx_1, const unsigned int idx_2, Vec3_t& pos_w) const {
    const auto& keypt_1 = keyfrm_1_->frm_obs_.undist_keypts_.at(idx_1);
    const float keypt_1_x_right = keyfrm_1_->frm_obs_.stereo_x_right_.empty() ? -1.0f : keyfrm_1_->frm_obs_.stereo_x_right_.at(idx_1);
    const bool is_stereo_1 = 0 <= keypt_1_x_right;

    const auto& keypt_2 = keyfrm_2_->frm_obs_.undist_keypts_.at(idx_2);
    const float keypt_2_x_right = keyfrm_2_->frm_obs_.stereo_x_right_.empty() ? -1.0f : keyfrm_2_->frm_obs_.stereo_x_right_.at(idx_2);
    const bool is_stereo_2 = 0 <= keypt_2_x_right;

    // rays with reference of each camera
    const Vec3_t ray_c_1 = keyfrm_1_->frm_obs_.bearings_.at(idx_1);
    const Vec3_t ray_c_2 = keyfrm_2_->frm_obs_.bearings_.at(idx_2);
    // rays with the world reference
    const Vec3_t ray_w_1 = rot_w1_ * ray_c_1;
    const Vec3_t ray_w_2 = rot_w2_ * ray_c_2;
    const auto cos_rays_parallax = ray_w_1.dot(ray_w_2);

    // compute the stereo parallax if the keypoint is observed as stereo
    const float depth_1 = keyfrm_1_->frm_obs_.depths_.empty() ? -1.0f : keyfrm_1_->frm_obs_.depths_.at(idx_1);
    const auto cos_stereo_parallax_1 = is_stereo_1
                                           ? std::cos(2.0 * atan2(camera_1_->true_baseline_ / 2.0, depth_1))
                                           : 2.0;
    const float depth_2 = keyfrm_2_->frm_obs_.depths_.empty() ? -1.0f : keyfrm_2_->frm_obs_.depths_.at(idx_2);
    const auto cos_stereo_parallax_2 = is_stereo_2
                                           ? std::cos(2.0 * atan2(camera_2_->true_baseline_ / 2.0, depth_2))
                                           : 2.0;
    const auto cos_stereo_parallax = std::min(cos_stereo_parallax_1, cos_stereo_parallax_2);

    // select to use "linear triangulation" or "stereo triangulation"
    // threshold of minimum angle of the two rays
    const bool triangulate_with_two_cameras =
        // check if the sufficient parallax is provided
        ((!is_stereo_1 && !is_stereo_2) && 0.0 < cos_rays_parallax && cos_rays_parallax < cos_rays_parallax_thr_)
        // check if the parallax between the two cameras is larger than the stereo parallax
        || ((is_stereo_1 || is_stereo_2) && 0.0 < cos_rays_parallax && cos_rays_parallax < cos_stereo_parallax);

    // triangulate
    if (triangulate_with_two_cameras) {
        pos_w = solve::triangulator::triangulate(ray_c_1, ray_c_2, cam_pose_1w_, cam_pose_2w_);
    }
    else if (is_stereo_1 && cos_stereo_parallax_1 < cos_stereo_parallax_2) {
        pos_w = keyfrm_1_->triangulate_stereo(idx_1);
    }
    else if (is_stereo_2 && cos_stereo_parallax_2 < cos_stereo_parallax_1) {
        pos_w = keyfrm_2_->triangulate_stereo(idx_2);
    }
    else {
        return false;
    }

    // check the triangulated point is located in front of the two cameras
    if (!check_depth_is_positive(pos_w, rot_1w_, trans_1w_, camera_1_)
        || !check_depth_is_positive(pos_w, rot_2w_, trans_2w_, camera_2_)) {
        return false;
    }

    // reject the point if reprojection errors are larger than reasonable threshold
    if (!check_reprojection_error(pos_w, rot_1w_, trans_1w_, camera_1_, keypt_1.pt, keypt_1_x_right,
                                  keyfrm_1_->orb_params_->level_sigma_sq_.at(keypt_1.octave), is_stereo_1)
        || !check_reprojection_error(pos_w, rot_2w_, trans_2w_, camera_2_, keypt_2.pt, keypt_2_x_right,
                                     keyfrm_2_->orb_params_->level_sigma_sq_.at(keypt_2.octave), is_stereo_2)) {
        return false;
    }

    // reject the point if the real scale factor and the predicted one are much different
    if (!check_scale_factors(pos_w,
                             keyfrm_1_->orb_params_->scale_factors_.at(keypt_1.octave),
                             keyfrm_2_->orb_params_->scale_factors_.at(keypt_2.octave))) {
        return false;
    }

    return true;
}

template<typename T>
bool two_view_triangulator::check_reprojection_error(const Vec3_t& pos_w, const Mat33_t& rot_cw, const Vec3_t& trans_cw, camera::base* const camera,
                                                     const cv::Point_<T>& keypt, const float x_right, const float sigma_sq, const bool is_stereo) const {
    assert(is_stereo ^ (x_right < 0));

    // chi-squared values for p=5%
    // (n=2)
    constexpr float chi_sq_2D = 5.99146;
    // (n=3)
    constexpr float chi_sq_3D = 7.81473;

    Vec2_t reproj_in_cur;
    float x_right_in_cur;
    camera->reproject_to_image(rot_cw, trans_cw, pos_w, reproj_in_cur, x_right_in_cur);

    if (is_stereo) {
        const Vec2_t reproj_err = reproj_in_cur - keypt;
        const auto reproj_err_x_right = x_right_in_cur - x_right;
        if ((chi_sq_3D * sigma_sq) < (reproj_err.squaredNorm() + reproj_err_x_right * reproj_err_x_right)) {
            return false;
        }
    }
    else {
        const Vec2_t reproj_err = reproj_in_cur - keypt;
        if ((chi_sq_2D * sigma_sq) < reproj_err.squaredNorm()) {
            return false;
        }
    }

    return true;
}

} // namespace module
} // namespace cv::slam
