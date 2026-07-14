#ifndef SLAM_CAMERA_RADIAL_DIVISION_H
#define SLAM_CAMERA_RADIAL_DIVISION_H

#include "camera/base.hpp"

#include <opencv2/core/mat.hpp>

namespace cv::slam {
namespace camera {

// This class implements the camera model presented in:
//
//   "Simultaneous linear estimation of multiple view geometry and lens
//   distortion" by Andrew Fitzgibbon, CVPR 2001.
//
// The model is easy to implement and fast to evaluate.
// It is well suited for wide angle lenses like used in action cameras
// implemented by Steffen Urban, March 2020 (urbste@googlemail.com)
class radial_division final : public base {
public:
    radial_division(const std::string& name, const setup_type_t& setup_type, const color_order_t& color_order,
                    const unsigned int cols, const unsigned int rows, const double fps,
                    const double fx, const double fy, const double cx, const double cy,
                    const double distortion, const double focal_x_baseline = 0.0, const double depth_thr = 0.0);

    radial_division(const cv::FileNode& yaml_node);

    ~radial_division() override;

    void show_parameters() const override final;

    image_bounds compute_image_bounds() const override final;

    cv::Point2f undistort_point(const cv::Point2f& dist_pt) const override final;

    Vec3_t convert_point_to_bearing(const cv::Point2f& undist_pt) const override final;

    cv::Point2f convert_bearing_to_point(const Vec3_t& bearing) const override final;

    bool reproject_to_image(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec2_t& reproj, float& x_right) const override final;

    bool reproject_to_bearing(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec3_t& reproj) const override final;

    nlohmann::json to_json() const override final;

    //-------------------------
    // Parameters specific to this model

    //! pinhole params
    const double fx_;
    const double fy_;
    const double cx_;
    const double cy_;
    const double fx_inv_;
    const double fy_inv_;

    //! distortion params
    const double distortion_;

    //! camera matrix in OpenCV format
    cv::Mat cv_cam_matrix_;
    //! camera matrix in Eigen format
    Mat33_t eigen_cam_matrix_;
};

} // namespace camera
} // namespace cv::slam

#endif // SLAM_CAMERA_RADIAL_DIVISION_H
