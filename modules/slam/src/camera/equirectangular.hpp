#ifndef SLAM_CAMERA_EQUIRECTANGULAR_H
#define SLAM_CAMERA_EQUIRECTANGULAR_H

#include "camera/base.hpp"

namespace cv::slam {
namespace camera {

class equirectangular final : public base {
public:
    equirectangular(const std::string& name, const color_order_t& color_order,
                    const unsigned int cols, const unsigned int rows, const double fps);

    equirectangular(const cv::FileNode& yaml_node);

    ~equirectangular() override;

    void show_parameters() const override final;

    image_bounds compute_image_bounds() const override final;

    cv::Point2f undistort_point(const cv::Point2f& dist_pt) const override final;

    Vec3_t convert_point_to_bearing(const cv::Point2f& undist_pt) const override final;

    cv::Point2f convert_bearing_to_point(const Vec3_t& bearing) const override final;

    bool reproject_to_image(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec2_t& reproj, float& x_right) const override final;

    bool reproject_to_bearing(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec3_t& reproj) const override final;

    nlohmann::json to_json() const override final;

    //! Override for optimization
    void undistort_points(const std::vector<cv::Point2f>& dist_pts, std::vector<cv::Point2f>& undist_pts) const override final;
    void undistort_keypoints(const std::vector<cv::KeyPoint>& dist_keypts, std::vector<cv::KeyPoint>& undist_keypts) const override final;
};

std::ostream& operator<<(std::ostream& os, const equirectangular& params);

} // namespace camera
} // namespace cv::slam

#endif // SLAM_CAMERA_EQUIRECTANGULAR_H
