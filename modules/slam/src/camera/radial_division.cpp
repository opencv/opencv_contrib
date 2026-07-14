// Created by Steffen Urban June 2019, urbste@googlemail.com, github.com/urbste

#include "camera/radial_division.hpp"
#include "util/yaml.hpp"

#include <iostream>

#include <opencv2/core/utils/logger.hpp>
#include <nlohmann/json.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace camera {

radial_division::radial_division(const std::string& name, const setup_type_t& setup_type, const color_order_t& color_order,
                                 const unsigned int cols, const unsigned int rows, const double fps,
                                 const double fx, const double fy, const double cx, const double cy,
                                 const double distortion, const double focal_x_baseline, const double depth_thr)
    : base(name, setup_type, model_type_t::RadialDivision, color_order, cols, rows, fps, focal_x_baseline, focal_x_baseline / fx, depth_thr),
      fx_(fx), fy_(fy), cx_(cx), cy_(cy), fx_inv_(1.0 / fx), fy_inv_(1.0 / fy),
      distortion_(distortion) {
    CV_LOG_DEBUG(&g_log_tag, "CONSTRUCT: camera::radial_division");

    cv_cam_matrix_ = (cv::Mat_<float>(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);

    eigen_cam_matrix_ << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;

    img_bounds_ = compute_image_bounds();
}

radial_division::radial_division(const cv::FileNode& yaml_node)
    : radial_division(util::yaml_get_req_str(yaml_node, "name"),
                      load_setup_type(yaml_node),
                      load_color_order(yaml_node),
                      util::yaml_get_req<unsigned int>(yaml_node, "cols"),
                      util::yaml_get_req<unsigned int>(yaml_node, "rows"),
                      util::yaml_get_req<double>(yaml_node, "fps"),
                      util::yaml_get_req<double>(yaml_node, "fx"),
                      util::yaml_get_req<double>(yaml_node, "fy"),
                      util::yaml_get_req<double>(yaml_node, "cx"),
                      util::yaml_get_req<double>(yaml_node, "cy"),
                      util::yaml_get_req<double>(yaml_node, "distortion"),
                      util::yaml_get_val<double>(yaml_node, "focal_x_baseline", 0.0),
                      util::yaml_get_val<double>(yaml_node, "depth_threshold", 40.0)) {}

radial_division::~radial_division() {
    CV_LOG_DEBUG(&g_log_tag, "DESTRUCT: camera::radial_division");
}

void radial_division::show_parameters() const {
    show_common_parameters();
    std::cout << "  - fx: " << fx_ << std::endl;
    std::cout << "  - fy: " << fy_ << std::endl;
    std::cout << "  - cx: " << cx_ << std::endl;
    std::cout << "  - cy: " << cy_ << std::endl;
    std::cout << "  - distortion: " << distortion_ << std::endl;
    std::cout << "  - min x: " << img_bounds_.min_x_ << std::endl;
    std::cout << "  - max x: " << img_bounds_.max_x_ << std::endl;
    std::cout << "  - min y: " << img_bounds_.min_y_ << std::endl;
    std::cout << "  - max y: " << img_bounds_.max_y_ << std::endl;
}

image_bounds radial_division::compute_image_bounds() const {
    CV_LOG_DEBUG(&g_log_tag, "compute image bounds");

    if (distortion_ == 0.0) {
        return image_bounds{0.0, cols_, 0.0, rows_};
    }
    else {
        const std::vector<cv::KeyPoint> corners{cv::KeyPoint(0.0, 0.0, 1.0),
                                                cv::KeyPoint(cols_, 0.0, 1.0),
                                                cv::KeyPoint(0.0, rows_, 1.0),
                                                cv::KeyPoint(cols_, rows_, 1.0)};

        std::vector<cv::KeyPoint> undist_corners;
        undistort_keypoints(corners, undist_corners);

        return image_bounds{std::min(undist_corners.at(0).pt.x, undist_corners.at(2).pt.x),
                            std::max(undist_corners.at(1).pt.x, undist_corners.at(3).pt.x),
                            std::min(undist_corners.at(0).pt.y, undist_corners.at(1).pt.y),
                            std::max(undist_corners.at(2).pt.y, undist_corners.at(3).pt.y)};
    }
}

cv::Point2f radial_division::undistort_point(const cv::Point2f& dist_pt) const {
    // undistort
    const double pixel_x = (dist_pt.x - cx_) / fx_;
    const double pixel_y = (dist_pt.y - cy_) / fy_;
    const double radius_distorted_squared = pixel_x * pixel_x + pixel_y * pixel_y;
    const double undistortion = 1.0 + distortion_ * radius_distorted_squared;

    const double undistorted_pt_x = pixel_x / undistortion;
    const double undistorted_pt_y = pixel_y / undistortion;

    cv::Point2f undist_pt;
    undist_pt.x = undistorted_pt_x * fx_ + cx_;
    undist_pt.y = undistorted_pt_y * fy_ + cy_;

    return undist_pt;
}

Vec3_t radial_division::convert_point_to_bearing(const cv::Point2f& undist_pt) const {
    const auto x_normalized = (undist_pt.x - cx_) / fx_;
    const auto y_normalized = (undist_pt.y - cy_) / fy_;
    const auto l2_norm = std::sqrt(x_normalized * x_normalized + y_normalized * y_normalized + 1.0);
    return Vec3_t{x_normalized / l2_norm, y_normalized / l2_norm, 1.0 / l2_norm};
}

cv::Point2f radial_division::convert_bearing_to_point(const Vec3_t& bearing) const {
    const auto x_normalized = bearing(0) / bearing(2);
    const auto y_normalized = bearing(1) / bearing(2);
    return cv::Point2f(fx_ * x_normalized + cx_, fy_ * y_normalized + cy_);
}

bool radial_division::reproject_to_image(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec2_t& reproj, float& x_right) const {
    const Vec3_t pos_c = rot_cw * pos_w + trans_cw;

    if (pos_c(2) <= 0.0) {
        return false;
    }

    const auto z_inv = 1.0 / pos_c(2);
    reproj(0) = fx_ * pos_c(0) * z_inv + cx_;
    reproj(1) = fy_ * pos_c(1) * z_inv + cy_;
    x_right = reproj(0) - focal_x_baseline_ * z_inv;

    if (reproj(0) < img_bounds_.min_x_ || reproj(0) > img_bounds_.max_x_) {
        return false;
    }
    if (reproj(1) < img_bounds_.min_y_ || reproj(1) > img_bounds_.max_y_) {
        return false;
    }

    return true;
}

bool radial_division::reproject_to_bearing(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec3_t& reproj) const {
    reproj = rot_cw * pos_w + trans_cw;

    if (reproj(2) <= 0.0) {
        return false;
    }

    const auto z_inv = 1.0 / reproj(2);
    const auto x = fx_ * reproj(0) * z_inv + cx_;
    const auto y = fy_ * reproj(1) * z_inv + cy_;

    if (x < img_bounds_.min_x_ || x > img_bounds_.max_x_) {
        return false;
    }
    if (y < img_bounds_.min_y_ || y > img_bounds_.max_y_) {
        return false;
    }

    reproj.normalize();

    return true;
}

nlohmann::json radial_division::to_json() const {
    return {
        {"model_type", get_model_type_string()},
        {"setup_type", get_setup_type_string()},
        {"color_order", get_color_order_string()},
        {"cols", cols_},
        {"rows", rows_},
        {"fps", fps_},
        {"focal_x_baseline", focal_x_baseline_},
        {"fx", fx_},
        {"fy", fy_},
        {"cx", cx_},
        {"cy", cy_},
        {"distortion", distortion_},
    };
}

} // namespace camera
} // namespace cv::slam
