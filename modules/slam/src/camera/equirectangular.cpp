#include "camera/equirectangular.hpp"

#include <opencv2/core/utils/logger.hpp>
#include <nlohmann/json.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace camera {

equirectangular::equirectangular(const std::string& name, const color_order_t& color_order,
                                 const unsigned int cols, const unsigned int rows, const double fps)
    : base(name, setup_type_t::Monocular, model_type_t::Equirectangular, color_order, cols, rows, fps, 0.0, 0.0, 0.0) {
    CV_LOG_DEBUG(&g_log_tag, "CONSTRUCT: camera::equirectangular");

    img_bounds_ = compute_image_bounds();
}

equirectangular::equirectangular(const YAML::Node& yaml_node)
    : equirectangular(yaml_node["name"].as<std::string>(),
                      load_color_order(yaml_node),
                      yaml_node["cols"].as<unsigned int>(),
                      yaml_node["rows"].as<unsigned int>(),
                      yaml_node["fps"].as<double>()) {}

equirectangular::~equirectangular() {
    CV_LOG_DEBUG(&g_log_tag, "DESTRUCT: camera::equirectangular");
}

void equirectangular::show_parameters() const {
    show_common_parameters();
}

image_bounds equirectangular::compute_image_bounds() const {
    CV_LOG_DEBUG(&g_log_tag, "compute image bounds");

    return image_bounds{0.0, cols_, 0.0, rows_};
}

cv::Point2f equirectangular::undistort_point(const cv::Point2f& dist_pt) const {
    return dist_pt;
}

Vec3_t equirectangular::convert_point_to_bearing(const cv::Point2f& undist_pt) const {
    // "From Google Street View to 3D City Models (ICCVW 2009)"
    // convert to unit polar coordinates
    const double lon = (undist_pt.x / cols_ - 0.5) * (2.0 * M_PI);
    const double lat = -(undist_pt.y / rows_ - 0.5) * M_PI;
    // convert to equirectangular coordinates
    return Vec3_t{std::cos(lat) * std::sin(lon), -std::sin(lat), std::cos(lat) * std::cos(lon)};
}

cv::Point2f equirectangular::convert_bearing_to_point(const Vec3_t& bearing) const {
    // convert to unit polar coordinates
    const double lat = -std::asin(bearing[1]);
    const double lon = std::atan2(bearing[0], bearing[2]);
    // convert to pixel image coordinated
    return cv::Point2f(cols_ * (0.5 + lon / (2.0 * M_PI)), rows_ * (0.5 - lat / M_PI));
}

bool equirectangular::reproject_to_image(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec2_t& reproj, float& x_right) const {
    // convert to camera-coordinates
    const Vec3_t bearing = (rot_cw * pos_w + trans_cw).normalized();

    // convert to unit polar coordinates
    const auto latitude = -std::asin(bearing(1));
    const auto longitude = std::atan2(bearing(0), bearing(2));

    // convert to pixel image coordinated
    reproj(0) = cols_ * (0.5 + longitude / (2.0 * M_PI));
    reproj(1) = rows_ * (0.5 - latitude / M_PI);
    x_right = 0.0;

    return true;
}

bool equirectangular::reproject_to_bearing(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec3_t& reproj) const {
    // convert to camera-coordinates
    reproj = (rot_cw * pos_w + trans_cw).normalized();

    return true;
}

nlohmann::json equirectangular::to_json() const {
    return {{"model_type", get_model_type_string()},
            {"setup_type", get_setup_type_string()},
            {"color_order", get_color_order_string()},
            {"cols", cols_},
            {"rows", rows_},
            {"fps", fps_},
            {"focal_x_baseline", focal_x_baseline_}};
}

std::ostream& operator<<(std::ostream& os, const equirectangular& params) {
    os << "- name: " << params.name_ << std::endl;
    os << "- setup: " << params.get_setup_type_string() << std::endl;
    os << "- fps: " << params.fps_ << std::endl;
    os << "- cols: " << params.cols_ << std::endl;
    os << "- rows: " << params.rows_ << std::endl;
    os << "- color: " << params.get_color_order_string() << std::endl;
    os << "- model: " << params.get_model_type_string() << std::endl;
    os << "- focal x baseline: " << params.focal_x_baseline_ << std::endl;
    return os;
}

//! Override for optimization

void equirectangular::undistort_points(const std::vector<cv::Point2f>& dist_pts, std::vector<cv::Point2f>& undist_pts) const {
    undist_pts = dist_pts;
}

void equirectangular::undistort_keypoints(const std::vector<cv::KeyPoint>& dist_keypts, std::vector<cv::KeyPoint>& undist_keypts) const {
    undist_keypts = dist_keypts;
}

} // namespace camera
} // namespace cv::slam
