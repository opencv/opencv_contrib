#include "camera/base.hpp"
#include "util/yaml.hpp"

#include <iostream>
#include <opencv2/core/mat.hpp>

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace camera {

base::base(const std::string& name, const setup_type_t setup_type, const model_type_t model_type, const color_order_t color_order,
           const unsigned int cols, const unsigned int rows, const double fps,
           const double focal_x_baseline, const double true_baseline, const double depth_thr)
    : name_(name), setup_type_(setup_type), model_type_(model_type), color_order_(color_order),
      cols_(cols), rows_(rows), fps_(fps),
      focal_x_baseline_(focal_x_baseline), true_baseline_(true_baseline), depth_thr_(depth_thr) {
    CV_LOG_DEBUG(&g_log_tag, "CONSTRUCT: camera::base");
}

base::~base() {
    CV_LOG_DEBUG(&g_log_tag, "DESTRUCT: camera::base");
}

setup_type_t base::load_setup_type(const cv::FileNode& yaml_node) {
    const auto setup_type_str = util::yaml_get_req_str(yaml_node, "setup");
    if (setup_type_str == "monocular") {
        return camera::setup_type_t::Monocular;
    }
    else if (setup_type_str == "stereo") {
        return camera::setup_type_t::Stereo;
    }
    else if (setup_type_str == "RGBD") {
        return camera::setup_type_t::RGBD;
    }

    throw std::runtime_error("Invalid setup type: " + setup_type_str);
}

setup_type_t base::load_setup_type(const std::string& setup_type_str) {
    const auto itr = std::find(setup_type_to_string.begin(), setup_type_to_string.end(), setup_type_str);
    if (itr == setup_type_to_string.end()) {
        throw std::runtime_error("Invalid setup type: " + setup_type_str);
    }
    return static_cast<setup_type_t>(std::distance(setup_type_to_string.begin(), itr));
}

model_type_t base::load_model_type(const cv::FileNode& yaml_node) {
    const auto model_type_str = util::yaml_get_req_str(yaml_node, "model");
    if (model_type_str == "perspective") {
        return camera::model_type_t::Perspective;
    }
    else if (model_type_str == "fisheye") {
        return camera::model_type_t::Fisheye;
    }
    else if (model_type_str == "equirectangular") {
        return camera::model_type_t::Equirectangular;
    }
    else if (model_type_str == "radial_division") {
        return camera::model_type_t::RadialDivision;
    }
    throw std::runtime_error("Invalid camera model: " + model_type_str);
}

model_type_t base::load_model_type(const std::string& model_type_str) {
    const auto itr = std::find(model_type_to_string.begin(), model_type_to_string.end(), model_type_str);
    if (itr == model_type_to_string.end()) {
        throw std::runtime_error("Invalid camera model: " + model_type_str);
    }
    return static_cast<model_type_t>(std::distance(model_type_to_string.begin(), itr));
}

color_order_t base::load_color_order(const cv::FileNode& yaml_node) {
    if (yaml_node["color_order"].empty()) {
        return color_order_t::Gray;
    }

    const auto color_order_str = util::yaml_get_req_str(yaml_node, "color_order");
    if (color_order_str == "Gray") {
        return color_order_t::Gray;
    }
    else if (color_order_str == "RGB" || color_order_str == "RGBA") {
        return color_order_t::RGB;
    }
    else if (color_order_str == "BGR" || color_order_str == "BGRA") {
        return color_order_t::BGR;
    }

    throw std::runtime_error("Invalid color order: " + color_order_str);
}

color_order_t base::load_color_order(const std::string& color_order_str) {
    const auto itr = std::find(color_order_to_string.begin(), color_order_to_string.end(), color_order_str);
    if (itr == color_order_to_string.end()) {
        throw std::runtime_error("Invalid color order: " + color_order_str);
    }
    return static_cast<color_order_t>(std::distance(color_order_to_string.begin(), itr));
}

bool base::is_valid_shape(const cv::Mat& img) const {
    return static_cast<int>(cols_) == img.cols && static_cast<int>(rows_) == img.rows;
}

void base::show_common_parameters() const {
    std::cout << "- name: " << name_ << std::endl;
    std::cout << "- setup: " << get_setup_type_string() << std::endl;
    std::cout << "- fps: " << fps_ << std::endl;
    std::cout << "- cols: " << cols_ << std::endl;
    std::cout << "- rows: " << rows_ << std::endl;
    std::cout << "- color: " << get_color_order_string() << std::endl;
    std::cout << "- model: " << get_model_type_string() << std::endl;
}

std::ostream& operator<<(std::ostream& os, const base& params) {
    os << "- name: " << params.name_ << std::endl;
    os << "- setup: " << params.get_setup_type_string() << std::endl;
    os << "- fps: " << params.fps_ << std::endl;
    os << "- cols: " << params.cols_ << std::endl;
    os << "- rows: " << params.rows_ << std::endl;
    os << "- color: " << params.get_color_order_string() << std::endl;
    os << "- model: " << params.get_model_type_string() << std::endl;
    return os;
}

cv::KeyPoint base::undistort_keypoint(const cv::KeyPoint& dist_keypt) const {
    cv::KeyPoint undist_keypt;
    undist_keypt.pt = undistort_point(dist_keypt.pt);
    undist_keypt.angle = dist_keypt.angle;
    undist_keypt.size = dist_keypt.size;
    undist_keypt.octave = dist_keypt.octave;
    return undist_keypt;
}

void base::undistort_points(const std::vector<cv::Point2f>& dist_pts, std::vector<cv::Point2f>& undist_pts) const {
    // fill cv::Mat with distorted points
    undist_pts.resize(dist_pts.size());
    for (unsigned long idx = 0; idx < dist_pts.size(); ++idx) {
        undist_pts.at(idx) = undistort_point(dist_pts.at(idx));
    }
}

void base::undistort_keypoints(const std::vector<cv::KeyPoint>& dist_keypts, std::vector<cv::KeyPoint>& undist_keypts) const {
    // fill cv::Mat with distorted keypoints
    undist_keypts.resize(dist_keypts.size());
    for (unsigned long idx = 0; idx < dist_keypts.size(); ++idx) {
        undist_keypts.at(idx) = undistort_keypoint(dist_keypts.at(idx));
        undist_keypts.at(idx).angle = dist_keypts.at(idx).angle;
        undist_keypts.at(idx).size = dist_keypts.at(idx).size;
        undist_keypts.at(idx).octave = dist_keypts.at(idx).octave;
    }
}

void base::convert_points_to_bearings(const std::vector<cv::Point2f>& undist_pts, eigen_alloc_vector<Vec3_t>& bearings) const {
    assert(bearings.size() == 0);
    std::transform(undist_pts.begin(), undist_pts.end(), std::back_inserter(bearings),
                   [this](const cv::Point2f& undist_pt) { return convert_point_to_bearing(undist_pt); });
}

void base::convert_keypoints_to_bearings(const std::vector<cv::KeyPoint>& undist_keypts, eigen_alloc_vector<Vec3_t>& bearings) const {
    assert(bearings.size() == 0);
    std::transform(undist_keypts.begin(), undist_keypts.end(), std::back_inserter(bearings),
                   [this](const cv::KeyPoint& undist_keypt) { return convert_point_to_bearing(undist_keypt.pt); });
}

void base::convert_bearings_to_points(const eigen_alloc_vector<Vec3_t>& bearings, std::vector<cv::Point2f>& undist_pts) const {
    std::transform(bearings.begin(), bearings.end(), std::back_inserter(undist_pts),
                   [this](const Vec3_t& bearing) { return convert_bearing_to_point(bearing); });
}

} // namespace camera
} // namespace cv::slam
