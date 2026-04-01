#include "camera/perspective.hpp"

#include <iostream>

#include <opencv2/core/utils/logger.hpp>
#include <nlohmann/json.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace camera {

perspective::perspective(const std::string& name, const setup_type_t& setup_type, const color_order_t& color_order,
                         const unsigned int cols, const unsigned int rows, const double fps,
                         const double fx, const double fy, const double cx, const double cy,
                         const double k1, const double k2, const double p1, const double p2, const double k3,
                         const double focal_x_baseline, const double depth_thr)
    : base(name, setup_type, model_type_t::Perspective, color_order, cols, rows, fps, focal_x_baseline, focal_x_baseline / fx, depth_thr),
      fx_(fx), fy_(fy), cx_(cx), cy_(cy), fx_inv_(1.0 / fx), fy_inv_(1.0 / fy),
      k1_(k1), k2_(k2), p1_(p1), p2_(p2), k3_(k3) {
    CV_LOG_DEBUG(&g_log_tag, "CONSTRUCT: camera::perspective");

    cv_cam_matrix_ = (cv::Mat_<float>(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
    cv_dist_params_ = (cv::Mat_<float>(5, 1) << k1_, k2_, p1_, p2_, k3_);

    eigen_cam_matrix_ << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
    eigen_dist_params_ << k1_, k2_, p1_, p2_, k3_;

    img_bounds_ = compute_image_bounds();
}

perspective::perspective(const YAML::Node& yaml_node)
    : perspective(yaml_node["name"].as<std::string>(),
                  load_setup_type(yaml_node),
                  load_color_order(yaml_node),
                  yaml_node["cols"].as<unsigned int>(),
                  yaml_node["rows"].as<unsigned int>(),
                  yaml_node["fps"].as<double>(),
                  yaml_node["fx"].as<double>(),
                  yaml_node["fy"].as<double>(),
                  yaml_node["cx"].as<double>(),
                  yaml_node["cy"].as<double>(),
                  yaml_node["k1"].as<double>(),
                  yaml_node["k2"].as<double>(),
                  yaml_node["p1"].as<double>(),
                  yaml_node["p2"].as<double>(),
                  yaml_node["k3"].as<double>(),
                  yaml_node["focal_x_baseline"].as<double>(0.0),
                  yaml_node["depth_threshold"].as<double>(40.0)) {}

perspective::~perspective() {
    CV_LOG_DEBUG(&g_log_tag, "DESTRUCT: camera::perspective");
}

void perspective::show_parameters() const {
    show_common_parameters();
    std::cout << "  - fx: " << fx_ << std::endl;
    std::cout << "  - fy: " << fy_ << std::endl;
    std::cout << "  - cx: " << cx_ << std::endl;
    std::cout << "  - cy: " << cy_ << std::endl;
    std::cout << "  - k1: " << k1_ << std::endl;
    std::cout << "  - k2: " << k2_ << std::endl;
    std::cout << "  - p1: " << p1_ << std::endl;
    std::cout << "  - p2: " << p2_ << std::endl;
    std::cout << "  - k3: " << k3_ << std::endl;
    std::cout << "  - min x: " << img_bounds_.min_x_ << std::endl;
    std::cout << "  - max x: " << img_bounds_.max_x_ << std::endl;
    std::cout << "  - min y: " << img_bounds_.min_y_ << std::endl;
    std::cout << "  - max y: " << img_bounds_.max_y_ << std::endl;
}

image_bounds perspective::compute_image_bounds() const {
    CV_LOG_DEBUG(&g_log_tag, "compute image bounds");

    if (k1_ == 0 && k2_ == 0 && p1_ == 0 && p2_ == 0 && k3_ == 0) {
        // any distortion does not exist

        return image_bounds{0.0, cols_, 0.0, rows_};
    }
    else {
        // distortion exists

        // corner coordinates: (x, y) = (col, row)
        const std::vector<cv::KeyPoint> corners{cv::KeyPoint(0.0, 0.0, 1.0),      // left top
                                                cv::KeyPoint(cols_, 0.0, 1.0),    // right top
                                                cv::KeyPoint(0.0, rows_, 1.0),    // left bottom
                                                cv::KeyPoint(cols_, rows_, 1.0)}; // right bottom

        std::vector<cv::KeyPoint> undist_corners;
        undistort_keypoints(corners, undist_corners);

        return image_bounds{std::min(undist_corners.at(0).pt.x, undist_corners.at(2).pt.x),
                            std::max(undist_corners.at(1).pt.x, undist_corners.at(3).pt.x),
                            std::min(undist_corners.at(0).pt.y, undist_corners.at(1).pt.y),
                            std::max(undist_corners.at(2).pt.y, undist_corners.at(3).pt.y)};
    }
}

cv::Point2f perspective::undistort_point(const cv::Point2f& dist_pt) const {
    // fill cv::Mat with distorted point
    cv::Mat mat(1, 2, CV_32F);
    mat.at<float>(0, 0) = dist_pt.x;
    mat.at<float>(0, 1) = dist_pt.y;

    // undistort
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, cv_cam_matrix_, cv_dist_params_, cv::Mat(), cv_cam_matrix_,
                        cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 20, 1e-6));
    mat = mat.reshape(1);

    // convert to cv::Mat
    cv::Point2f undist_pt;
    undist_pt.x = mat.at<float>(0, 0);
    undist_pt.y = mat.at<float>(0, 1);

    return undist_pt;
}

Vec3_t perspective::convert_point_to_bearing(const cv::Point2f& undist_pt) const {
    const auto x_normalized = (undist_pt.x - cx_) / fx_;
    const auto y_normalized = (undist_pt.y - cy_) / fy_;
    const auto l2_norm = std::sqrt(x_normalized * x_normalized + y_normalized * y_normalized + 1.0);
    return Vec3_t{x_normalized / l2_norm, y_normalized / l2_norm, 1.0 / l2_norm};
}

cv::Point2f perspective::convert_bearing_to_point(const Vec3_t& bearing) const {
    const auto x_normalized = bearing(0) / bearing(2);
    const auto y_normalized = bearing(1) / bearing(2);
    return cv::Point2f(fx_ * x_normalized + cx_, fy_ * y_normalized + cy_);
}

bool perspective::reproject_to_image(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec2_t& reproj, float& x_right) const {
    // convert to camera-coordinates
    const Vec3_t pos_c = rot_cw * pos_w + trans_cw;

    // check if the point is visible
    if (pos_c(2) <= 0.0) {
        return false;
    }

    // reproject onto the image
    const auto z_inv = 1.0 / pos_c(2);
    reproj(0) = fx_ * pos_c(0) * z_inv + cx_;
    reproj(1) = fy_ * pos_c(1) * z_inv + cy_;
    x_right = reproj(0) - focal_x_baseline_ * z_inv;

    // check if the point is visible
    return (img_bounds_.min_x_ < reproj(0) && reproj(0) < img_bounds_.max_x_
            && img_bounds_.min_y_ < reproj(1) && reproj(1) < img_bounds_.max_y_);
}

bool perspective::reproject_to_bearing(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec3_t& reproj) const {
    // convert to camera-coordinates
    reproj = rot_cw * pos_w + trans_cw;

    // check if the point is visible
    if (reproj(2) <= 0.0) {
        return false;
    }

    // reproject onto the image
    const auto z_inv = 1.0 / reproj(2);
    const auto x = fx_ * reproj(0) * z_inv + cx_;
    const auto y = fy_ * reproj(1) * z_inv + cy_;

    // convert to a bearing
    reproj.normalize();

    // check if the point is visible
    return (img_bounds_.min_x_ < x && x < img_bounds_.max_x_
            && img_bounds_.min_y_ < y && y < img_bounds_.max_y_);
}

nlohmann::json perspective::to_json() const {
    return {{"model_type", get_model_type_string()},
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
            {"k1", k1_},
            {"k2", k2_},
            {"p1", p1_},
            {"p2", p2_},
            {"k3", k3_}};
}

std::ostream& operator<<(std::ostream& os, const perspective& params) {
    os << "- name: " << params.name_ << std::endl;
    os << "- setup: " << params.get_setup_type_string() << std::endl;
    os << "- fps: " << params.fps_ << std::endl;
    os << "- cols: " << params.cols_ << std::endl;
    os << "- rows: " << params.rows_ << std::endl;
    os << "- color: " << params.get_color_order_string() << std::endl;
    os << "- model: " << params.get_model_type_string() << std::endl;
    os << "  - fx: " << params.fx_ << std::endl;
    os << "  - fy: " << params.fy_ << std::endl;
    os << "  - cx: " << params.cx_ << std::endl;
    os << "  - cy: " << params.cy_ << std::endl;
    os << "  - k1: " << params.k1_ << std::endl;
    os << "  - k2: " << params.k2_ << std::endl;
    os << "  - p1: " << params.p1_ << std::endl;
    os << "  - p2: " << params.p2_ << std::endl;
    os << "  - k3: " << params.k3_ << std::endl;
    os << "  - min x: " << params.img_bounds_.min_x_ << std::endl;
    os << "  - max x: " << params.img_bounds_.max_x_ << std::endl;
    os << "  - min y: " << params.img_bounds_.min_y_ << std::endl;
    os << "  - max y: " << params.img_bounds_.max_y_ << std::endl;
    return os;
}

//! Override for optimization

void perspective::undistort_points(const std::vector<cv::Point2f>& dist_pts, std::vector<cv::Point2f>& undist_pts) const {
    // cv::undistortPoints does not accept an empty input
    if (dist_pts.empty()) {
        undist_pts.clear();
        return;
    }

    // fill cv::Mat with distorted points
    cv::Mat mat(dist_pts.size(), 2, CV_32F);
    for (unsigned long idx = 0; idx < dist_pts.size(); ++idx) {
        mat.at<float>(idx, 0) = dist_pts.at(idx).x;
        mat.at<float>(idx, 1) = dist_pts.at(idx).y;
    }

    // undistort
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, cv_cam_matrix_, cv_dist_params_, cv::Mat(), cv_cam_matrix_,
                        cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 20, 1e-6));
    mat = mat.reshape(1);

    // convert to cv::Mat
    undist_pts.resize(dist_pts.size());
    for (unsigned long idx = 0; idx < undist_pts.size(); ++idx) {
        undist_pts.at(idx).x = mat.at<float>(idx, 0);
        undist_pts.at(idx).y = mat.at<float>(idx, 1);
    }
}

void perspective::undistort_keypoints(const std::vector<cv::KeyPoint>& dist_keypts, std::vector<cv::KeyPoint>& undist_keypts) const {
    // cv::undistortPoints does not accept an empty input
    if (dist_keypts.empty()) {
        undist_keypts.clear();
        return;
    }

    // fill cv::Mat with distorted keypoints
    cv::Mat mat(dist_keypts.size(), 2, CV_32F);
    for (unsigned long idx = 0; idx < dist_keypts.size(); ++idx) {
        mat.at<float>(idx, 0) = dist_keypts.at(idx).pt.x;
        mat.at<float>(idx, 1) = dist_keypts.at(idx).pt.y;
    }

    // undistort
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, cv_cam_matrix_, cv_dist_params_, cv::Mat(), cv_cam_matrix_,
                        cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 20, 1e-6));
    mat = mat.reshape(1);

    // convert to cv::Mat
    undist_keypts.resize(dist_keypts.size());
    for (unsigned long idx = 0; idx < undist_keypts.size(); ++idx) {
        undist_keypts.at(idx).pt.x = mat.at<float>(idx, 0);
        undist_keypts.at(idx).pt.y = mat.at<float>(idx, 1);
        undist_keypts.at(idx).angle = dist_keypts.at(idx).angle;
        undist_keypts.at(idx).size = dist_keypts.at(idx).size;
        undist_keypts.at(idx).octave = dist_keypts.at(idx).octave;
    }
}

} // namespace camera
} // namespace cv::slam
