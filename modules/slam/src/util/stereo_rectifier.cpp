#include "camera/perspective.hpp"
#include "camera/fisheye.hpp"
#include "util/stereo_rectifier.hpp"
#include "util/yaml.hpp"

#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace util {

stereo_rectifier::stereo_rectifier(const std::shared_ptr<cv::slam::config>& cfg, camera::base* camera)
    : stereo_rectifier(camera,
                       cv::slam::util::yaml_optional_ref(cfg->yaml_node_, "StereoRectifier")) {}

stereo_rectifier::stereo_rectifier(camera::base* camera, const YAML::Node& yaml_node)
    : model_type_(load_model_type(yaml_node)) {
    CV_LOG_DEBUG(&g_log_tag, "CONSTRUCT: util::stereo_rectifier");
    if (camera->setup_type_ != camera::setup_type_t::Stereo) {
        throw std::runtime_error("When stereo rectification is used, 'setup' must be set to 'stereo'");
    }
    if (camera->model_type_ != camera::model_type_t::Perspective) {
        throw std::runtime_error("When stereo rectification is used, 'model' must be set to 'perspective'");
    }
    // set image size
    const cv::Size img_size(camera->cols_, camera->rows_);
    // set camera matrices
    const auto K_l = parse_vector_as_mat(cv::Size(3, 3), yaml_node["K_left"].as<std::vector<double>>());
    const auto K_r = parse_vector_as_mat(cv::Size(3, 3), yaml_node["K_right"].as<std::vector<double>>());
    // set rotation matrices
    const auto R_l = parse_vector_as_mat(cv::Size(3, 3), yaml_node["R_left"].as<std::vector<double>>());
    const auto R_r = parse_vector_as_mat(cv::Size(3, 3), yaml_node["R_right"].as<std::vector<double>>());
    // set distortion parameters depending on the camera model
    const auto D_l_vec = yaml_node["D_left"].as<std::vector<double>>();
    const auto D_r_vec = yaml_node["D_right"].as<std::vector<double>>();
    const auto D_l = parse_vector_as_mat(cv::Size(1, D_l_vec.size()), D_l_vec);
    const auto D_r = parse_vector_as_mat(cv::Size(1, D_r_vec.size()), D_r_vec);
    // get camera matrix after rectification
    const auto K_rect = static_cast<camera::perspective*>(camera)->cv_cam_matrix_;
    // create undistortion maps
    switch (model_type_) {
        case camera::model_type_t::Perspective: {
            cv::initUndistortRectifyMap(K_l, D_l, R_l, K_rect, img_size, CV_32F, undist_map_x_l_, undist_map_y_l_);
            cv::initUndistortRectifyMap(K_r, D_r, R_r, K_rect, img_size, CV_32F, undist_map_x_r_, undist_map_y_r_);
            break;
        }
        case camera::model_type_t::Fisheye: {
            cv::fisheye::initUndistortRectifyMap(K_l, D_l, R_l, K_rect, img_size, CV_32F, undist_map_x_l_, undist_map_y_l_);
            cv::fisheye::initUndistortRectifyMap(K_r, D_r, R_r, K_rect, img_size, CV_32F, undist_map_x_r_, undist_map_y_r_);
            break;
        }
        default: {
            throw std::runtime_error("Invalid model type for stereo rectification: " + camera->get_model_type_string());
        }
    }
}

stereo_rectifier::~stereo_rectifier() {
    CV_LOG_DEBUG(&g_log_tag, "DESTRUCT: util::stereo_rectifier");
}

void stereo_rectifier::rectify(const cv::Mat& in_img_l, const cv::Mat& in_img_r,
                               cv::Mat& out_img_l, cv::Mat& out_img_r) const {
    cv::remap(in_img_l, out_img_l, undist_map_x_l_, undist_map_y_l_, cv::INTER_LINEAR);
    cv::remap(in_img_r, out_img_r, undist_map_x_r_, undist_map_y_r_, cv::INTER_LINEAR);
}

cv::Mat stereo_rectifier::parse_vector_as_mat(const cv::Size& shape, const std::vector<double>& vec) {
    cv::Mat mat(shape, CV_64F);
    std::memcpy(mat.data, vec.data(), shape.height * shape.width * sizeof(double));
    return mat;
}

camera::model_type_t stereo_rectifier::load_model_type(const YAML::Node& yaml_node) {
    const auto model_type_str = yaml_node["model"].as<std::string>("perspective");
    if (model_type_str == "perspective") {
        return camera::model_type_t::Perspective;
    }
    else if (model_type_str == "fisheye") {
        return camera::model_type_t::Fisheye;
    }
    else if (model_type_str == "equirectangular") {
        return camera::model_type_t::Equirectangular;
    }

    throw std::runtime_error("Invalid camera model: " + model_type_str);
}

} // namespace util
} // namespace cv::slam
