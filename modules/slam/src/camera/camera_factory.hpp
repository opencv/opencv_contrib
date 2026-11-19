#ifndef SLAM_CAMERA_CAMERA_FACTORY_H
#define SLAM_CAMERA_CAMERA_FACTORY_H

#include "camera/base.hpp"
#include "camera/perspective.hpp"
#include "camera/fisheye.hpp"
#include "camera/equirectangular.hpp"
#include "camera/radial_division.hpp"

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {


namespace camera {

class camera_factory {
public:
    static camera::base* create(const cv::FileNode& node) {
        const auto camera_model_type = camera::base::load_model_type(node);

        camera::base* camera = nullptr;
        try {
            switch (camera_model_type) {
                case camera::model_type_t::Perspective: {
                    camera = new camera::perspective(node);
                    break;
                }
                case camera::model_type_t::Fisheye: {
                    camera = new camera::fisheye(node);
                    break;
                }
                case camera::model_type_t::Equirectangular: {
                    camera = new camera::equirectangular(node);
                    break;
                }
                case camera::model_type_t::RadialDivision: {
                    camera = new camera::radial_division(node);
                    break;
                }
            }
        }
        catch (const std::exception& e) {
            CV_LOG_DEBUG(&g_log_tag, "failed in loading camera model parameters: {}", e.what());
            if (camera) {
                delete camera;
                camera = nullptr;
            }
            throw;
        }

        assert(camera != nullptr);
        if (camera->setup_type_ == camera::setup_type_t::Stereo || camera->setup_type_ == camera::setup_type_t::RGBD) {
            if (camera->model_type_ == camera::model_type_t::Equirectangular) {
                throw std::runtime_error("Not implemented: Stereo or RGBD of equirectangular camera model");
            }
        }
        return camera;
    }
};

} // namespace camera
} // namespace cv::slam

#endif // SLAM_CAMERA_CAMERA_FACTORY_H
