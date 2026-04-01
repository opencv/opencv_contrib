#ifndef SLAM_OPTIMIZE_G2O_SE3_POSE_OPT_EDGE_WRAPPER_H
#define SLAM_OPTIMIZE_G2O_SE3_POSE_OPT_EDGE_WRAPPER_H

#include "camera/perspective.hpp"
#include "camera/fisheye.hpp"
#include "camera/equirectangular.hpp"
#include "camera/radial_division.hpp"
#include "optimize/internal/se3/perspective_pose_opt_edge.hpp"
#include "optimize/internal/se3/equirectangular_pose_opt_edge.hpp"

#include <g2o/core/robust_kernel_impl.h>

namespace cv::slam {

namespace data {
class landmark;
} // namespace data

namespace camera {
class base;
} // namespace camera

namespace optimize {
namespace internal {
namespace se3 {

class pose_opt_edge_wrapper {
public:
    pose_opt_edge_wrapper() = delete;

    pose_opt_edge_wrapper(const camera::base* camera, shot_vertex* shot_vtx, const Vec3_t& pos_w,
                          const unsigned int idx, const float obs_x, const float obs_y, const float obs_x_right,
                          const float inv_sigma_sq, const float sqrt_chi_sq);

    virtual ~pose_opt_edge_wrapper() = default;

    bool is_inlier() const;

    bool is_outlier() const;

    void set_as_inlier() const;

    void set_as_outlier() const;

    bool depth_is_positive() const;

    g2o::OptimizableGraph::Edge* edge_;

    const camera::base* camera_;
    const unsigned int idx_;
    const bool is_monocular_;
};

inline pose_opt_edge_wrapper::pose_opt_edge_wrapper(const camera::base* camera, shot_vertex* shot_vtx, const Vec3_t& pos_w,
                                                    const unsigned int idx, const float obs_x, const float obs_y, const float obs_x_right,
                                                    const float inv_sigma_sq, const float sqrt_chi_sq)
    : camera_(camera), idx_(idx), is_monocular_(obs_x_right < 0) {
    
    switch (camera_->model_type_) {
        case camera::model_type_t::Perspective: {
            const auto c = static_cast<const camera::perspective*>(camera_);
            if (is_monocular_) {
                auto edge = new mono_perspective_pose_opt_edge();

                const Vec2_t obs{obs_x, obs_y};
                edge->setMeasurement(obs);
                edge->setInformation(Mat22_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;

                edge->pos_w_ = pos_w;

                edge->setVertex(0, shot_vtx);

                edge_ = edge;
            }
            else {
                auto edge = new stereo_perspective_pose_opt_edge();

                const Vec3_t obs{obs_x, obs_y, obs_x_right};
                edge->setMeasurement(obs);
                edge->setInformation(Mat33_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;
                edge->focal_x_baseline_ = camera_->focal_x_baseline_;

                edge->pos_w_ = pos_w;

                edge->setVertex(0, shot_vtx);

                edge_ = edge;
            }
            break;
        }
        case camera::model_type_t::Fisheye: {
            const auto c = static_cast<const camera::fisheye*>(camera_);
            if (is_monocular_) {
                auto edge = new mono_perspective_pose_opt_edge();

                const Vec2_t obs{obs_x, obs_y};
                edge->setMeasurement(obs);
                edge->setInformation(Mat22_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;

                edge->pos_w_ = pos_w;

                edge->setVertex(0, shot_vtx);

                edge_ = edge;
            }
            else {
                auto edge = new stereo_perspective_pose_opt_edge();

                const Vec3_t obs{obs_x, obs_y, obs_x_right};
                edge->setMeasurement(obs);
                edge->setInformation(Mat33_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;
                edge->focal_x_baseline_ = camera_->focal_x_baseline_;

                edge->pos_w_ = pos_w;

                edge->setVertex(0, shot_vtx);

                edge_ = edge;
            }
            break;
        }
        case camera::model_type_t::Equirectangular: {
            assert(is_monocular_);

            const auto c = static_cast<const camera::equirectangular*>(camera_);

            auto edge = new equirectangular_pose_opt_edge();

            const Vec2_t obs{obs_x, obs_y};
            edge->setMeasurement(obs);
            edge->setInformation(Mat22_t::Identity() * inv_sigma_sq);

            edge->cols_ = c->cols_;
            edge->rows_ = c->rows_;

            edge->pos_w_ = pos_w;

            edge->setVertex(0, shot_vtx);

            edge_ = edge;

            break;
        }
        case camera::model_type_t::RadialDivision: {
            const auto c = static_cast<const camera::radial_division*>(camera_);
            if (is_monocular_) {
                auto edge = new mono_perspective_pose_opt_edge();

                const Vec2_t obs{obs_x, obs_y};
                edge->setMeasurement(obs);
                edge->setInformation(Mat22_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;

                edge->pos_w_ = pos_w;

                edge->setVertex(0, shot_vtx);

                edge_ = edge;
            }
            else {
                auto edge = new stereo_perspective_pose_opt_edge();

                const Vec3_t obs{obs_x, obs_y, obs_x_right};
                edge->setMeasurement(obs);
                edge->setInformation(Mat33_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;
                edge->focal_x_baseline_ = camera_->focal_x_baseline_;

                edge->pos_w_ = pos_w;

                edge->setVertex(0, shot_vtx);

                edge_ = edge;
            }
            break;
        }
    }

    
    auto huber_kernel = new g2o::RobustKernelHuber();
    huber_kernel->setDelta(sqrt_chi_sq);
    edge_->setRobustKernel(huber_kernel);
}

inline bool pose_opt_edge_wrapper::is_inlier() const {
    return edge_->level() == 0;
}

inline bool pose_opt_edge_wrapper::is_outlier() const {
    return edge_->level() != 0;
}

inline void pose_opt_edge_wrapper::set_as_inlier() const {
    edge_->setLevel(0);
}

inline void pose_opt_edge_wrapper::set_as_outlier() const {
    edge_->setLevel(1);
}

inline bool pose_opt_edge_wrapper::depth_is_positive() const {
    switch (camera_->model_type_) {
        case camera::model_type_t::Perspective: {
            if (is_monocular_) {
                return static_cast<mono_perspective_pose_opt_edge*>(edge_)->mono_perspective_pose_opt_edge::depth_is_positive();
            }
            else {
                return static_cast<stereo_perspective_pose_opt_edge*>(edge_)->stereo_perspective_pose_opt_edge::depth_is_positive();
            }
        }
        case camera::model_type_t::Fisheye: {
            if (is_monocular_) {
                return static_cast<mono_perspective_pose_opt_edge*>(edge_)->mono_perspective_pose_opt_edge::depth_is_positive();
            }
            else {
                return static_cast<stereo_perspective_pose_opt_edge*>(edge_)->stereo_perspective_pose_opt_edge::depth_is_positive();
            }
        }
        case camera::model_type_t::Equirectangular: {
            return true;
        }
        case camera::model_type_t::RadialDivision: {
            if (is_monocular_) {
                return static_cast<mono_perspective_pose_opt_edge*>(edge_)->mono_perspective_pose_opt_edge::depth_is_positive();
            }
            else {
                return static_cast<stereo_perspective_pose_opt_edge*>(edge_)->stereo_perspective_pose_opt_edge::depth_is_positive();
            }
        }
    }

    return true;
}

} // namespace se3
} // namespace internal
} // namespace optimize
} // namespace cv::slam

#endif // SLAM_OPTIMIZE_G2O_SE3_POSE_OPT_EDGE_WRAPPER_H
