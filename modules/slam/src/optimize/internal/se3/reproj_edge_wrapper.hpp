#ifndef SLAM_OPTIMIZE_G2O_SE3_REPROJ_EDGE_WRAPPER_H
#define SLAM_OPTIMIZE_G2O_SE3_REPROJ_EDGE_WRAPPER_H

#include "camera/perspective.hpp"
#include "camera/fisheye.hpp"
#include "camera/equirectangular.hpp"
#include "camera/radial_division.hpp"
#include "optimize/internal/se3/perspective_reproj_edge.hpp"
#include "optimize/internal/se3/equirectangular_reproj_edge.hpp"

#include <g2o/core/robust_kernel_impl.h>

#include <memory>

namespace cv::slam {

namespace data {
class landmark;
} // namespace data

namespace optimize {
namespace internal {
namespace se3 {

template<typename T>
class reproj_edge_wrapper {
public:
    reproj_edge_wrapper() = delete;

    reproj_edge_wrapper(const std::shared_ptr<T>& shot, shot_vertex* shot_vtx,
                        const std::shared_ptr<data::landmark>& lm, landmark_vertex* lm_vtx,
                        const unsigned int idx, const float obs_x, const float obs_y, const float obs_x_right,
                        const float inv_sigma_sq, const float sqrt_chi_sq, const bool use_huber_loss = true);

    virtual ~reproj_edge_wrapper() = default;

    bool is_inlier() const;

    bool is_outlier() const;

    void set_as_inlier() const;

    void set_as_outlier() const;

    bool depth_is_positive() const;

    g2o::OptimizableGraph::Edge* edge_;

    camera::base* camera_;
    std::shared_ptr<T> shot_;
    std::shared_ptr<data::landmark> lm_;
    const unsigned int idx_;
    const bool is_monocular_;
};

template<typename T>
inline reproj_edge_wrapper<T>::reproj_edge_wrapper(const std::shared_ptr<T>& shot, shot_vertex* shot_vtx,
                                                   const std::shared_ptr<data::landmark>& lm, landmark_vertex* lm_vtx,
                                                   const unsigned int idx, const float obs_x, const float obs_y, const float obs_x_right,
                                                   const float inv_sigma_sq, const float sqrt_chi_sq, const bool use_huber_loss)
    : camera_(shot->camera_), shot_(shot), lm_(lm), idx_(idx), is_monocular_(obs_x_right < 0) {

    switch (camera_->model_type_) {
        case camera::model_type_t::Perspective: {
            auto c = static_cast<camera::perspective*>(camera_);
            if (is_monocular_) {
                auto edge = new mono_perspective_reproj_edge();

                const Vec2_t obs{obs_x, obs_y};
                edge->setMeasurement(obs);
                edge->setInformation(Mat22_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;

                edge->setVertex(0, lm_vtx);
                edge->setVertex(1, shot_vtx);

                edge_ = edge;
            }
            else {
                auto edge = new stereo_perspective_reproj_edge();

                const Vec3_t obs{obs_x, obs_y, obs_x_right};
                edge->setMeasurement(obs);
                edge->setInformation(Mat33_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;
                edge->focal_x_baseline_ = camera_->focal_x_baseline_;

                edge->setVertex(0, lm_vtx);
                edge->setVertex(1, shot_vtx);

                edge_ = edge;
            }
            break;
        }
        case camera::model_type_t::Fisheye: {
            auto c = static_cast<camera::fisheye*>(camera_);
            if (is_monocular_) {
                auto edge = new mono_perspective_reproj_edge();

                const Vec2_t obs{obs_x, obs_y};
                edge->setMeasurement(obs);
                edge->setInformation(Mat22_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;

                edge->setVertex(0, lm_vtx);
                edge->setVertex(1, shot_vtx);

                edge_ = edge;
            }
            else {
                auto edge = new stereo_perspective_reproj_edge();

                const Vec3_t obs{obs_x, obs_y, obs_x_right};
                edge->setMeasurement(obs);
                edge->setInformation(Mat33_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;
                edge->focal_x_baseline_ = camera_->focal_x_baseline_;

                edge->setVertex(0, lm_vtx);
                edge->setVertex(1, shot_vtx);

                edge_ = edge;
            }
            break;
        }
        case camera::model_type_t::Equirectangular: {
            assert(is_monocular_);

            auto c = static_cast<camera::equirectangular*>(camera_);

            auto edge = new equirectangular_reproj_edge();

            const Vec2_t obs{obs_x, obs_y};
            edge->setMeasurement(obs);
            edge->setInformation(Mat22_t::Identity() * inv_sigma_sq);

            edge->cols_ = c->cols_;
            edge->rows_ = c->rows_;

            edge->setVertex(0, lm_vtx);
            edge->setVertex(1, shot_vtx);

            edge_ = edge;

            break;
        }
        case camera::model_type_t::RadialDivision: {
            auto c = static_cast<camera::radial_division*>(camera_);
            if (is_monocular_) {
                auto edge = new mono_perspective_reproj_edge();

                const Vec2_t obs{obs_x, obs_y};
                edge->setMeasurement(obs);
                edge->setInformation(Mat22_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;

                edge->setVertex(0, lm_vtx);
                edge->setVertex(1, shot_vtx);

                edge_ = edge;
            }
            else {
                auto edge = new stereo_perspective_reproj_edge();

                const Vec3_t obs{obs_x, obs_y, obs_x_right};
                edge->setMeasurement(obs);
                edge->setInformation(Mat33_t::Identity() * inv_sigma_sq);

                edge->fx_ = c->fx_;
                edge->fy_ = c->fy_;
                edge->cx_ = c->cx_;
                edge->cy_ = c->cy_;
                edge->focal_x_baseline_ = camera_->focal_x_baseline_;

                edge->setVertex(0, lm_vtx);
                edge->setVertex(1, shot_vtx);

                edge_ = edge;
            }
            break;
        }
    }


    if (use_huber_loss) {
        auto huber_kernel = new g2o::RobustKernelHuber();
        huber_kernel->setDelta(sqrt_chi_sq);
        edge_->setRobustKernel(huber_kernel);
    }
}

template<typename T>
inline bool reproj_edge_wrapper<T>::is_inlier() const {
    return edge_->level() == 0;
}

template<typename T>
inline bool reproj_edge_wrapper<T>::is_outlier() const {
    return edge_->level() != 0;
}

template<typename T>
inline void reproj_edge_wrapper<T>::set_as_inlier() const {
    edge_->setLevel(0);
}

template<typename T>
inline void reproj_edge_wrapper<T>::set_as_outlier() const {
    edge_->setLevel(1);
}

template<typename T>
inline bool reproj_edge_wrapper<T>::depth_is_positive() const {
    switch (camera_->model_type_) {
        case camera::model_type_t::Perspective: {
            if (is_monocular_) {
                return static_cast<mono_perspective_reproj_edge*>(edge_)->mono_perspective_reproj_edge::depth_is_positive();
            }
            else {
                return static_cast<stereo_perspective_reproj_edge*>(edge_)->stereo_perspective_reproj_edge::depth_is_positive();
            }
        }
        case camera::model_type_t::Fisheye: {
            if (is_monocular_) {
                return static_cast<mono_perspective_reproj_edge*>(edge_)->mono_perspective_reproj_edge::depth_is_positive();
            }
            else {
                return static_cast<stereo_perspective_reproj_edge*>(edge_)->stereo_perspective_reproj_edge::depth_is_positive();
            }
        }
        case camera::model_type_t::Equirectangular: {
            return true;
        }
        case camera::model_type_t::RadialDivision: {
            if (is_monocular_) {
                return static_cast<mono_perspective_reproj_edge*>(edge_)->mono_perspective_reproj_edge::depth_is_positive();
            }
            else {
                return static_cast<stereo_perspective_reproj_edge*>(edge_)->stereo_perspective_reproj_edge::depth_is_positive();
            }
        }
    }

    return true;
}

} // namespace se3
} // namespace internal
} // namespace optimize
} // namespace cv::slam

#endif // SLAM_OPTIMIZE_G2O_SE3_REPROJ_EDGE_WRAPPER_H
