#ifndef SLAM_OPTIMIZE_G2O_SIM3_MUTUAL_REPROJ_EDGE_WRAPPER_H
#define SLAM_OPTIMIZE_G2O_SIM3_MUTUAL_REPROJ_EDGE_WRAPPER_H

#include "camera/perspective.hpp"
#include "camera/fisheye.hpp"
#include "camera/equirectangular.hpp"
#include "camera/radial_division.hpp"
#include "data/landmark.hpp"
#include "optimize/internal/sim3/forward_reproj_edge.hpp"
#include "optimize/internal/sim3/backward_reproj_edge.hpp"

#include <g2o/core/robust_kernel_impl.h>

#include <memory>

namespace cv::slam {

namespace data {
class landmark;
} // namespace data

namespace optimize {
namespace internal {
namespace sim3 {

template<typename T>
class mutual_reproj_edge_wapper {
public:
    mutual_reproj_edge_wapper() = delete;

    mutual_reproj_edge_wapper(const std::shared_ptr<T>& shot1, unsigned int idx1, const std::shared_ptr<data::landmark>& lm1,
                              const std::shared_ptr<T>& shot2, unsigned int idx2, const std::shared_ptr<data::landmark>& lm2,
                              internal::sim3::transform_vertex* Sim3_12_vtx, const float sqrt_chi_sq);

    bool is_inlier() const;

    bool is_outlier() const;

    void set_as_inlier() const;

    void set_as_outlier() const;



    base_forward_reproj_edge* edge_12_;


    base_backward_reproj_edge* edge_21_;

    std::shared_ptr<T> shot1_, shot2_;
    unsigned int idx1_, idx2_;
    std::shared_ptr<data::landmark> lm1_, lm2_;
};

template<typename T>
inline mutual_reproj_edge_wapper<T>::mutual_reproj_edge_wapper(const std::shared_ptr<T>& shot1, unsigned int idx1, const std::shared_ptr<data::landmark>& lm1,
                                                               const std::shared_ptr<T>& shot2, unsigned int idx2, const std::shared_ptr<data::landmark>& lm2,
                                                               internal::sim3::transform_vertex* Sim3_12_vtx, const float sqrt_chi_sq)
    : shot1_(shot1), shot2_(shot2), idx1_(idx1), idx2_(idx2), lm1_(lm1), lm2_(lm2) {

    {
        camera::base* camera1 = shot1->camera_;

        switch (camera1->model_type_) {
            case camera::model_type_t::Perspective: {
                auto c = static_cast<camera::perspective*>(camera1);


                auto edge_12 = new internal::sim3::perspective_forward_reproj_edge();

                const auto& undist_keypt_1 = shot1->frm_obs_.undist_keypts_.at(idx1);
                const Vec2_t obs_1{undist_keypt_1.pt.x, undist_keypt_1.pt.y};
                const float inv_sigma_sq_1 = shot1->orb_params_->inv_level_sigma_sq_.at(undist_keypt_1.octave);
                edge_12->setMeasurement(obs_1);
                edge_12->setInformation(Mat22_t::Identity() * inv_sigma_sq_1);

                edge_12->pos_w_ = lm2->get_pos_in_world();

                edge_12->fx_ = c->fx_;
                edge_12->fy_ = c->fy_;
                edge_12->cx_ = c->cx_;
                edge_12->cy_ = c->cy_;

                edge_12->setVertex(0, Sim3_12_vtx);

                edge_12_ = edge_12;
                break;
            }
            case camera::model_type_t::Fisheye: {
                auto c = static_cast<camera::fisheye*>(camera1);


                auto edge_12 = new internal::sim3::perspective_forward_reproj_edge();

                const auto& undist_keypt_1 = shot1->frm_obs_.undist_keypts_.at(idx1);
                const Vec2_t obs_1{undist_keypt_1.pt.x, undist_keypt_1.pt.y};
                const float inv_sigma_sq_1 = shot1->orb_params_->inv_level_sigma_sq_.at(undist_keypt_1.octave);
                edge_12->setMeasurement(obs_1);
                edge_12->setInformation(Mat22_t::Identity() * inv_sigma_sq_1);

                edge_12->pos_w_ = lm2->get_pos_in_world();

                edge_12->fx_ = c->fx_;
                edge_12->fy_ = c->fy_;
                edge_12->cx_ = c->cx_;
                edge_12->cy_ = c->cy_;

                edge_12->setVertex(0, Sim3_12_vtx);

                edge_12_ = edge_12;
                break;
            }
            case camera::model_type_t::Equirectangular: {
                auto c = static_cast<camera::equirectangular*>(camera1);


                auto edge_12 = new internal::sim3::equirectangular_forward_reproj_edge();

                const auto& undist_keypt_1 = shot1->frm_obs_.undist_keypts_.at(idx1);
                const Vec2_t obs_1{undist_keypt_1.pt.x, undist_keypt_1.pt.y};
                const float inv_sigma_sq_1 = shot1->orb_params_->inv_level_sigma_sq_.at(undist_keypt_1.octave);
                edge_12->setMeasurement(obs_1);
                edge_12->setInformation(Mat22_t::Identity() * inv_sigma_sq_1);

                edge_12->pos_w_ = lm2->get_pos_in_world();

                edge_12->cols_ = c->cols_;
                edge_12->rows_ = c->rows_;

                edge_12->setVertex(0, Sim3_12_vtx);

                edge_12_ = edge_12;
                break;
            }
            case camera::model_type_t::RadialDivision: {
                auto c = static_cast<camera::radial_division*>(camera1);


                auto edge_12 = new internal::sim3::perspective_forward_reproj_edge();

                const auto& undist_keypt_1 = shot1->frm_obs_.undist_keypts_.at(idx1);
                const Vec2_t obs_1{undist_keypt_1.pt.x, undist_keypt_1.pt.y};
                const float inv_sigma_sq_1 = shot1->orb_params_->inv_level_sigma_sq_.at(undist_keypt_1.octave);
                edge_12->setMeasurement(obs_1);
                edge_12->setInformation(Mat22_t::Identity() * inv_sigma_sq_1);

                edge_12->pos_w_ = lm2->get_pos_in_world();

                edge_12->fx_ = c->fx_;
                edge_12->fy_ = c->fy_;
                edge_12->cx_ = c->cx_;
                edge_12->cy_ = c->cy_;

                edge_12->setVertex(0, Sim3_12_vtx);

                edge_12_ = edge_12;
                break;
            }
        }


        auto huber_kernel_12 = new g2o::RobustKernelHuber();
        huber_kernel_12->setDelta(sqrt_chi_sq);
        edge_12_->setRobustKernel(huber_kernel_12);
    }


    {
        camera::base* camera2 = shot2->camera_;

        switch (camera2->model_type_) {
            case camera::model_type_t::Perspective: {
                auto c = static_cast<camera::perspective*>(camera2);


                auto edge_21 = new internal::sim3::perspective_backward_reproj_edge();

                const auto& undist_keypt_2 = shot2->frm_obs_.undist_keypts_.at(idx2);
                const Vec2_t obs_2{undist_keypt_2.pt.x, undist_keypt_2.pt.y};
                const float inv_sigma_sq_2 = shot2->orb_params_->inv_level_sigma_sq_.at(undist_keypt_2.octave);
                edge_21->setMeasurement(obs_2);
                edge_21->setInformation(Mat22_t::Identity() * inv_sigma_sq_2);

                edge_21->pos_w_ = lm1->get_pos_in_world();

                edge_21->fx_ = c->fx_;
                edge_21->fy_ = c->fy_;
                edge_21->cx_ = c->cx_;
                edge_21->cy_ = c->cy_;

                edge_21->setVertex(0, Sim3_12_vtx);

                edge_21_ = edge_21;
                break;
            }
            case camera::model_type_t::Fisheye: {
                auto c = static_cast<camera::fisheye*>(camera2);


                auto edge_21 = new internal::sim3::perspective_backward_reproj_edge();

                const auto& undist_keypt_2 = shot2->frm_obs_.undist_keypts_.at(idx2);
                const Vec2_t obs_2{undist_keypt_2.pt.x, undist_keypt_2.pt.y};
                const float inv_sigma_sq_2 = shot2->orb_params_->inv_level_sigma_sq_.at(undist_keypt_2.octave);
                edge_21->setMeasurement(obs_2);
                edge_21->setInformation(Mat22_t::Identity() * inv_sigma_sq_2);

                edge_21->pos_w_ = lm1->get_pos_in_world();

                edge_21->fx_ = c->fx_;
                edge_21->fy_ = c->fy_;
                edge_21->cx_ = c->cx_;
                edge_21->cy_ = c->cy_;

                edge_21->setVertex(0, Sim3_12_vtx);

                edge_21_ = edge_21;
                break;
            }
            case camera::model_type_t::Equirectangular: {
                auto c = static_cast<camera::equirectangular*>(camera2);


                auto edge_21 = new internal::sim3::equirectangular_backward_reproj_edge();

                const auto& undist_keypt_2 = shot2->frm_obs_.undist_keypts_.at(idx2);
                const Vec2_t obs_2{undist_keypt_2.pt.x, undist_keypt_2.pt.y};
                const float inv_sigma_sq_2 = shot2->orb_params_->inv_level_sigma_sq_.at(undist_keypt_2.octave);
                edge_21->setMeasurement(obs_2);
                edge_21->setInformation(Mat22_t::Identity() * inv_sigma_sq_2);

                edge_21->pos_w_ = lm1->get_pos_in_world();

                edge_21->cols_ = c->cols_;
                edge_21->rows_ = c->rows_;

                edge_21->setVertex(0, Sim3_12_vtx);

                edge_21_ = edge_21;
                break;
            }
            case camera::model_type_t::RadialDivision: {
                auto c = static_cast<camera::radial_division*>(camera2);


                auto edge_21 = new internal::sim3::perspective_backward_reproj_edge();

                const auto& undist_keypt_2 = shot2->frm_obs_.undist_keypts_.at(idx2);
                const Vec2_t obs_2{undist_keypt_2.pt.x, undist_keypt_2.pt.y};
                const float inv_sigma_sq_2 = shot2->orb_params_->inv_level_sigma_sq_.at(undist_keypt_2.octave);
                edge_21->setMeasurement(obs_2);
                edge_21->setInformation(Mat22_t::Identity() * inv_sigma_sq_2);

                edge_21->pos_w_ = lm1->get_pos_in_world();

                edge_21->fx_ = c->fx_;
                edge_21->fy_ = c->fy_;
                edge_21->cx_ = c->cx_;
                edge_21->cy_ = c->cy_;

                edge_21->setVertex(0, Sim3_12_vtx);

                edge_21_ = edge_21;
                break;
            }
        }


        auto huber_kernel_21 = new g2o::RobustKernelHuber();
        huber_kernel_21->setDelta(sqrt_chi_sq);
        edge_21_->setRobustKernel(huber_kernel_21);
    }
}

template<typename T>
inline bool mutual_reproj_edge_wapper<T>::is_inlier() const {
    return edge_12_->level() == 0 && edge_21_->level() == 0;
}

template<typename T>
inline bool mutual_reproj_edge_wapper<T>::is_outlier() const {
    return !is_inlier();
}

template<typename T>
inline void mutual_reproj_edge_wapper<T>::set_as_inlier() const {
    edge_12_->setLevel(0);
    edge_21_->setLevel(0);
}

template<typename T>
inline void mutual_reproj_edge_wapper<T>::set_as_outlier() const {
    edge_12_->setLevel(1);
    edge_21_->setLevel(1);
}

} // namespace sim3
} // namespace internal
} // namespace optimize
} // namespace cv::slam

#endif // SLAM_OPTIMIZE_G2O_SIM3_MUTUAL_REPROJ_EDGE_WRAPPER_H
