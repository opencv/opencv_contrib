#ifndef SLAM_OPTIMIZE_G2O_BACKWARD_REPROJ_EDGE_H
#define SLAM_OPTIMIZE_G2O_BACKWARD_REPROJ_EDGE_H

#include "type.hpp"
#include "optimize/internal/sim3/transform_vertex.hpp"

#include <g2o/core/base_unary_edge.h>

namespace cv::slam {
namespace optimize {
namespace internal {
namespace sim3 {

class base_backward_reproj_edge : public g2o::BaseUnaryEdge<2, Vec2_t, transform_vertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    base_backward_reproj_edge();

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void computeError() final;

    virtual Vec2_t cam_project(const Vec3_t& pos_c) const = 0;

    Vec3_t pos_w_;
};

inline base_backward_reproj_edge::base_backward_reproj_edge()
    : g2o::BaseUnaryEdge<2, Vec2_t, transform_vertex>() {}

inline bool base_backward_reproj_edge::read(std::istream& is) {
    for (unsigned int i = 0; i < 2; ++i) {
        is >> _measurement(i);
    }
    for (int i = 0; i < information().rows(); ++i) {
        for (int j = i; j < information().cols(); ++j) {
            is >> information()(i, j);
            if (i != j) {
                information()(j, i) = information()(i, j);
            }
        }
    }
    return true;
}

inline bool base_backward_reproj_edge::write(std::ostream& os) const {
    for (unsigned int i = 0; i < 2; ++i) {
        os << measurement()(i) << " ";
    }
    for (int i = 0; i < information().rows(); ++i) {
        for (int j = i; j < information().cols(); ++j) {
            os << " " << information()(i, j);
        }
    }
    return os.good();
}

inline void base_backward_reproj_edge::computeError() {
    
    const auto v1 = static_cast<const transform_vertex*>(_vertices.at(0));
    const g2o::Sim3& Sim3_12 = v1->estimate();
    const g2o::Sim3 Sim3_21 = Sim3_12.inverse();
    
    const Mat33_t& rot_1w = v1->rot_1w_;
    const Vec3_t& trans_1w = v1->trans_1w_;

    
    const Vec3_t pos_1 = rot_1w * pos_w_ + trans_1w;
    
    const Vec3_t pos_2 = Sim3_21.map(pos_1);
    
    const Vec2_t obs(_measurement);
    _error = obs - cam_project(pos_2);
}

class perspective_backward_reproj_edge final : public base_backward_reproj_edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    perspective_backward_reproj_edge();

    Vec2_t cam_project(const Vec3_t& pos_c) const override;

    double fx_, fy_, cx_, cy_;
};

inline perspective_backward_reproj_edge::perspective_backward_reproj_edge()
    : base_backward_reproj_edge() {}

inline Vec2_t perspective_backward_reproj_edge::cam_project(const Vec3_t& pos_c) const {
    return {fx_ * pos_c(0) / pos_c(2) + cx_, fy_ * pos_c(1) / pos_c(2) + cy_};
}

class equirectangular_backward_reproj_edge final : public base_backward_reproj_edge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    equirectangular_backward_reproj_edge();

    Vec2_t cam_project(const Vec3_t& pos_c) const override;

    double cols_, rows_;
};

inline equirectangular_backward_reproj_edge::equirectangular_backward_reproj_edge()
    : base_backward_reproj_edge() {}

inline Vec2_t equirectangular_backward_reproj_edge::cam_project(const Vec3_t& pos_c) const {
    const double theta = std::atan2(pos_c(0), pos_c(2));
    const double phi = -std::asin(pos_c(1) / pos_c.norm());
    return {cols_ * (0.5 + theta / (2 * M_PI)), rows_ * (0.5 - phi / M_PI)};
}

} // namespace sim3
} // namespace internal
} // namespace optimize
} // namespace cv::slam

#endif // SLAM_OPTIMIZE_G2O_BACKWARD_REPROJ_EDGE_H
