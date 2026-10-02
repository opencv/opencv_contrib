#ifndef SLAM_OPTIMIZER_G2O_SE3_PERSPECTIVE_REPROJ_EDGE_H
#define SLAM_OPTIMIZER_G2O_SE3_PERSPECTIVE_REPROJ_EDGE_H

#include "type.hpp"
#include "optimize/internal/landmark_vertex.hpp"
#include "optimize/internal/se3/shot_vertex.hpp"

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>

namespace cv::slam {
namespace optimize {
namespace internal {
namespace se3 {

class mono_perspective_reproj_edge final : public g2o::BaseBinaryEdge<2, Vec2_t, landmark_vertex, shot_vertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    mono_perspective_reproj_edge();

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void computeError() override;

    void linearizeOplus() override;

    bool depth_is_positive() const;

    Vec2_t cam_project(const Vec3_t& pos_c) const;

    double fx_, fy_, cx_, cy_;
};

inline mono_perspective_reproj_edge::mono_perspective_reproj_edge()
    : g2o::BaseBinaryEdge<2, Vec2_t, landmark_vertex, shot_vertex>() {}

inline bool mono_perspective_reproj_edge::read(std::istream& is) {
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

inline bool mono_perspective_reproj_edge::write(std::ostream& os) const {
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

inline void mono_perspective_reproj_edge::computeError() {
    const auto v1 = static_cast<const shot_vertex*>(_vertices.at(1));
    const auto v2 = static_cast<const landmark_vertex*>(_vertices.at(0));
    const Vec2_t obs(_measurement);
    _error = obs - cam_project(v1->estimate().map(v2->estimate()));
}

inline void mono_perspective_reproj_edge::linearizeOplus() {
    auto vj = static_cast<shot_vertex*>(_vertices.at(1));
    const g2o::SE3Quat& cam_pose_cw = vj->shot_vertex::estimate();

    auto vi = static_cast<landmark_vertex*>(_vertices.at(0));
    const Vec3_t& pos_w = vi->landmark_vertex::estimate();
    const Vec3_t pos_c = cam_pose_cw.map(pos_w);

    const auto x = pos_c(0);
    const auto y = pos_c(1);
    const auto z = pos_c(2);
    const auto z_sq = z * z;

    const Mat33_t rot_cw = cam_pose_cw.rotation().toRotationMatrix();

    _jacobianOplusXi(0, 0) = -fx_ * rot_cw(0, 0) / z + fx_ * x * rot_cw(2, 0) / z_sq;
    _jacobianOplusXi(0, 1) = -fx_ * rot_cw(0, 1) / z + fx_ * x * rot_cw(2, 1) / z_sq;
    _jacobianOplusXi(0, 2) = -fx_ * rot_cw(0, 2) / z + fx_ * x * rot_cw(2, 2) / z_sq;

    _jacobianOplusXi(1, 0) = -fy_ * rot_cw(1, 0) / z + fy_ * y * rot_cw(2, 0) / z_sq;
    _jacobianOplusXi(1, 1) = -fy_ * rot_cw(1, 1) / z + fy_ * y * rot_cw(2, 1) / z_sq;
    _jacobianOplusXi(1, 2) = -fy_ * rot_cw(1, 2) / z + fy_ * y * rot_cw(2, 2) / z_sq;

    _jacobianOplusXj(0, 0) = x * y / z_sq * fx_;
    _jacobianOplusXj(0, 1) = -(1.0 + (x * x / z_sq)) * fx_;
    _jacobianOplusXj(0, 2) = y / z * fx_;
    _jacobianOplusXj(0, 3) = -1.0 / z * fx_;
    _jacobianOplusXj(0, 4) = 0.0;
    _jacobianOplusXj(0, 5) = x / z_sq * fx_;

    _jacobianOplusXj(1, 0) = (1.0 + y * y / z_sq) * fy_;
    _jacobianOplusXj(1, 1) = -x * y / z_sq * fy_;
    _jacobianOplusXj(1, 2) = -x / z * fy_;
    _jacobianOplusXj(1, 3) = 0.0;
    _jacobianOplusXj(1, 4) = -1.0 / z * fy_;
    _jacobianOplusXj(1, 5) = y / z_sq * fy_;
}

inline bool mono_perspective_reproj_edge::depth_is_positive() const {
    const auto v1 = static_cast<const shot_vertex*>(_vertices.at(1));
    const auto v2 = static_cast<const landmark_vertex*>(_vertices.at(0));
    return 0.0 < (v1->estimate().map(v2->estimate()))(2);
}

inline Vec2_t mono_perspective_reproj_edge::cam_project(const Vec3_t& pos_c) const {
    return {fx_ * pos_c(0) / pos_c(2) + cx_, fy_ * pos_c(1) / pos_c(2) + cy_};
}

class stereo_perspective_reproj_edge final : public g2o::BaseBinaryEdge<3, Vec3_t, landmark_vertex, shot_vertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    stereo_perspective_reproj_edge();

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void computeError() override;

    void linearizeOplus() override;

    bool depth_is_positive() const;

    Vec3_t cam_project(const Vec3_t& pos_c) const;

    double fx_, fy_, cx_, cy_, focal_x_baseline_;
};

inline stereo_perspective_reproj_edge::stereo_perspective_reproj_edge()
    : g2o::BaseBinaryEdge<3, Vec3_t, landmark_vertex, shot_vertex>() {}

inline bool stereo_perspective_reproj_edge::read(std::istream& is) {
    for (unsigned int i = 0; i < 3; ++i) {
        is >> _measurement(i);
    }
    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int j = i; j < 3; ++j) {
            is >> information()(i, j);
            if (i != j) {
                information()(j, i) = information()(i, j);
            }
        }
    }
    return true;
}

inline bool stereo_perspective_reproj_edge::write(std::ostream& os) const {
    for (unsigned int i = 0; i < 3; ++i) {
        os << measurement()(i) << " ";
    }
    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int j = i; j < 3; ++j) {
            os << " " << information()(i, j);
        }
    }
    return os.good();
}

inline void stereo_perspective_reproj_edge::computeError() {
    const auto v1 = static_cast<const shot_vertex*>(_vertices.at(1));
    const auto v2 = static_cast<const landmark_vertex*>(_vertices.at(0));
    const Vec3_t obs(_measurement);
    _error = obs - cam_project(v1->estimate().map(v2->estimate()));
}

inline void stereo_perspective_reproj_edge::linearizeOplus() {
    auto vj = static_cast<shot_vertex*>(_vertices.at(1));
    const g2o::SE3Quat& cam_pose_cw = vj->shot_vertex::estimate();

    auto vi = static_cast<landmark_vertex*>(_vertices.at(0));
    const Vec3_t& pos_w = vi->landmark_vertex::estimate();
    const Vec3_t pos_c = cam_pose_cw.map(pos_w);

    const auto x = pos_c(0);
    const auto y = pos_c(1);
    const auto z = pos_c(2);
    const auto z_sq = z * z;

    const Mat33_t rot_cw = cam_pose_cw.rotation().toRotationMatrix();

    _jacobianOplusXi(0, 0) = -fx_ * rot_cw(0, 0) / z + fx_ * x * rot_cw(2, 0) / z_sq;
    _jacobianOplusXi(0, 1) = -fx_ * rot_cw(0, 1) / z + fx_ * x * rot_cw(2, 1) / z_sq;
    _jacobianOplusXi(0, 2) = -fx_ * rot_cw(0, 2) / z + fx_ * x * rot_cw(2, 2) / z_sq;

    _jacobianOplusXi(1, 0) = -fy_ * rot_cw(1, 0) / z + fy_ * y * rot_cw(2, 0) / z_sq;
    _jacobianOplusXi(1, 1) = -fy_ * rot_cw(1, 1) / z + fy_ * y * rot_cw(2, 1) / z_sq;
    _jacobianOplusXi(1, 2) = -fy_ * rot_cw(1, 2) / z + fy_ * y * rot_cw(2, 2) / z_sq;

    _jacobianOplusXi(2, 0) = _jacobianOplusXi(0, 0) - focal_x_baseline_ * rot_cw(2, 0) / z_sq;
    _jacobianOplusXi(2, 1) = _jacobianOplusXi(0, 1) - focal_x_baseline_ * rot_cw(2, 1) / z_sq;
    _jacobianOplusXi(2, 2) = _jacobianOplusXi(0, 2) - focal_x_baseline_ * rot_cw(2, 2) / z_sq;

    _jacobianOplusXj(0, 0) = x * y / z_sq * fx_;
    _jacobianOplusXj(0, 1) = -(1.0 + (x * x / z_sq)) * fx_;
    _jacobianOplusXj(0, 2) = y / z * fx_;
    _jacobianOplusXj(0, 3) = -1.0 / z * fx_;
    _jacobianOplusXj(0, 4) = 0.0;
    _jacobianOplusXj(0, 5) = x / z_sq * fx_;

    _jacobianOplusXj(1, 0) = (1.0 + y * y / z_sq) * fy_;
    _jacobianOplusXj(1, 1) = -x * y / z_sq * fy_;
    _jacobianOplusXj(1, 2) = -x / z * fy_;
    _jacobianOplusXj(1, 3) = 0.0;
    _jacobianOplusXj(1, 4) = -1.0 / z * fy_;
    _jacobianOplusXj(1, 5) = y / z_sq * fy_;

    _jacobianOplusXj(2, 0) = _jacobianOplusXj(0, 0) - focal_x_baseline_ * y / z_sq;
    _jacobianOplusXj(2, 1) = _jacobianOplusXj(0, 1) + focal_x_baseline_ * x / z_sq;
    _jacobianOplusXj(2, 2) = _jacobianOplusXj(0, 2);
    _jacobianOplusXj(2, 3) = _jacobianOplusXj(0, 3);
    _jacobianOplusXj(2, 4) = 0;
    _jacobianOplusXj(2, 5) = _jacobianOplusXj(0, 5) - focal_x_baseline_ / z_sq;
}

inline bool stereo_perspective_reproj_edge::depth_is_positive() const {
    const auto v1 = static_cast<const shot_vertex*>(_vertices.at(1));
    const auto v2 = static_cast<const landmark_vertex*>(_vertices.at(0));
    return 0 < (v1->estimate().map(v2->estimate()))(2);
}

inline Vec3_t stereo_perspective_reproj_edge::cam_project(const Vec3_t& pos_c) const {
    const double reproj_x = fx_ * pos_c(0) / pos_c(2) + cx_;
    return {reproj_x, fy_ * pos_c(1) / pos_c(2) + cy_, reproj_x - focal_x_baseline_ / pos_c(2)};
}

} // namespace se3
} // namespace internal
} // namespace optimize
} // namespace cv::slam

#endif // SLAM_OPTIMIZER_G2O_SE3_PERSPECTIVE_REPROJ_EDGE_H
